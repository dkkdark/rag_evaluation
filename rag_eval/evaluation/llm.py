from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, Sequence

from rag_eval.core.models import LLMCallResult, LLMConfig


def get_openai_client(llm_config: LLMConfig):
    if not llm_config.enabled:
        return None
    api_key = os.environ.get(llm_config.api_key_env, "").strip()
    if not api_key:
        return None

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def context_text_from_rows(retrieved: Sequence[Dict], *, style: str = "standard") -> str:
    parts = []
    for index, row in enumerate(retrieved, start=1):
        citation = f"[{index}]"
        metadata = " | ".join(
            str(row.get(key, ""))
            for key in ["doc_id", "section_id", "title"]
            if str(row.get(key, "")).strip()
        )
        header = str(row.get("kg_context_header", "")).strip()
        if header:
            metadata = f"{metadata} | {header}" if metadata else header
        text = str(row.get("text", ""))
        if style == "minimal":
            parts.append(f"{citation} {text}")
        elif style == "cite_first":
            parts.append(f"{citation} {metadata}\n{text}")
        else:
            parts.append(f"{citation} {metadata}\n{text}")
    return "\n\n".join(part for part in parts if part.strip())


def augment_query_with_llm(
    question: str,
    llm_config: LLMConfig,
    *,
    mode: str = "none",
    max_terms: int = 8,
) -> LLMCallResult:
    if mode == "none":
        return LLMCallResult(answer=question, used=False, status="disabled", error=None)
    client = get_openai_client(llm_config)
    if not llm_config.enabled:
        return LLMCallResult(answer=question, used=False, status="llm_disabled", error=None)
    if client is None:
        return LLMCallResult(answer=question, used=False, status="missing_api_key", error=None)
    if mode == "hyde":
        system_prompt = (
            "Generate a short hypothetical source passage for dense retrieval. "
            "It should describe the kind of real document passage that would answer the user's question. "
            "Keep it concise, preserve the user's language when possible, and avoid adding exact facts you cannot infer."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            "Return only the hypothetical passage. It should be useful as a retrieval query, not as a final answer."
        )
    else:
        system_prompt = (
            "Expand the user's retrieval query for a RAG system. "
            "Keep the original intent, do not answer the question, and do not add unsupported facts. "
            "Return one concise query containing useful synonyms and legal/technical terms."
        )
        user_prompt = (
            f"Original query:\n{question}\n\n"
            f"Add at most {max_terms} helpful search terms or paraphrases. "
            "Return only the augmented query."
        )

    try:
        response = client.responses.create(
            model=llm_config.model,
            temperature=0.0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        augmented = (getattr(response, "output_text", "") or "").strip()
        if not augmented:
            return LLMCallResult(answer=question, used=True, status="empty_response", error=None)
        return LLMCallResult(answer=augmented, used=True, status="success", error=None)
    except Exception as exc:
        return LLMCallResult(answer=question, used=True, status="error", error=str(exc))


def rewrite_query_for_retry(
    question: str,
    retrieved: Sequence[Dict],
    llm_config: LLMConfig,
    *,
    reason: str = "",
) -> LLMCallResult:
    client = get_openai_client(llm_config)
    if not llm_config.enabled:
        return LLMCallResult(answer=question, used=False, status="llm_disabled", error=None)
    if client is None:
        return LLMCallResult(answer=question, used=False, status="missing_api_key", error=None)
    weak_context = context_text_from_rows(retrieved[:3], style="minimal") if retrieved else "No retrieved context."
    try:
        response = client.responses.create(
            model=llm_config.model,
            temperature=0.0,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite a retrieval query after a weak RAG retrieval attempt. "
                        "Preserve the user's intent, add missing terminology, and avoid answering the question."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Weak retrieval reason:\n{reason}\n\n"
                        f"Current weak context:\n{weak_context}\n\n"
                        "Return one improved retrieval query only."
                    ),
                },
            ],
        )
        rewritten = (getattr(response, "output_text", "") or "").strip()
        if not rewritten:
            return LLMCallResult(answer=question, used=True, status="empty_response", error=None)
        return LLMCallResult(answer=rewritten, used=True, status="success", error=None)
    except Exception as exc:
        return LLMCallResult(answer=question, used=True, status="error", error=str(exc))


def extract_json_object(text: str) -> Dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def critique_and_revise_answer_with_llm(
    *,
    question: str,
    answer: str,
    retrieved: Sequence[Dict],
    llm_config: LLMConfig,
) -> LLMCallResult:
    client = get_openai_client(llm_config)
    if not llm_config.enabled:
        return LLMCallResult(answer=answer, used=False, status="llm_disabled", error=None)
    if client is None:
        return LLMCallResult(answer=answer, used=False, status="missing_api_key", error=None)
    if not retrieved or not answer.strip():
        return LLMCallResult(answer=answer, used=False, status="no_context_or_answer", error=None)

    context_text = context_text_from_rows(retrieved, style="cite_first")
    try:
        response = client.responses.create(
            model=llm_config.model,
            temperature=0.0,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict RAG self-critique step. Check whether the answer is fully supported "
                        "by the cited context. If any claim is unsupported, revise the answer so every remaining "
                        "claim is supported. If the context is insufficient, return a brief refusal. "
                        "Return JSON with keys status, critique, revised_answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Retrieved context:\n{context_text}\n\n"
                        f"Draft answer:\n{answer}\n\n"
                        "JSON only."
                    ),
                },
            ],
        )
        raw = (getattr(response, "output_text", "") or "").strip()
        if not raw:
            return LLMCallResult(answer=answer, used=True, status="empty_response", error=None)
        payload = extract_json_object(raw)
        revised = str(payload.get("revised_answer") or "").strip()
        status = str(payload.get("status") or "success").strip() or "success"
        if not revised:
            revised = answer
            status = "empty_revised_answer"
        critique = str(payload.get("critique") or "").strip()
        return LLMCallResult(answer=revised, used=True, status=status, error=critique or None)
    except Exception as exc:
        return LLMCallResult(answer=answer, used=True, status="error", error=str(exc))


def generate_answer_with_llm(
    question: str,
    retrieved: Sequence[Dict],
    llm_config: LLMConfig,
    *,
    answer_mode: str = "grounded_llm",
    context_style: str = "standard",
) -> LLMCallResult:
    client = get_openai_client(llm_config)
    if not llm_config.enabled:
        return LLMCallResult(answer=None, used=False, status="disabled", error=None)
    if client is None:
        api_key = os.environ.get(llm_config.api_key_env, "").strip()
        if not api_key:
            return LLMCallResult(
                answer=None,
                used=False,
                status="missing_api_key",
                error=f"Environment variable {llm_config.api_key_env} is empty or unset.",
            )
        return LLMCallResult(answer=None, used=False, status="client_unavailable", error=None)
    if not retrieved:
        return LLMCallResult(answer=None, used=False, status="no_context", error=None)

    context_text = context_text_from_rows(retrieved, style=context_style)
    if answer_mode == "cite_first":
        answer_instruction = "Answer concisely and cite supporting chunk numbers like [1]."
    elif answer_mode == "claim_checklist":
        answer_instruction = (
            "Answer as a checklist of required factual claims. "
            "Each bullet must be supported by the context and include chunk citations like [1]."
        )
    else:
        answer_instruction = "Answer the question using only this context."
    try:
        print(
            f"[LLM] question={question!r} model={llm_config.model} temperature={llm_config.temperature} status=request_started",
            file=sys.stderr,
        )
        response = client.responses.create(
            model=llm_config.model,
            temperature=llm_config.temperature,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the provided context chunks. "
                        "Do not invent facts. If the context is insufficient, say so briefly. "
                        "Prefer a concise factual answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Retrieved context:\n{context_text}\n\n"
                        f"{answer_instruction}"
                    ),
                },
            ],
        )
        answer = getattr(response, "output_text", "") or ""
        answer = answer.strip()
        if not answer:
            print(
                f"[LLM] question={question!r} model={llm_config.model} temperature={llm_config.temperature} status=empty_response",
                file=sys.stderr,
            )
            return LLMCallResult(
                answer=None,
                used=True,
                status="empty_response",
                error="LLM returned an empty response.",
            )
        print(
            f"[LLM] question={question!r} model={llm_config.model} temperature={llm_config.temperature} status=success",
            file=sys.stderr,
        )
        return LLMCallResult(answer=answer, used=True, status="success", error=None)
    except Exception as exc:
        print(
            f"[LLM] question={question!r} model={llm_config.model} temperature={llm_config.temperature} status=error error={exc}",
            file=sys.stderr,
        )
        return LLMCallResult(answer=None, used=True, status="error", error=str(exc))
