from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Sequence

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


LLM_QUERY_AUGMENTATION_MODES = frozenset({"llm", "hyde"})
STRUCTURED_QUERY_AUGMENTATION_MODES = frozenset({"translate_en"})
QUERY_AUGMENTATION_MODES = LLM_QUERY_AUGMENTATION_MODES | STRUCTURED_QUERY_AUGMENTATION_MODES


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
    if mode == "translate_en":
        return LLMCallResult(
            answer=question,
            used=False,
            status="use_structured_enrichment",
            error="translate_en is handled by structured query enrichment, not augment_query_with_llm.",
        )
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
        success_status = "success"
        append_original = False
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
        success_status = "success"
        append_original = False

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
        if append_original:
            augmented = f"{question} {augmented}".strip()
        return LLMCallResult(answer=augmented, used=True, status=success_status, error=None)
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


def judge_cpv_candidates_with_llm(
    *,
    question: str,
    rows: Sequence[Dict],
    llm_config: LLMConfig,
    top_k: int,
    judge_top_n: int = 3,
    judge_weight: float = 0.55,
) -> tuple[List[Dict], LLMCallResult]:
    client = get_openai_client(llm_config)
    if not llm_config.enabled:
        return list(rows[:top_k]), LLMCallResult(answer=None, used=False, status="llm_disabled", error=None)
    if client is None:
        return list(rows[:top_k]), LLMCallResult(answer=None, used=False, status="missing_api_key", error=None)
    if top_k <= 0 or not rows:
        return [], LLMCallResult(answer=None, used=False, status="no_candidates", error=None)

    limited_top_n = min(len(rows), max(2, judge_top_n))
    head = [dict(row) for row in rows[:limited_top_n]]
    tail = [dict(row) for row in rows[limited_top_n:]]

    candidate_lines = []
    for index, row in enumerate(head, start=1):
        candidate_lines.append(
            (
                f"{index}. code={row.get('cpv_code', '')}\n"
                f"label={row.get('cpv_label', row.get('title', ''))}\n"
                f"description={row.get('description_en', '')}\n"
                f"procurement_type={row.get('procurement_type', '')}\n"
                f"parent_label={row.get('cpv_parent_label', '')}\n"
                f"children_labels={row.get('children_labels', '')}\n"
                f"sibling_labels={row.get('sibling_labels', '')}\n"
                f"use_when={row.get('use_when_text', '')}\n"
                f"do_not_use_when={row.get('do_not_use_when_text', '')}\n"
                f"common_tender_phrases={row.get('common_tender_phrases', '')}\n"
                f"current_score={row.get('score', 0.0)}"
            )
        )
    candidates_text = "\n\n".join(candidate_lines)

    try:
        response = client.responses.create(
            model=llm_config.model,
            temperature=0.0,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a final CPV selection judge for public-procurement classification. "
                        "Choose the single best CPV code from the provided shortlist only. "
                        "Prefer the most specific valid code that still matches the query. "
                        "Avoid broader parent-like codes when a more precise child candidate clearly fits. "
                        "Pay attention to goods vs services vs installation vs maintenance distinctions. "
                        "Compare the shortlist contrastively: explain why the selected code is better than the closest alternative in the same branch. "
                        "Return JSON only with keys selected_code, ranking, and rationale. "
                        "selected_code must be one of the provided codes. "
                        "ranking should order only the provided codes from best to worst."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Procurement query:\n{question}\n\n"
                        f"Candidate shortlist:\n{candidates_text}\n\n"
                        "Return JSON only."
                    ),
                },
            ],
        )
        raw = (getattr(response, "output_text", "") or "").strip()
        if not raw:
            return list(rows[:top_k]), LLMCallResult(answer=None, used=True, status="empty_response", error=None)
        payload = extract_json_object(raw)
        selected_code = str(payload.get("selected_code") or "").strip()
        ranking_raw = payload.get("ranking")
        ranking_list = ranking_raw if isinstance(ranking_raw, list) else []
        requested_codes = [str(code).strip() for code in ranking_list if str(code).strip()]

        by_code = {str(row.get("cpv_code", "")).strip(): dict(row) for row in head}
        if not selected_code or selected_code not in by_code:
            return list(rows[:top_k]), LLMCallResult(answer=raw, used=True, status="invalid_selected_code", error=raw[:500])

        ordered_codes = []
        seen = set()
        if selected_code:
            ordered_codes.append(selected_code)
            seen.add(selected_code)
        for code in requested_codes:
            if code in by_code and code not in seen:
                ordered_codes.append(code)
                seen.add(code)
        for row in head:
            code = str(row.get("cpv_code", "")).strip()
            if code and code not in seen:
                ordered_codes.append(code)
                seen.add(code)

        base_scores = [float(row.get("score") or 0.0) for row in head]
        low = min(base_scores) if base_scores else 0.0
        high = max(base_scores) if base_scores else 0.0

        def normalize_base_score(value: float) -> float:
            if abs(high - low) < 1e-12:
                return 1.0 if high > 0 else 0.0
            return (value - low) / (high - low)

        reranked_head: List[Dict] = []
        for rank_index, code in enumerate(ordered_codes, start=1):
            row = by_code[code]
            base_score = float(row.get("score") or 0.0)
            base_norm = normalize_base_score(base_score)
            judge_rank_score = float(limited_top_n - rank_index) / float(max(limited_top_n - 1, 1))
            final_score = (1.0 - judge_weight) * base_norm + judge_weight * judge_rank_score
            row["base_score_before_llm_judge"] = base_score
            row["base_score_before_llm_judge_normalized"] = base_norm
            row["llm_judge_selected"] = 1 if code == selected_code else 0
            row["llm_judge_rank"] = rank_index
            row["llm_judge_score"] = judge_rank_score
            row["score"] = final_score
            row["reranked"] = True
            row["reranker"] = "llm_judge"
            reranked_head.append(row)

        reranked_head.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        combined = reranked_head + tail
        rationale = str(payload.get("rationale") or "").strip()
        return combined[:top_k], LLMCallResult(answer=raw, used=True, status="success", error=rationale or None)
    except Exception as exc:
        return list(rows[:top_k]), LLMCallResult(answer=None, used=True, status="error", error=str(exc))


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
