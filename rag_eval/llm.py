from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Sequence

from rag_eval.models import LLMCallResult, LLMConfig


def get_openai_client(llm_config: LLMConfig):
    if not llm_config.enabled:
        return None
    api_key = os.environ.get(llm_config.api_key_env, "").strip()
    if not api_key:
        return None

    from openai import OpenAI

    return OpenAI(
        api_key=api_key, 
        #base_url="https://chat-ai.academiccloud.de/v1"
    )


def parse_json_response(text: str) -> Dict:
    import json

    return json.loads(text.strip())


def call_llm_json(
    client: Any,
    *,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: Dict,
) -> Dict:
    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )
    response_text = getattr(response, "output_text", "") or ""
    if not response_text:
        raise ValueError("LLM returned an empty response.")
    return parse_json_response(response_text)


def generate_answer_with_llm(
    question: str,
    retrieved: Sequence[Dict],
    llm_config: LLMConfig,
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

    context_text = "\n\n".join(
        f"[{row['section_id']}] {row['title']}\n{row['text']}" for row in retrieved
    )
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
                        "Answer the question using only this context."
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
