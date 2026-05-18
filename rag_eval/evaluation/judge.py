from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Sequence

from rag_eval.evaluation.metrics import (
    classify_claim_diagnostic,
    compute_attribution_metrics,
    harmonic_mean,
    reference_claims_from_item,
    safe_ratio,
    split_claims,
)
from rag_eval.core.models import ClaimJudgeResult, LLMConfig


def get_openai_client(config: LLMConfig):
    if not config.enabled:
        return None
    api_key = os.environ.get(config.api_key_env, "").strip()
    if not api_key:
        return None

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def extract_json_object(text: str) -> Dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_label(value: object) -> str:
    label = str(value or "").strip().casefold()
    aliases = {
        "supported": "supported",
        "entails": "supported",
        "entailed": "supported",
        "entailment": "supported",
        "present": "supported",
        "contradicted": "contradicted",
        "contradiction": "contradicted",
        "conflict": "contradicted",
        "not_enough_info": "not_enough_info",
        "nei": "not_enough_info",
        "unsupported": "not_enough_info",
        "missing": "not_enough_info",
    }
    return aliases.get(label, "not_enough_info")


def normalize_supporting_chunk_ids(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def valid_supporting_chunk_ids(ids: Sequence[str], valid_chunk_ids: set[str] | None) -> List[str]:
    if valid_chunk_ids is None:
        return list(ids)
    return [chunk_id for chunk_id in ids if chunk_id in valid_chunk_ids]


def indexed_context(retrieved: Sequence[Dict]) -> str:
    parts = []
    for rank, row in enumerate(retrieved, start=1):
        parts.append(
            "\n".join(
                [
                    f"chunk_id: {row.get('chunk_id', f'rank_{rank}')}",
                    f"rank: {rank}",
                    f"section: {row.get('section_id', '')} {row.get('title', '')}",
                    f"text: {row.get('text', '')}",
                ]
            )
        )
    return "\n\n".join(parts)


def metrics_from_judge_payload(
    payload: Dict,
    fallback_gold_claims: Sequence[str],
    fallback_answer_claims: Sequence[str],
    valid_chunk_ids: set[str] | None = None,
) -> Dict[str, object]:
    gold_rows = payload.get("gold_claims") or []
    answer_rows = payload.get("answer_claims") or []

    if not isinstance(gold_rows, list):
        gold_rows = []
    if not isinstance(answer_rows, list):
        answer_rows = []

    gold_claims = [
        str(row.get("claim", "")).strip()
        for row in gold_rows
        if isinstance(row, dict) and str(row.get("claim", "")).strip()
    ] or list(fallback_gold_claims)
    answer_claims = [
        str(row.get("claim", "")).strip()
        for row in answer_rows
        if isinstance(row, dict) and str(row.get("claim", "")).strip()
    ] or list(fallback_answer_claims)

    gold_supported_by_context = 0
    gold_present_in_answer = 0
    claim_evidence_map: List[Dict[str, object]] = []
    for row in gold_rows:
        if not isinstance(row, dict):
            continue
        context_nli = normalize_label(row.get("context_nli"))
        answer_nli = normalize_label(row.get("answer_nli"))
        raw_ids = normalize_supporting_chunk_ids(row.get("supporting_chunk_ids"))
        valid_ids = valid_supporting_chunk_ids(raw_ids, valid_chunk_ids)
        if context_nli == "supported":
            gold_supported_by_context += 1
        if answer_nli == "supported":
            gold_present_in_answer += 1
        claim_evidence_map.append(
            {
                "claim_type": "gold",
                "claim": str(row.get("claim", "")).strip(),
                "context_nli": context_nli,
                "answer_nli": answer_nli,
                "supporting_chunk_ids": valid_ids,
                "raw_supporting_chunk_ids": raw_ids,
                "invalid_attribution_count": len(raw_ids) - len(valid_ids),
            }
        )

    answer_supported_by_gold = 0
    answer_supported_by_context = 0
    answer_contradicted_by_context = 0
    for row in answer_rows:
        if not isinstance(row, dict):
            continue
        gold_label = normalize_label(row.get("gold_nli"))
        context_label = normalize_label(row.get("context_nli"))
        raw_ids = normalize_supporting_chunk_ids(row.get("supporting_chunk_ids"))
        valid_ids = valid_supporting_chunk_ids(raw_ids, valid_chunk_ids)
        if gold_label == "supported":
            answer_supported_by_gold += 1
        if context_label == "supported":
            answer_supported_by_context += 1
        if context_label == "contradicted":
            answer_contradicted_by_context += 1
        claim_evidence_map.append(
            {
                "claim_type": "answer",
                "claim": str(row.get("claim", "")).strip(),
                "context_nli": context_label,
                "gold_nli": gold_label,
                "supporting_chunk_ids": valid_ids,
                "raw_supporting_chunk_ids": raw_ids,
                "invalid_attribution_count": len(raw_ids) - len(valid_ids),
            }
        )

    gold_count = len(gold_claims)
    answer_count = len(answer_claims)
    context_claim_recall = safe_ratio(gold_supported_by_context, gold_count)
    answer_claim_recall = safe_ratio(gold_present_in_answer, gold_count)
    answer_claim_precision = safe_ratio(answer_supported_by_gold, answer_count)
    answer_claim_f1 = harmonic_mean(answer_claim_precision, answer_claim_recall)
    grounded_claim_ratio = safe_ratio(answer_supported_by_context, answer_count)
    hallucinated_claim_ratio = safe_ratio(answer_count - answer_supported_by_context, answer_count)
    context_utilization = safe_ratio(
        sum(
            1
            for row in gold_rows
            if isinstance(row, dict)
            and normalize_label(row.get("context_nli")) == "supported"
            and normalize_label(row.get("answer_nli")) == "supported"
        ),
        gold_supported_by_context,
    )
    claim_diagnostic = classify_claim_diagnostic(
        gold_claim_count=gold_count,
        answer_claim_count=answer_count,
        context_claim_recall=context_claim_recall,
        answer_claim_recall=answer_claim_recall,
        answer_claim_precision=answer_claim_precision,
        grounded_claim_ratio=grounded_claim_ratio,
        hallucinated_claim_ratio=hallucinated_claim_ratio,
        context_utilization=context_utilization,
        contradicted_claim_count=answer_contradicted_by_context,
    )
    attribution_metrics = compute_attribution_metrics(claim_evidence_map)

    return {
        "gold_claim_count": gold_count,
        "answer_claim_count": answer_count,
        "context_claim_recall": context_claim_recall,
        "answer_claim_recall": answer_claim_recall,
        "answer_claim_precision": answer_claim_precision,
        "answer_claim_f1": answer_claim_f1,
        "grounded_claim_ratio": grounded_claim_ratio,
        "hallucinated_claim_ratio": hallucinated_claim_ratio,
        "context_utilization": context_utilization,
        "evidence_attribution_precision": attribution_metrics["evidence_attribution_precision"],
        "evidence_attribution_recall": attribution_metrics["evidence_attribution_recall"],
        "evidence_attribution_f1": attribution_metrics["evidence_attribution_f1"],
        "evidence_coverage": attribution_metrics["evidence_coverage"],
        "attributed_answer_claim_count": attribution_metrics["attributed_answer_claim_count"],
        "attributed_gold_claim_count": attribution_metrics["attributed_gold_claim_count"],
        "invalid_attribution_count": attribution_metrics["invalid_attribution_count"],
        "claim_evidence_map": claim_evidence_map,
        "unsupported_claim_count": max(answer_count - answer_supported_by_context, 0),
        "missing_gold_claim_count": max(gold_count - gold_present_in_answer, 0),
        "contradicted_claim_count": answer_contradicted_by_context,
        "claim_diagnostic": claim_diagnostic,
    }


def judge_claims_with_llm(
    *,
    item: Dict,
    answer: str,
    retrieved: Sequence[Dict],
    config: LLMConfig,
) -> ClaimJudgeResult:
    if not config.enabled:
        return ClaimJudgeResult(used=False, status="disabled", error=None, model=None)

    client = get_openai_client(config)
    if client is None:
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            return ClaimJudgeResult(
                used=False,
                status="missing_api_key",
                error=f"Environment variable {config.api_key_env} is empty or unset.",
                model=config.model,
            )
        return ClaimJudgeResult(used=False, status="client_unavailable", error=None, model=config.model)

    gold_claims = reference_claims_from_item(item)
    answer_claims = split_claims(answer)
    if not gold_claims and not answer_claims:
        return ClaimJudgeResult(
            used=False,
            status="no_claims",
            error=None,
            model=config.model,
            metrics=None,
        )

    prompt_payload = {
        "question": item.get("question", ""),
        "gold_answer": item.get("gold_answer", ""),
        "expected_keywords": item.get("expected_keywords", []),
        "system_answer": answer,
        "gold_claims": gold_claims,
        "answer_claims": answer_claims,
        "retrieved_context": indexed_context(retrieved),
    }
    try:
        print(
            f"[JUDGE] question={item.get('id', item.get('question', ''))!r} model={config.model} status=request_started",
            file=sys.stderr,
        )
        response = client.responses.create(
            model=config.model,
            temperature=config.temperature,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an NLI-style evaluator for retrieval-augmented generation. "
                        "Return only valid JSON. Use labels supported, contradicted, or not_enough_info. "
                        "A claim is supported when the evidence entails the substance of the claim, including "
                        "clear paraphrases. Do not require identical wording. If the question already states a "
                        "condition or event, do not penalize an answer for not restating that condition. "
                        "Do not give credit for vague lexical overlap alone."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Evaluate the RAG answer. For each gold claim, compare it to retrieved_context "
                        "and system_answer. For each answer claim, compare it to retrieved_context and "
                        "gold_answer. Judge whether the answer correctly addresses the user's question; "
                        "minor omissions are acceptable when they are already present in the question or are "
                        "not needed for the requested answer. Return JSON with this shape:\n"
                        "{"
                        "\"gold_claims\":[{\"claim\":\"...\",\"context_nli\":\"supported|contradicted|not_enough_info\","
                        "\"answer_nli\":\"supported|contradicted|not_enough_info\",\"supporting_chunk_ids\":[\"...\"]}],"
                        "\"answer_claims\":[{\"claim\":\"...\",\"context_nli\":\"supported|contradicted|not_enough_info\","
                        "\"gold_nli\":\"supported|contradicted|not_enough_info\",\"supporting_chunk_ids\":[\"...\"]}],"
                        "\"overall_reason\":\"short diagnostic reason\""
                        "}\n\n"
                        f"Input JSON:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
                    ),
                },
            ],
        )
        raw_response = (getattr(response, "output_text", "") or "").strip()
        payload = extract_json_object(raw_response)
        valid_chunk_ids = {str(row.get("chunk_id", "")) for row in retrieved if row.get("chunk_id")}
        metrics = metrics_from_judge_payload(payload, gold_claims, answer_claims, valid_chunk_ids)
        print(
            f"[JUDGE] question={item.get('id', item.get('question', ''))!r} model={config.model} status=success",
            file=sys.stderr,
        )
        return ClaimJudgeResult(
            used=True,
            status="success",
            error=None,
            model=config.model,
            metrics=metrics,
            raw_response=raw_response,
        )
    except Exception as exc:
        print(
            f"[JUDGE] question={item.get('id', item.get('question', ''))!r} model={config.model} status=error error={exc}",
            file=sys.stderr,
        )
        return ClaimJudgeResult(
            used=True,
            status="error",
            error=str(exc),
            model=config.model,
            metrics=None,
            raw_response=None,
        )
