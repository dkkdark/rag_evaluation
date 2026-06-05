from __future__ import annotations

from typing import Dict, List, Sequence

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_CROSS_ENCODER_CACHE: Dict[str, object] = {}


def get_cross_encoder(model_name: str):
    if model_name not in _CROSS_ENCODER_CACHE:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]


def _candidate_passage(row: Dict[str, object]) -> str:
    parts = [str(row.get("title") or ""), str(row.get("text") or "")]
    return "\n".join(part for part in parts if part.strip())


def rerank_with_cross_encoder(
    *,
    query: str,
    rows: Sequence[Dict[str, object]],
    top_k: int,
    rerank_top_n: int = 10,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> List[Dict[str, object]]:
    if top_k <= 0 or not rows:
        return []

    limited_top_n = min(len(rows), max(top_k, rerank_top_n))
    head = [dict(row) for row in rows[:limited_top_n]]
    pairs = [(query, _candidate_passage(row)) for row in head]
    model = get_cross_encoder(model_name)
    scores = model.predict(pairs)

    reranked: List[Dict[str, object]] = []
    for row, ce_score in zip(head, scores):
        row["base_score_before_cross_encoder"] = float(row.get("score") or 0.0)
        row["cross_encoder_score"] = float(ce_score)
        row["score"] = float(ce_score)
        row["reranked"] = True
        row["reranker"] = "cross_encoder"
        reranked.append(row)

    reranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return reranked[:top_k]
