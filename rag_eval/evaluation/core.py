from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Callable, Dict, List, Sequence


@dataclass
class RankedCandidate:
    label: str
    score: float | None = None
    metadata: Dict[str, object] | None = None


@dataclass
class EvaluationItem:
    id: str
    query: str
    gold_labels: List[str]
    metadata: Dict[str, object] | None = None


@dataclass
class PredictionRecord:
    id: str
    candidates: List[RankedCandidate]
    metadata: Dict[str, object] | None = None


def hierarchical_distance_from_parent_map(
    predicted: str,
    gold: str,
    parent_map: Dict[str, str],
) -> int | None:
    if not predicted or not gold:
        return None
    if predicted == gold:
        return 0

    def chain(label: str) -> List[str]:
        out: List[str] = []
        seen = set()
        current = label
        while current and current not in seen:
            out.append(current)
            seen.add(current)
            current = parent_map.get(current, "")
        return out

    predicted_chain = chain(predicted)
    gold_chain = chain(gold)
    gold_steps = {label: index for index, label in enumerate(gold_chain)}
    for predicted_index, label in enumerate(predicted_chain):
        if label in gold_steps:
            return predicted_index + gold_steps[label]
    return None


CPV_HIERARCHY_LEVELS = [2, 4, 6, 8]
CPV_MAX_STRUCTURAL_DISTANCE = (len(CPV_HIERARCHY_LEVELS) - 1) * 2 + 2
CPV_HIERARCHY_LABELS = {
    0: "no_overlap",
    2: "division",
    4: "group",
    6: "class",
    8: "category",
}
CPV_HIERARCHY_SCORES = {
    0: 0.0,
    2: 0.25,
    4: 0.5,
    6: 0.75,
    8: 1.0,
}


def normalize_cpv_code(value: str) -> str:
    match = re.search(r"\b(\d{8})\b", str(value or ""))
    return match.group(1) if match else ""


def cpv_common_prefix_length(predicted: str, gold: str) -> int | None:
    predicted_code = normalize_cpv_code(predicted)
    gold_code = normalize_cpv_code(gold)
    if not predicted_code or not gold_code:
        return None
    common = 0
    for predicted_char, gold_char in zip(predicted_code, gold_code):
        if predicted_char != gold_char:
            break
        common += 1
    return common


def cpv_structural_distance(predicted: str, gold: str) -> int | None:
    common = cpv_common_prefix_length(predicted, gold)
    if common is None:
        return None
    if common == 8:
        return 0
    common_level_depth = -1
    for depth, prefix_length in enumerate(CPV_HIERARCHY_LEVELS):
        if common >= prefix_length:
            common_level_depth = depth
    leaf_depth = len(CPV_HIERARCHY_LEVELS) - 1
    if common_level_depth < 0:
        return CPV_MAX_STRUCTURAL_DISTANCE
    return (leaf_depth - common_level_depth) * 2


def cpv_structural_similarity(predicted: str, gold: str) -> float | None:
    distance = cpv_structural_distance(predicted, gold)
    if distance is None:
        return None
    return max(0.0, 1.0 - (distance / CPV_MAX_STRUCTURAL_DISTANCE))


def cpv_hierarchy_match_level(predicted: str, gold: str) -> int | None:
    common = cpv_common_prefix_length(predicted, gold)
    if common is None:
        return None
    for prefix_length in reversed(CPV_HIERARCHY_LEVELS):
        if common >= prefix_length:
            return prefix_length
    return 0


def cpv_hierarchy_match(predicted: str, gold: str) -> Dict[str, object] | None:
    level = cpv_hierarchy_match_level(predicted, gold)
    if level is None:
        return None
    return {
        "level": level,
        "label": CPV_HIERARCHY_LABELS[level],
        "score": CPV_HIERARCHY_SCORES[level],
    }


def best_cpv_hierarchy_match(predicted: str, gold_labels: Sequence[str]) -> Dict[str, object] | None:
    matches = [
        match
        for gold_label in gold_labels
        for match in [cpv_hierarchy_match(predicted, gold_label)]
        if match is not None
    ]
    if not matches:
        return None
    return max(matches, key=lambda match: float(match["score"]))


def _dcg(gains: Sequence[float]) -> float:
    import math

    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def evaluate_ranked_predictions(
    items: Sequence[EvaluationItem],
    predictions: Sequence[PredictionRecord],
    *,
    top_k: int = 5,
    distance_fn: Callable[[str, str], int | None] | None = None,
) -> Dict[str, object]:
    prediction_map = {prediction.id: prediction for prediction in predictions}

    exact_top1 = 0
    hit_at_k = 0
    precision_at_k_values: List[float] = []
    reciprocal_ranks: List[float] = []
    hierarchy_similarities: List[float] = []
    hierarchy_scores_top1: List[float] = []
    best_hierarchy_scores_at_k: List[float] = []
    ndcg_at_k_values: List[float] = []
    top1_match_breakdown: Counter[str] = Counter()
    ranked_best_match_breakdown: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    miss_examples: List[Dict[str, object]] = []

    for item in items:
        record = prediction_map.get(item.id)
        ranked_labels = [candidate.label for candidate in (record.candidates if record else [])[:top_k]]
        top1 = ranked_labels[0] if ranked_labels else ""
        gold_set = set(item.gold_labels)

        if not ranked_labels:
            reciprocal_ranks.append(0.0)
            precision_at_k_values.append(0.0)
            best_hierarchy_scores_at_k.append(0.0)
            ndcg_at_k_values.append(0.0)
            top1_match_breakdown["no_prediction"] += 1
            ranked_best_match_breakdown["no_prediction"] += 1
            reasons["no_candidates_returned"] += 1
            miss_examples.append(
                {
                    "id": item.id,
                    "query": item.query,
                    "gold_labels": item.gold_labels,
                    "predicted_top1": "",
                }
            )
            continue

        exact_hits = [label for label in ranked_labels if label in gold_set]
        precision_at_k_values.append(len(exact_hits) / top_k if top_k else 0.0)

        if top1 in gold_set:
            exact_top1 += 1
            hit_at_k += 1
            reciprocal_ranks.append(1.0)
            reasons["ok"] += 1
        else:
            first_hit_rank = next(
                (rank for rank, label in enumerate(ranked_labels, start=1) if label in gold_set),
                None,
            )
            if first_hit_rank is not None:
                hit_at_k += 1
                reciprocal_ranks.append(1.0 / first_hit_rank)
                reasons["gold_present_but_not_ranked_first"] += 1
            else:
                reciprocal_ranks.append(0.0)
                reasons["gold_missing_from_top_k"] += 1
                miss_examples.append(
                    {
                        "id": item.id,
                        "query": item.query,
                        "gold_labels": item.gold_labels,
                        "predicted_top1": top1,
                    }
                )

        top1_match = best_cpv_hierarchy_match(top1, item.gold_labels)
        if top1_match is not None:
            hierarchy_scores_top1.append(float(top1_match["score"]))
            top1_match_breakdown[str(top1_match["label"])] += 1

        ranked_matches = [
            match
            for label in ranked_labels
            for match in [best_cpv_hierarchy_match(label, item.gold_labels)]
            if match is not None
        ]
        if ranked_matches:
            best_ranked_match = max(ranked_matches, key=lambda match: float(match["score"]))
            best_hierarchy_scores_at_k.append(float(best_ranked_match["score"]))
            ranked_best_match_breakdown[str(best_ranked_match["label"])] += 1
        else:
            best_hierarchy_scores_at_k.append(0.0)
            ranked_best_match_breakdown["no_overlap"] += 1

        graded_gains = [
            float(match["score"]) if match is not None else 0.0
            for label in ranked_labels
            for match in [best_cpv_hierarchy_match(label, item.gold_labels)]
        ]
        ideal_gains = sorted(graded_gains, reverse=True)
        ideal_dcg = _dcg(ideal_gains)
        ndcg_at_k_values.append(_dcg(graded_gains) / ideal_dcg if ideal_dcg else 0.0)

        if distance_fn is not None and top1:
            gold_distance_candidates = [
                distance
                for gold_label in item.gold_labels
                for distance in [distance_fn(top1, gold_label)]
                if distance is not None
            ]
            if gold_distance_candidates:
                best_distance = min(gold_distance_candidates)
                hierarchy_similarities.append(
                    max(0.0, 1.0 - (best_distance / CPV_MAX_STRUCTURAL_DISTANCE))
                )
    n_items = len(items) or 1
    mean_hierarchy_score_top1 = (
        sum(hierarchy_scores_top1) / len(hierarchy_scores_top1) if hierarchy_scores_top1 else None
    )
    mean_best_hierarchy_score_at_k = (
        sum(best_hierarchy_scores_at_k) / len(best_hierarchy_scores_at_k)
        if best_hierarchy_scores_at_k
        else None
    )
    mean_ndcg_at_k = sum(ndcg_at_k_values) / n_items
    mean_precision_at_k = sum(precision_at_k_values) / n_items
    mrr_at_k = sum(reciprocal_ranks) / n_items
    exact_top1_accuracy = exact_top1 / n_items
    hit_at_k_rate = hit_at_k / n_items
    return {
        "n_items": len(items),
        "top_k": top_k,
        "exact_top1_accuracy": exact_top1_accuracy,
        "hit_at_k": hit_at_k_rate,
        "precision_at_k": mean_precision_at_k,
        "mrr_at_k": mrr_at_k,
        "ndcg_at_k": mean_ndcg_at_k,
        "mean_hierarchy_score_top1": mean_hierarchy_score_top1,
        "mean_best_hierarchy_score_at_k": mean_best_hierarchy_score_at_k,
        "mean_cpv_hierarchy_similarity_top1": (
            sum(hierarchy_similarities) / len(hierarchy_similarities)
            if hierarchy_similarities
            else None
        ),
        "top1_metrics": {
            "exact_accuracy": exact_top1_accuracy,
            "mean_hierarchy_score": mean_hierarchy_score_top1,
            "mean_cpv_hierarchy_similarity": (
                sum(hierarchy_similarities) / len(hierarchy_similarities)
                if hierarchy_similarities
                else None
            ),
            "match_breakdown": dict(top1_match_breakdown),
            "n_exact": exact_top1,
        },
        "ranked_list_metrics": {
            "top_k": top_k,
            "hit_at_k": hit_at_k_rate,
            "precision_at_k": mean_precision_at_k,
            "mrr_at_k": mrr_at_k,
            "ndcg_at_k": mean_ndcg_at_k,
            "mean_best_hierarchy_score_at_k": mean_best_hierarchy_score_at_k,
            "best_match_breakdown": dict(ranked_best_match_breakdown),
            "n_hit_at_k": hit_at_k,
            "n_missed_at_k": len(items) - hit_at_k,
        },
        "n_exact_top1": exact_top1,
        "n_hit_at_k": hit_at_k,
        "n_missed_at_k": len(items) - hit_at_k,
        "failure_summary": {
            "counts_by_reason": dict(reasons),
            "most_common_reason": reasons.most_common(1)[0][0] if reasons else None,
        },
        "miss_examples": miss_examples[:10],
    }
