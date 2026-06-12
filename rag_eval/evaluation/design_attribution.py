from __future__ import annotations

import json
import os
import math
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence


DESIGN_FACTORS = [
    "chunking_strategy",
    "chunk_size",
    "chunk_overlap",
    "retriever",
    "top_k",
    "hybrid_alpha",
    "rerank_top_n",
    "rerank_weight",
    "query_augmentation",
    "context_mode",
    "answer_mode",
    "llm_model",
    "judge_model",
    "kg_enabled",
    "kg_profile",
    "kg_algorithm",
    "kg_graph_weight",
    "self_rag_retry_on_weak_evidence",
    "self_rag_critique",
]


QUALITY_METRICS = [
    "auto_correct",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "ragas_recall_at_k",
    "context_claim_recall",
    "answer_claim_recall",
    "grounded_claim_ratio",
    "hallucinated_claim_ratio",
    "factual_correctness_recall",
    "evidence_attribution_f1",
    "gold_kg_relation_evidence_recall",
    "kg_graph_gain_at_k",
    "kg_graph_noise_at_k",
    "kg_added_evidence_precision",
    "kg_path_correctness",
]


NEGATIVE_METRICS = {
    "hallucinated_claim_ratio",
    "kg_graph_noise_at_k",
}


CONTROL_FACTORS = [
    factor for factor in DESIGN_FACTORS
    if factor not in {"llm_model", "judge_model"}
]


COMPONENT_SCORE_DEFINITIONS = {
    "retrieval_score": ["mrr_at_k", "ndcg_at_k", "recall_at_k", "ragas_recall_at_k"],
    "generation_score": ["auto_correct", "answer_claim_recall", "factual_correctness_recall"],
    "grounding_score": ["grounded_claim_ratio", "evidence_attribution_f1"],
    "kg_score": ["gold_kg_relation_evidence_recall", "kg_graph_gain_at_k", "kg_added_evidence_precision", "kg_path_correctness"],
    "calibration_score": ["prediction_confidence"],
}


FAILURE_TAXONOMY = {
    "candidate_generation": {
        "label": "Candidate generation / retriever coverage",
        "explanation": "The expected answer or required evidence is not available in the retrieved candidate set, so later reranking or prompting cannot recover it.",
        "advisor_component": "retriever",
        "advisor_issue": "Expected answer is missing from the candidate list",
        "recommendation": "Treat this as a candidate generation problem before tuning the final prompt.",
        "next_experiment": "Increase top-k, compare BM25/dense/hybrid retrieval, and enrich candidate or chunk text with descriptions, examples, synonyms, and hierarchy context.",
    },
    "candidate_ranking": {
        "label": "Candidate ranking / selection",
        "explanation": "Useful evidence or the expected candidate is present, but it is ranked below a wrong item or not selected as the final answer.",
        "advisor_component": "prompt",
        "advisor_issue": "Top candidates are present but not selected correctly",
        "recommendation": "Focus on the decision layer: reranker, score fusion, or a contrastive prompt over the top candidates.",
        "next_experiment": "Freeze candidate generation and compare rerank_top_n, rerank_weight, cross-encoder reranking, and contrastive selection prompts.",
    },
    "taxonomy_disambiguation": {
        "label": "Taxonomy / hierarchy disambiguation",
        "explanation": "The prediction is near the gold item in the taxonomy but still wrong, so the issue is distinguishing related labels rather than broad retrieval coverage.",
        "advisor_component": "prompt",
        "advisor_issue": "Related hierarchy branch but wrong final code",
        "recommendation": "Improve hierarchy-aware disambiguation instead of treating these as random retrieval misses.",
        "next_experiment": "Add parent/child/sibling labels, require contrast against close alternatives, and evaluate same-division/same-group errors separately.",
    },
    "calibration": {
        "label": "Confidence calibration",
        "explanation": "The system's confidence is not reliable for the observed outcome, especially when wrong predictions are assigned high confidence.",
        "advisor_component": "calibration",
        "advisor_issue": "Wrong answer is predicted with high confidence",
        "recommendation": "Do not auto-accept high-confidence predictions until scores are calibrated.",
        "next_experiment": "Create reliability bins and test auto-accept/manual-review thresholds using confidence, score margin, and entropy.",
    },
    "document_parsing": {
        "label": "Document parsing / extraction",
        "explanation": "The source document may have been parsed into empty, fragmented, or malformed sections, limiting every downstream retrieval method.",
        "advisor_component": "retriever",
        "advisor_issue": "Relevant evidence is unavailable because parsing degraded the source text",
        "recommendation": "Inspect parsed sections before tuning retrieval or prompts.",
        "next_experiment": "Compare PDF extraction output, section counts, empty-section rate, and paragraph lengths across documents; fix parsing before retrieval ablations.",
    },
    "chunking": {
        "label": "Chunking / context segmentation",
        "explanation": "Evidence may be split too narrowly, merged too broadly, or separated from headings/metadata needed to answer the question.",
        "advisor_component": "retriever",
        "advisor_issue": "Chunk boundaries prevent complete evidence retrieval",
        "recommendation": "Treat this as a segmentation problem and compare chunking strategies before changing the generator.",
        "next_experiment": "Compare by_section, by_paragraph, fixed_words, fixed_tokens, chunk_size, and overlap while holding retriever and top-k fixed.",
    },
    "context_selection": {
        "label": "Context selection / evidence coverage",
        "explanation": "Some useful evidence exists, but the final context is incomplete, noisy, or not enough for a complete answer.",
        "advisor_component": "retriever",
        "advisor_issue": "Some required evidence is found, but not enough for complete answers",
        "recommendation": "Improve evidence coverage before answer-side tuning.",
        "next_experiment": "Increase top-k, add reranking, try query decomposition or HyDE, and inspect context_claim_recall by question type.",
    },
    "kg_expansion": {
        "label": "KG expansion coverage",
        "explanation": "Graph expansion did not add useful missing evidence, so the graph traversal/profile may not match the question type.",
        "advisor_component": "retriever",
        "advisor_issue": "KG expansion is not recovering additional useful evidence",
        "recommendation": "Compare KG traversal profiles before relying on graph expansion.",
        "next_experiment": "Compare conservative, balanced, exploratory, direct_only, ppr_only, and ppr_direct with KG gain/noise and path correctness.",
    },
    "kg_noise": {
        "label": "KG expansion noise",
        "explanation": "Graph expansion adds related but distracting chunks, increasing noise or unsupported claims.",
        "advisor_component": "retriever",
        "advisor_issue": "KG expansion adds noisy evidence",
        "recommendation": "Tighten graph expansion before increasing context size.",
        "next_experiment": "Lower kg_graph_weight, use conservative/direct profiles, raise quality threshold, and track kg_graph_noise_at_k.",
    },
    "answer_generation": {
        "label": "Answer generation / synthesis",
        "explanation": "The context contains enough evidence, but the generated answer omits required facts, adds unsupported claims, or synthesizes incorrectly.",
        "advisor_component": "prompt",
        "advisor_issue": "The generator receives enough evidence but answer synthesis fails",
        "recommendation": "Treat this as a synthesis/completeness problem rather than a retrieval problem.",
        "next_experiment": "Freeze retrieval and compare grounded_llm, cite_first, claim_checklist, stricter prompts, and claim coverage instructions.",
    },
    "citation_attribution": {
        "label": "Citation / evidence attribution",
        "explanation": "The answer may be substantively close, but its claims are not mapped reliably to retrieved chunks or source spans.",
        "advisor_component": "attribution",
        "advisor_issue": "Claims appear grounded, but source mapping is weak",
        "recommendation": "Require claim-level citations or extract evidence spans before answer generation.",
        "next_experiment": "Compare cite_first and claim_checklist modes, constrain citation IDs to retrieved chunks, and measure evidence_attribution_f1.",
    },
    "benchmark_ambiguity": {
        "label": "Benchmark / input ambiguity",
        "explanation": "The query or reference may be too short, ambiguous, or missing acceptable alternatives, so the row may not represent a pure system failure.",
        "advisor_component": "benchmark",
        "advisor_issue": "Query may be too short or ambiguous",
        "recommendation": "Treat this row as needing extra domain context before blaming only the classifier.",
        "next_experiment": "Add title, description, source metadata, manual ambiguity labels, or accepted alternative answers and rerun.",
    },
    "success": {
        "label": "No failure",
        "explanation": "The row is currently marked successful by the available evaluation signals.",
        "advisor_component": "none",
        "advisor_issue": "No targeted remediation needed",
        "recommendation": "Keep it as a regression example.",
        "next_experiment": "Use this row to make sure future changes do not regress already solved cases.",
    },
    "unknown": {
        "label": "Unknown / manual review",
        "explanation": "The available metrics do not identify a single likely failing component with enough confidence.",
        "advisor_component": "manual_review",
        "advisor_issue": "Automatic diagnosis is inconclusive",
        "recommendation": "Send this row to manual review or add more diagnostic signals.",
        "next_experiment": "Inspect retrieved chunks, expected answer, diagnostics, and run a focused one-factor ablation.",
    },
}


def _read_csv(path: object):
    if not path or not os.path.exists(str(path)):
        return None
    import pandas as pd

    try:
        return pd.read_csv(str(path))
    except Exception:
        return None


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        if number != number:
            return None
        return number
    except (TypeError, ValueError):
        return None


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except TypeError:
        pass
    return str(value)


def _first_present(row: Dict[str, object], keys: Sequence[str]) -> object:
    for key in keys:
        if key in row and row.get(key) not in {None, ""}:
            return row.get(key)
    return None


def _join_by_question_id(frames: Sequence[object]) -> Dict[str, Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}
    for frame in frames:
        if frame is None or getattr(frame, "empty", True):
            continue
        for raw_row in frame.to_dict(orient="records"):
            question_id = _safe_str(
                _first_present(raw_row, ["question_id", "id", "query_id", "ID"])
                or raw_row.get("question")
                or raw_row.get("query")
            )
            if not question_id:
                continue
            target = merged.setdefault(question_id, {})
            for key, value in raw_row.items():
                if key not in target or target.get(key) in {None, ""}:
                    target[key] = value
    return merged


def _stable_design_quality_score(component_scores: Dict[str, float | None], row: Dict[str, object]) -> tuple[float, int]:
    values = [
        _safe_float(component_scores.get("retrieval_score")),
        _safe_float(component_scores.get("generation_score")),
        _safe_float(component_scores.get("grounding_score")),
    ]
    if row.get("kg_enabled") in {True, "True", "true", "1"}:
        values.append(_safe_float(component_scores.get("kg_score")))
    present = [value for value in values if value is not None]
    return (sum(present) / len(present), len(present)) if present else (0.0, 0)


def _metric_score(row: Dict[str, object]) -> float:
    stored = _safe_float(row.get("design_quality_score"))
    if stored is not None:
        return stored
    scores = _component_scores(row)
    value, _ = _stable_design_quality_score(scores, row)
    return value


def _mean_present(values: Iterable[object]) -> float | None:
    numbers = [_safe_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _component_scores(row: Dict[str, object]) -> Dict[str, float | None]:
    scores: Dict[str, float | None] = {}
    for score_name, metrics in COMPONENT_SCORE_DEFINITIONS.items():
        values = []
        for metric in metrics:
            value = _safe_float(row.get(metric))
            if value is None:
                continue
            if metric == "prediction_confidence" and _is_failure(row):
                value = min(max(value, 0.0), 1.0)
                value = 1.0 - min(max(value, 0.0), 1.0)
            elif metric == "prediction_confidence":
                value = min(max(value, 0.0), 1.0)
            values.append(value)
        scores[score_name] = sum(values) / len(values) if values else None
    hallucinated = _safe_float(row.get("hallucinated_claim_ratio"))
    if hallucinated is not None:
        grounding_values = [value for value in [scores.get("grounding_score"), 1.0 - hallucinated] if value is not None]
        scores["grounding_score"] = sum(grounding_values) / len(grounding_values) if grounding_values else None
    kg_noise = _safe_float(row.get("kg_graph_noise_at_k"))
    if kg_noise is not None:
        kg_values = [value for value in [scores.get("kg_score"), 1.0 - kg_noise] if value is not None]
        scores["kg_score"] = sum(kg_values) / len(kg_values) if kg_values else None
    cost = _safe_float(row.get("estimated_extra_cost_units"))
    scores["cost_efficiency_score"] = 1.0 / (1.0 + cost) if cost is not None else None
    return scores


def _config_from_summary(summary: Dict[str, object]) -> Dict[str, object]:
    evaluation_settings = summary.get("evaluation_settings", {}) if isinstance(summary.get("evaluation_settings"), dict) else {}
    kg = summary.get("kg", {}) if isinstance(summary.get("kg"), dict) else {}
    reranker = summary.get("reranker", {}) if isinstance(summary.get("reranker"), dict) else {}
    llm = summary.get("llm", {}) if isinstance(summary.get("llm"), dict) else {}
    judge = summary.get("judge", {}) if isinstance(summary.get("judge"), dict) else {}
    classifier = summary.get("classifier", {}) if isinstance(summary.get("classifier"), dict) else {}
    return {
        "experiment": summary.get("experiment") or classifier.get("type") or summary.get("classifier_type"),
        "chunking_strategy": summary.get("chunking_strategy"),
        "chunk_size": summary.get("chunk_size"),
        "chunk_overlap": summary.get("chunk_overlap"),
        "retriever": summary.get("retriever"),
        "top_k": summary.get("top_k"),
        "hybrid_alpha": summary.get("hybrid_alpha"),
        "rerank_top_n": reranker.get("top_n"),
        "rerank_weight": reranker.get("weight"),
        "query_augmentation": evaluation_settings.get("query_augmentation") or llm.get("query_augmentation"),
        "context_mode": evaluation_settings.get("context_mode"),
        "answer_mode": evaluation_settings.get("answer_mode"),
        "answer_generation_enabled": llm.get("answer_generation"),
        "llm_model": llm.get("model"),
        "judge_model": judge.get("model"),
        "kg_enabled": kg.get("enabled"),
        "kg_profile": kg.get("profile"),
        "kg_algorithm": kg.get("algorithm"),
        "kg_graph_weight": kg.get("graph_weight"),
        "self_rag_retry_on_weak_evidence": evaluation_settings.get("self_rag_retry_on_weak_evidence") or llm.get("self_rag_retry_on_weak_evidence"),
        "self_rag_critique": evaluation_settings.get("self_rag_critique") or llm.get("self_rag_critique"),
        "runtime_seconds": summary.get("runtime_seconds"),
    }


def _normalize_trace_row(row: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    auto_flag = _safe_str(row.get("auto_flag") or row.get("answer_accuracy_label"))
    auto_correct = 1.0 if auto_flag == "correct" else 0.0 if auto_flag else None
    normalized = {
        **config,
        "question_id": _safe_str(_first_present(row, ["question_id", "id", "query_id", "ID"]) or row.get("question")),
        "question": _safe_str(_first_present(row, ["question", "query", "Query (BANF)"])),
        "question_type": _safe_str(row.get("question_type") or row.get("task_type") or "unknown"),
        "auto_flag": auto_flag,
        "auto_correct": auto_correct,
        "primary_error_reason": _safe_str(row.get("primary_error_reason")),
        "secondary_error_reason": _safe_str(row.get("secondary_error_reason")),
        "claim_diagnostic": _safe_str(row.get("claim_diagnostic")),
        "runtime_retrieval_status": _safe_str(row.get("runtime_retrieval_status")),
        "prediction_confidence": _safe_float(row.get("prediction_confidence")),
        "first_relevant_rank": _safe_float(row.get("first_relevant_rank")),
        "first_target_doc_rank": _safe_float(row.get("first_target_doc_rank")),
    }
    for metric in QUALITY_METRICS:
        if metric == "auto_correct":
            continue
        normalized[metric] = _safe_float(row.get(metric))
    component_scores = _component_scores(normalized)
    normalized.update(component_scores)
    design_quality_score, design_quality_metric_count = _stable_design_quality_score(component_scores, normalized)
    normalized["design_quality_score"] = design_quality_score
    normalized["design_quality_metric_count"] = design_quality_metric_count
    normalized["estimated_llm_call_count"] = _estimate_llm_calls(normalized)
    normalized["estimated_extra_cost_units"] = normalized["estimated_llm_call_count"] + (1 if normalized.get("kg_enabled") else 0)
    return normalized


def _estimate_llm_calls(row: Dict[str, object]) -> int:
    calls = 0
    if _safe_str(row.get("query_augmentation")) in {"llm", "hyde", "translate_en"}:
        calls += 1
    if row.get("answer_generation_enabled") in {True, "True", "true", "1"}:
        calls += 1
    if row.get("judge_model"):
        calls += 1
    if row.get("self_rag_retry_on_weak_evidence") in {True, "True", "true", "1"}:
        calls += 1
    if row.get("self_rag_critique") in {True, "True", "true", "1"}:
        calls += 1
    return calls


def _iter_experiment_summaries(run_summary: Dict[str, object]) -> List[Dict[str, object]]:
    experiments = run_summary.get("experiments")
    if isinstance(experiments, list) and experiments:
        return [item for item in experiments if isinstance(item, dict)]
    classifier_summary = run_summary.get("classifier_summary")
    if isinstance(classifier_summary, dict):
        return [classifier_summary]
    return []


def build_design_trace(run_summary: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary in _iter_experiment_summaries(run_summary):
        outputs = summary.get("outputs", {}) if isinstance(summary.get("outputs"), dict) else {}
        merged = _join_by_question_id(
            [
                _read_csv(outputs.get("rag_results_csv")),
                _read_csv(outputs.get("retrieval_metrics_csv")),
                _read_csv(outputs.get("answer_metrics_csv")),
                _read_csv(outputs.get("diagnostics_csv")),
            ]
        )
        config = _config_from_summary(summary)
        if not config.get("experiment"):
            config["experiment"] = _safe_str(run_summary.get("classifier_type") or run_summary.get("mode") or "single_config")
        for row in merged.values():
            normalized = _normalize_trace_row(row, config)
            if normalized.get("question_id"):
                rows.append(normalized)
    return rows


def build_paired_comparisons(trace_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    comparisons: List[Dict[str, object]] = []
    by_question: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in trace_rows:
        by_question[_safe_str(row.get("question_id"))].append(row)
    for question_id, rows in by_question.items():
        if len(rows) < 2:
            continue
        for factor in DESIGN_FACTORS:
            grouped_by_skeleton: Dict[tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
            for row in rows:
                skeleton = tuple(
                    _safe_str(row.get(item))
                    for item in CONTROL_FACTORS
                    if item != factor
                )
                grouped_by_skeleton[skeleton].append(row)
            for skeleton_rows in grouped_by_skeleton.values():
                values = sorted({_safe_str(row.get(factor)) for row in skeleton_rows})
                if len(values) < 2:
                    continue
                by_value: Dict[str, Dict[str, object]] = {}
                for row in skeleton_rows:
                    value = _safe_str(row.get(factor))
                    if not value:
                        continue
                    score = _metric_score(row)
                    existing = by_value.get(value)
                    if existing is None or score > _metric_score(existing):
                        by_value[value] = row
                ordered_values = sorted(by_value)
                for index, left_value in enumerate(ordered_values):
                    left_row = by_value[left_value]
                    for right_value in ordered_values[index + 1:]:
                        right_row = by_value[right_value]
                        left_score = _metric_score(left_row)
                        right_score = _metric_score(right_row)
                        comparisons.append(
                            {
                                "question_id": question_id,
                                "question_type": left_row.get("question_type") or right_row.get("question_type"),
                                "factor": factor,
                                "left_value": left_value,
                                "right_value": right_value,
                                "left_experiment": left_row.get("experiment"),
                                "right_experiment": right_row.get("experiment"),
                                "left_design_quality_score": left_score,
                                "right_design_quality_score": right_score,
                                "score_delta_right_minus_left": right_score - left_score,
                                "comparison_type": "controlled",
                                "n_other_changed_factors": 0,
                                "other_changed_factors": "",
                                "left_retrieval_score": left_row.get("retrieval_score"),
                                "right_retrieval_score": right_row.get("retrieval_score"),
                                "retrieval_score_delta_right_minus_left": (_safe_float(right_row.get("retrieval_score")) or 0.0) - (_safe_float(left_row.get("retrieval_score")) or 0.0)
                                if _safe_float(right_row.get("retrieval_score")) is not None and _safe_float(left_row.get("retrieval_score")) is not None else None,
                                "left_generation_score": left_row.get("generation_score"),
                                "right_generation_score": right_row.get("generation_score"),
                                "generation_score_delta_right_minus_left": (_safe_float(right_row.get("generation_score")) or 0.0) - (_safe_float(left_row.get("generation_score")) or 0.0)
                                if _safe_float(right_row.get("generation_score")) is not None and _safe_float(left_row.get("generation_score")) is not None else None,
                                "left_grounding_score": left_row.get("grounding_score"),
                                "right_grounding_score": right_row.get("grounding_score"),
                                "grounding_score_delta_right_minus_left": (_safe_float(right_row.get("grounding_score")) or 0.0) - (_safe_float(left_row.get("grounding_score")) or 0.0)
                                if _safe_float(right_row.get("grounding_score")) is not None and _safe_float(left_row.get("grounding_score")) is not None else None,
                                "left_kg_score": left_row.get("kg_score"),
                                "right_kg_score": right_row.get("kg_score"),
                                "kg_score_delta_right_minus_left": (_safe_float(right_row.get("kg_score")) or 0.0) - (_safe_float(left_row.get("kg_score")) or 0.0)
                                if _safe_float(right_row.get("kg_score")) is not None and _safe_float(left_row.get("kg_score")) is not None else None,
                                "left_primary_error_reason": left_row.get("primary_error_reason"),
                                "right_primary_error_reason": right_row.get("primary_error_reason"),
                                "left_auto_flag": left_row.get("auto_flag"),
                                "right_auto_flag": right_row.get("auto_flag"),
                            }
                        )
    return comparisons


def build_factor_effects(trace_rows: Sequence[Dict[str, object]], paired_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    effects: List[Dict[str, object]] = []
    for factor in DESIGN_FACTORS:
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in trace_rows:
            value = _safe_str(row.get(factor))
            if value:
                grouped[value].append(row)
        if len(grouped) < 2:
            continue
        for value, rows in sorted(grouped.items()):
            metric_means = {}
            for metric in QUALITY_METRICS + [
                "design_quality_score",
                "retrieval_score",
                "generation_score",
                "grounding_score",
                "kg_score",
                "calibration_score",
                "cost_efficiency_score",
                "estimated_llm_call_count",
                "estimated_extra_cost_units",
            ]:
                values = [_safe_float(row.get(metric)) for row in rows]
                values = [value for value in values if value is not None]
                metric_means[f"mean_{metric}"] = sum(values) / len(values) if values else None
            relevant_pairs = [
                row for row in paired_rows
                if row.get("factor") == factor and (row.get("left_value") == value or row.get("right_value") == value)
            ]
            pair_deltas = []
            for row in relevant_pairs:
                delta = _safe_float(row.get("score_delta_right_minus_left"))
                if delta is None:
                    continue
                pair_deltas.append(delta if row.get("right_value") == value else -delta)
            controlled_pair_deltas = []
            for row in relevant_pairs:
                if row.get("comparison_type") != "controlled":
                    continue
                delta = _safe_float(row.get("score_delta_right_minus_left"))
                if delta is None:
                    continue
                controlled_pair_deltas.append(delta if row.get("right_value") == value else -delta)
            all_stats = _delta_stats(pair_deltas)
            controlled_stats = _delta_stats(controlled_pair_deltas)
            effects.append(
                {
                    "factor": factor,
                    "value": value,
                    "n_rows": len(rows),
                    "n_questions": len({_safe_str(row.get("question_id")) for row in rows}),
                    **metric_means,
                    "mean_paired_design_quality_delta": all_stats["mean"],
                    "paired_delta_ci95_low": all_stats["ci95_low"],
                    "paired_delta_ci95_high": all_stats["ci95_high"],
                    "paired_win_rate": all_stats["win_rate"],
                    "paired_loss_rate": all_stats["loss_rate"],
                    "paired_tie_rate": all_stats["tie_rate"],
                    "mean_controlled_design_quality_delta": controlled_stats["mean"],
                    "controlled_delta_ci95_low": controlled_stats["ci95_low"],
                    "controlled_delta_ci95_high": controlled_stats["ci95_high"],
                    "controlled_win_rate": controlled_stats["win_rate"],
                    "controlled_loss_rate": controlled_stats["loss_rate"],
                    "n_paired_comparisons": len(pair_deltas),
                    "n_controlled_comparisons": len(controlled_pair_deltas),
                    "failure_count": sum(1 for row in rows if _is_failure(row)),
                    "top_failure_reason": _top_reason(row.get("primary_error_reason") for row in rows),
                }
            )
    return effects


def _is_failure(row: Dict[str, object]) -> bool:
    auto_flag = _safe_str(row.get("auto_flag"))
    primary = _safe_str(row.get("primary_error_reason"))
    if auto_flag:
        return auto_flag not in {"correct", "ok"}
    return bool(primary and primary != "ok")


def _top_reason(values: Iterable[object]) -> str:
    counter = Counter(_safe_str(value) for value in values if _safe_str(value))
    return counter.most_common(1)[0][0] if counter else ""


def _taxonomy_details(component: object) -> Dict[str, str]:
    key = _safe_str(component) or "unknown"
    return FAILURE_TAXONOMY.get(key, FAILURE_TAXONOMY["unknown"])


def _delta_stats(values: Sequence[float]) -> Dict[str, object]:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "win_rate": None,
            "loss_rate": None,
            "tie_rate": None,
        }
    mean_value = sum(numbers) / len(numbers)
    if len(numbers) > 1:
        variance = sum((value - mean_value) ** 2 for value in numbers) / (len(numbers) - 1)
        ci = 1.96 * math.sqrt(variance / len(numbers))
    else:
        ci = 0.0
    wins = sum(1 for value in numbers if value > 0.0)
    losses = sum(1 for value in numbers if value < 0.0)
    ties = len(numbers) - wins - losses
    return {
        "mean": mean_value,
        "ci95_low": mean_value - ci,
        "ci95_high": mean_value + ci,
        "win_rate": wins / len(numbers),
        "loss_rate": losses / len(numbers),
        "tie_rate": ties / len(numbers),
    }


def _paired_improvements_for_question(question_id: str, paired_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    candidates = []
    for row in paired_rows:
        if _safe_str(row.get("question_id")) != question_id:
            continue
        delta = _safe_float(row.get("score_delta_right_minus_left"))
        if delta is None or abs(delta) < 0.05:
            continue
        if delta > 0:
            candidates.append({"factor": row.get("factor"), "better_value": row.get("right_value"), "worse_value": row.get("left_value"), "delta": delta})
        else:
            candidates.append({"factor": row.get("factor"), "better_value": row.get("left_value"), "worse_value": row.get("right_value"), "delta": abs(delta)})
    return sorted(candidates, key=lambda item: float(item["delta"]), reverse=True)[:5]


def build_failure_attribution(trace_rows: Sequence[Dict[str, object]], paired_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in trace_rows:
        if not _is_failure(row):
            continue
        suspects = _suspect_design_factors(row)
        paired = _paired_improvements_for_question(_safe_str(row.get("question_id")), paired_rows)
        if paired:
            suspects.extend(f"{item['factor']}={item['better_value']} fixed/improved this question" for item in paired[:3])
        best_counterfactual = paired[0] if paired else {}
        likely_component = _likely_component(row)
        taxonomy = _taxonomy_details(likely_component)
        causal_evidence = _causal_evidence(row, paired)
        rows.append(
            {
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "question_type": row.get("question_type"),
                "experiment": row.get("experiment"),
                "auto_flag": row.get("auto_flag"),
                "primary_error_reason": row.get("primary_error_reason"),
                "claim_diagnostic": row.get("claim_diagnostic"),
                "likely_component": likely_component,
                "component_label": taxonomy["label"],
                "component_explanation": taxonomy["explanation"],
                "advisor_component": taxonomy["advisor_component"],
                "advisor_issue": taxonomy["advisor_issue"],
                "advisor_recommendation": taxonomy["recommendation"],
                "advisor_next_experiment": taxonomy["next_experiment"],
                "causal_explanation": _causal_explanation(row, likely_component, taxonomy, paired, causal_evidence),
                "causal_evidence": "; ".join(causal_evidence),
                "retrieval_score": row.get("retrieval_score"),
                "generation_score": row.get("generation_score"),
                "grounding_score": row.get("grounding_score"),
                "kg_score": row.get("kg_score"),
                "calibration_score": row.get("calibration_score"),
                "mrr_at_k": row.get("mrr_at_k"),
                "recall_at_k": row.get("recall_at_k"),
                "ndcg_at_k": row.get("ndcg_at_k"),
                "context_claim_recall": row.get("context_claim_recall"),
                "grounded_claim_ratio": row.get("grounded_claim_ratio"),
                "hallucinated_claim_ratio": row.get("hallucinated_claim_ratio"),
                "kg_graph_gain_at_k": row.get("kg_graph_gain_at_k"),
                "kg_graph_noise_at_k": row.get("kg_graph_noise_at_k"),
                "evidence_attribution_f1": row.get("evidence_attribution_f1"),
                "retriever": row.get("retriever"),
                "top_k": row.get("top_k"),
                "chunking_strategy": row.get("chunking_strategy"),
                "context_mode": row.get("context_mode"),
                "query_augmentation": row.get("query_augmentation"),
                "answer_mode": row.get("answer_mode"),
                "llm_model": row.get("llm_model"),
                "judge_model": row.get("judge_model"),
                "kg_enabled": row.get("kg_enabled"),
                "kg_profile": row.get("kg_profile"),
                "kg_algorithm": row.get("kg_algorithm"),
                "rerank_top_n": row.get("rerank_top_n"),
                "rerank_weight": row.get("rerank_weight"),
                "self_rag_retry_on_weak_evidence": row.get("self_rag_retry_on_weak_evidence"),
                "self_rag_critique": row.get("self_rag_critique"),
                "design_factor_suspects": "; ".join(dict.fromkeys(suspects)),
                "best_observed_fix_factor": best_counterfactual.get("factor"),
                "best_observed_fix_value": best_counterfactual.get("better_value"),
                "best_observed_fix_delta": best_counterfactual.get("delta"),
                "recommended_next_ablation": _recommended_next_ablation(row, paired) or taxonomy["next_experiment"],
                "evidence": _failure_evidence(row),
            }
        )
    return rows


def _suspect_design_factors(row: Dict[str, object]) -> List[str]:
    suspects: List[str] = []
    context_recall = _safe_float(row.get("context_claim_recall"))
    recall = _safe_float(row.get("recall_at_k"))
    mrr = _safe_float(row.get("mrr_at_k"))
    grounded = _safe_float(row.get("grounded_claim_ratio"))
    hallucinated = _safe_float(row.get("hallucinated_claim_ratio"))
    kg_noise = _safe_float(row.get("kg_graph_noise_at_k"))
    kg_gain = _safe_float(row.get("kg_graph_gain_at_k"))
    attribution = _safe_float(row.get("evidence_attribution_f1"))
    primary = _safe_str(row.get("primary_error_reason"))
    if context_recall is not None and context_recall < 0.5:
        suspects.extend(["retriever", "top_k", "chunking_strategy", "chunk_size", "overlap", "document_parsing"])
    if mrr is not None and recall is not None and mrr < 0.5 <= recall:
        suspects.extend(["rerank_top_n", "rerank_weight", "retriever score fusion"])
    if grounded is not None and grounded < 0.7 and context_recall is not None and context_recall >= 0.7:
        suspects.extend(["answer_mode", "prompt", "llm_model"])
    if hallucinated is not None and hallucinated > 0.25:
        suspects.extend(["answer_mode", "prompt", "llm_model", "context_mode"])
    if kg_noise is not None and kg_noise > 0.25:
        suspects.extend(["kg_profile", "kg_graph_weight", "kg_quality_threshold", "kg_algorithm"])
    if kg_gain is not None and kg_gain <= 0 and row.get("kg_enabled"):
        suspects.extend(["kg_profile", "kg_algorithm", "kg_max_added_chunks"])
    if attribution is not None and attribution < 0.7:
        suspects.extend(["answer_mode", "citation schema", "prompt"])
    if primary in {"over_answering", "false_refusal"}:
        suspects.extend(["abstain_on_weak_evidence", "decision thresholds", "runtime_retrieval_evaluator"])
    return suspects or ["manual_review"]


def _likely_component(row: Dict[str, object]) -> str:
    primary = _safe_str(row.get("primary_error_reason"))
    secondary = _safe_str(row.get("secondary_error_reason"))
    claim = _safe_str(row.get("claim_diagnostic"))
    recall = _safe_float(row.get("recall_at_k"))
    mrr = _safe_float(row.get("mrr_at_k"))
    context_recall = _safe_float(row.get("context_claim_recall"))
    retrieval_score = _safe_float(row.get("retrieval_score"))
    grounding_score = _safe_float(row.get("grounding_score"))
    generation_score = _safe_float(row.get("generation_score"))
    calibration_score = _safe_float(row.get("calibration_score"))
    if "parse" in primary or "empty_section" in primary:
        return "document_parsing"
    if "chunk" in primary:
        return "chunking"
    if "answer_incomplete_from_good_context" in primary and context_recall is not None and context_recall >= 0.8 and grounding_score is not None and grounding_score >= 0.8:
        return "answer_generation"
    if "gold_missing" in primary or "expected_answer_missing" in primary or "candidate_generation" in primary:
        return "candidate_generation"
    if "gold_present_but_not_ranked_first" in primary or "ranking" in primary or "rerank" in primary:
        return "candidate_ranking"
    if "weak_evidence" in primary or "missing_evidence" in primary or "retrieval" in primary:
        return "candidate_generation" if retrieval_score is not None and retrieval_score < 0.35 else "context_selection"
    if recall is not None and recall < 0.35:
        return "candidate_generation"
    if context_recall is not None and context_recall < 0.5:
        return "context_selection"
    if mrr is not None and recall is not None and mrr < 0.5 <= recall:
        return "candidate_ranking"
    if "hierarchy_near_miss" in primary or "same_branch" in primary or "same_class" in primary:
        return "taxonomy_disambiguation"
    if "calibration" in primary or "confidence" in primary or (calibration_score is not None and calibration_score < 0.4):
        return "calibration"
    if "hallucination" in primary or "hallucinated" in claim:
        return "answer_generation"
    if "incomplete" in primary or "claim" in primary:
        return "answer_generation" if generation_score is not None and generation_score < 0.5 else "context_selection"
    if "attribution" in primary or "citation" in primary or (grounding_score is not None and grounding_score < 0.5):
        return "citation_attribution"
    if "ambiguous" in primary or "benchmark" in primary or "ambiguous" in secondary:
        return "benchmark_ambiguity"
    kg_noise = _safe_float(row.get("kg_graph_noise_at_k"))
    if kg_noise is not None and kg_noise > 0.25:
        return "kg_noise"
    kg_gain = _safe_float(row.get("kg_graph_gain_at_k"))
    if row.get("kg_enabled") and kg_gain is not None and kg_gain <= 0:
        return "kg_expansion"
    return "unknown"


def _recommended_next_ablation(row: Dict[str, object], paired: Sequence[Dict[str, object]]) -> str:
    if paired:
        best = paired[0]
        return f"Hold question fixed and compare {best['factor']}={best['worse_value']} vs {best['factor']}={best['better_value']}."
    suspects = _suspect_design_factors(row)
    if "top_k" in suspects:
        return "Run top_k 5 vs 10/15 with the same chunking and retriever."
    if "kg_profile" in suspects:
        return "Compare conservative, balanced, exploratory, ppr_only, and direct_only KG profiles."
    if "answer_mode" in suspects:
        return "Freeze retrieval and compare extractive, cite_first, and claim_checklist answer modes."
    if "chunking_strategy" in suspects:
        return "Compare fixed_words, by_section, and by_paragraph with the same retriever/top_k."
    return "Add this question to a focused ablation set and vary one design factor at a time."


def _fmt_metric(value: object) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _causal_evidence(row: Dict[str, object], paired: Sequence[Dict[str, object]]) -> List[str]:
    evidence: List[str] = []
    mrr = _safe_float(row.get("mrr_at_k"))
    recall = _safe_float(row.get("recall_at_k"))
    ndcg = _safe_float(row.get("ndcg_at_k"))
    context_recall = _safe_float(row.get("context_claim_recall"))
    grounded = _safe_float(row.get("grounded_claim_ratio"))
    hallucinated = _safe_float(row.get("hallucinated_claim_ratio"))
    kg_gain = _safe_float(row.get("kg_graph_gain_at_k"))
    kg_noise = _safe_float(row.get("kg_graph_noise_at_k"))
    attribution = _safe_float(row.get("evidence_attribution_f1"))
    confidence = _safe_float(row.get("prediction_confidence"))
    if mrr is not None and mrr < 0.35:
        evidence.append(f"low MRR@K={_fmt_metric(mrr)}")
    if recall is not None and recall < 0.5:
        evidence.append(f"low Recall@K={_fmt_metric(recall)}")
    if ndcg is not None and ndcg < 0.4:
        evidence.append(f"low NDCG@K={_fmt_metric(ndcg)}")
    if context_recall is not None and context_recall < 0.5:
        evidence.append(f"low context-claim recall={_fmt_metric(context_recall)}")
    if grounded is not None and grounded < 0.7:
        evidence.append(f"low grounded-claim ratio={_fmt_metric(grounded)}")
    if hallucinated is not None and hallucinated > 0.25:
        evidence.append(f"high hallucinated-claim ratio={_fmt_metric(hallucinated)}")
    if kg_gain is not None and kg_gain <= 0:
        evidence.append(f"no positive KG gain={_fmt_metric(kg_gain)}")
    if kg_noise is not None and kg_noise > 0.25:
        evidence.append(f"high KG graph noise={_fmt_metric(kg_noise)}")
    if attribution is not None and attribution < 0.7:
        evidence.append(f"weak evidence attribution F1={_fmt_metric(attribution)}")
    if confidence is not None and confidence > 0.8 and _is_failure(row):
        evidence.append(f"high confidence on failed row={_fmt_metric(confidence)}")
    if paired:
        fixes = [
            f"{item.get('factor')} {item.get('worse_value')} -> {item.get('better_value')} improved same question by +{_fmt_metric(item.get('delta'))}"
            for item in paired[:3]
        ]
        evidence.extend(fixes)
    return evidence or ["automatic metrics did not expose a strong single signal"]


def _causal_explanation(
    row: Dict[str, object],
    likely_component: str,
    taxonomy: Dict[str, str],
    paired: Sequence[Dict[str, object]],
    evidence: Sequence[str],
) -> str:
    label = taxonomy.get("label", likely_component)
    question_id = _safe_str(row.get("question_id")) or "this question"
    metric_clause = ", ".join(evidence[:4])
    current_config = _current_config_phrase(row)
    if paired:
        fix_clause = _paired_fix_phrase(paired)
        counterfactual = (
            f"For the same question_id, {fix_clause} performed better than the current setting"
            f"{current_config}."
        )
        confidence = "This is stronger than a metric-only diagnosis because it uses an observed same-question counterfactual."
    else:
        counterfactual = "No same-question counterfactual improvement was observed in this run."
        confidence = "Treat this as a metric-based diagnosis and validate it with the recommended ablation."
    templates = {
        "candidate_generation": (
            f"Retrieval coverage is probably the limiting factor for {question_id}. {counterfactual} "
            f"The current row shows {metric_clause}, which means the downstream selector or generator likely did not receive enough useful candidates/evidence. {confidence}"
        ),
        "context_selection": (
            f"Evidence selection looks incomplete for {question_id}. {counterfactual} "
            f"The signal is {metric_clause}; this suggests the system retrieved some material but did not assemble enough of the right context for a complete answer. {confidence}"
        ),
        "candidate_ranking": (
            f"Ranking or final selection is the likely bottleneck for {question_id}. {counterfactual} "
            f"The row shows {metric_clause}, so the issue is less about making candidates available and more about ordering or choosing among them. {confidence}"
        ),
        "taxonomy_disambiguation": (
            f"The failure for {question_id} looks like taxonomy-level disambiguation rather than a broad retrieval miss. {counterfactual} "
            f"The observed signals are {metric_clause}; this points to confusion among related labels/classes that need hierarchy-aware contrast. {confidence}"
        ),
        "kg_expansion": (
            f"KG expansion did not appear to add useful relation evidence for {question_id}. {counterfactual} "
            f"The row shows {metric_clause}, so the graph traversal/profile may not be recovering the missing facts. {confidence}"
        ),
        "kg_noise": (
            f"Graph expansion is likely introducing distracting evidence for {question_id}. {counterfactual} "
            f"The strongest signal is {metric_clause}; this suggests KG expansion should be tightened before increasing graph context. {confidence}"
        ),
        "answer_generation": (
            f"Answer synthesis is the likely failure point for {question_id}. {counterfactual} "
            f"The row shows {metric_clause}, which suggests retrieval/context may be sufficient but the generated answer still misses or invents claims. {confidence}"
        ),
        "citation_attribution": (
            f"Evidence attribution is the weak link for {question_id}. {counterfactual} "
            f"The row shows {metric_clause}; the answer may be close, but its claims are not reliably tied back to retrieved chunks or source spans. {confidence}"
        ),
        "calibration": (
            f"Confidence calibration is unsafe for {question_id}. {counterfactual} "
            f"The row shows {metric_clause}, meaning the score/confidence should not be used for automatic acceptance without calibration or review thresholds. {confidence}"
        ),
        "document_parsing": (
            f"Document parsing may be the upstream bottleneck for {question_id}. {counterfactual} "
            f"The diagnostic signal is {metric_clause}; if parsing produced weak or fragmented text, retrieval and KG expansion inherit that limitation. {confidence}"
        ),
        "chunking": (
            f"Chunking or segmentation is likely constraining evidence for {question_id}. {counterfactual} "
            f"The row shows {metric_clause}, suggesting the relevant facts may be split, merged with noise, or detached from useful headings/metadata. {confidence}"
        ),
        "benchmark_ambiguity": (
            f"The row {question_id} may be ambiguous as an evaluation item. {counterfactual} "
            f"The available signals are {metric_clause}; before blaming one system component, check whether the query/reference needs more context or accepted alternatives. {confidence}"
        ),
        "unknown": (
            f"The bottleneck for {question_id} is not identifiable from the current metrics alone. {counterfactual} "
            f"The available signal is {metric_clause}. Add diagnostics or run a one-factor ablation before making a strong component claim."
        ),
    }
    return templates.get(
        likely_component,
        f"{label} is likely the bottleneck for {question_id}. {counterfactual} The row shows {metric_clause}. {confidence}",
    )


def _current_config_phrase(row: Dict[str, object]) -> str:
    parts = []
    for key in ["retriever", "top_k", "chunking_strategy", "query_augmentation", "kg_profile", "context_mode", "answer_mode"]:
        value = _safe_str(row.get(key))
        if value:
            parts.append(f"{key}={value}")
    return f" ({', '.join(parts[:5])})" if parts else ""


def _paired_fix_phrase(paired: Sequence[Dict[str, object]]) -> str:
    fixes = [
        f"{item.get('factor')}={item.get('better_value')}"
        for item in paired[:3]
        if item.get("factor")
    ]
    if not fixes:
        return "an alternative configuration"
    if len(fixes) == 1:
        return f"a configuration with {fixes[0]}"
    return f"configurations with {', '.join(fixes[:-1])}, and {fixes[-1]}"


def _failure_evidence(row: Dict[str, object]) -> str:
    parts = []
    for key in [
        "context_claim_recall",
        "mrr_at_k",
        "recall_at_k",
        "grounded_claim_ratio",
        "hallucinated_claim_ratio",
        "kg_graph_gain_at_k",
        "kg_graph_noise_at_k",
        "prediction_confidence",
    ]:
        value = row.get(key)
        if value not in {None, ""}:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _rate(rows: Sequence[Dict[str, object]], key: str, predicate) -> float | None:
    values = [_safe_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(1 for value in values if predicate(value)) / len(values)


def _top_values(rows: Sequence[Dict[str, object]], key: str, limit: int = 3) -> str:
    counter = Counter(_safe_str(row.get(key)) for row in rows if _safe_str(row.get(key)))
    return "; ".join(f"{value} ({count})" for value, count in counter.most_common(limit))


def _dominant_value(rows: Sequence[Dict[str, object]], key: str) -> str:
    counter = Counter(_safe_str(row.get(key)) for row in rows if _safe_str(row.get(key)))
    return counter.most_common(1)[0][0] if counter else ""


def _truthy_share(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = [_safe_str(row.get(key)).lower() for row in rows if row.get(key) not in {None, ""}]
    if not values:
        return 0.0
    return sum(1 for value in values if value in {"true", "1", "yes"}) / len(values)


def _top_observed_fixes(rows: Sequence[Dict[str, object]], limit: int = 3) -> str:
    counter = Counter()
    deltas: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        factor = _safe_str(row.get("best_observed_fix_factor"))
        value = _safe_str(row.get("best_observed_fix_value"))
        if not factor or not value:
            continue
        key = f"{factor}={value}"
        counter[key] += 1
        delta = _safe_float(row.get("best_observed_fix_delta"))
        if delta is not None:
            deltas[key].append(delta)
    parts = []
    for key, count in counter.most_common(limit):
        mean_delta = sum(deltas[key]) / len(deltas[key]) if deltas.get(key) else None
        suffix = f", mean_delta=+{_fmt_metric(mean_delta)}" if mean_delta is not None else ""
        parts.append(f"{key} ({count}{suffix})")
    return "; ".join(parts)


def _paired_group_insights(rows: Sequence[Dict[str, object]], paired_rows: Sequence[Dict[str, object]], limit: int = 8) -> List[Dict[str, object]]:
    question_ids = {_safe_str(row.get("question_id")) for row in rows if _safe_str(row.get("question_id"))}
    dominant_chunking = _dominant_value(rows, "chunking_strategy")
    buckets: Dict[tuple[str, str], Dict[str, object]] = {}
    for row in paired_rows:
        if _safe_str(row.get("question_id")) not in question_ids:
            continue
        delta = _safe_float(row.get("score_delta_right_minus_left"))
        if delta is None or abs(delta) < 0.03:
            continue
        if delta > 0:
            factor = _safe_str(row.get("factor"))
            better_value = _safe_str(row.get("right_value"))
            worse_value = _safe_str(row.get("left_value"))
        else:
            factor = _safe_str(row.get("factor"))
            better_value = _safe_str(row.get("left_value"))
            worse_value = _safe_str(row.get("right_value"))
        if not factor or not better_value:
            continue
        if factor in {"chunk_size", "chunk_overlap"} and dominant_chunking in {"by_section", "by_paragraph", "cpv_entry"}:
            continue
        key = (factor, better_value)
        bucket = buckets.setdefault(
            key,
            {
                "factor": factor,
                "better_value": better_value,
                "count": 0,
                "controlled_count": 0,
                "deltas": [],
                "worse_values": Counter(),
            },
        )
        bucket["count"] = int(bucket["count"]) + 1
        if row.get("comparison_type") == "controlled":
            bucket["controlled_count"] = int(bucket["controlled_count"]) + 1
        bucket["deltas"].append(abs(delta))
        bucket["worse_values"][worse_value] += 1
    insights: List[Dict[str, object]] = []
    for bucket in buckets.values():
        deltas = list(bucket.get("deltas") or [])
        insights.append(
            {
                "factor": bucket["factor"],
                "better_value": bucket["better_value"],
                "count": bucket["count"],
                "controlled_count": bucket["controlled_count"],
                "mean_delta": sum(deltas) / len(deltas) if deltas else None,
                "worse_values": "; ".join(f"{value} ({count})" for value, count in bucket["worse_values"].most_common(3)),
            }
        )
    insights = sorted(
        insights,
        key=lambda item: (int(item.get("controlled_count") or 0), int(item.get("count") or 0), float(item.get("mean_delta") or 0.0)),
        reverse=True,
    )
    if any(item.get("factor") == "chunking_strategy" for item in insights):
        insights = [item for item in insights if item.get("factor") not in {"chunk_size", "chunk_overlap"}]
    return insights[:limit]


def _format_pair_insights(insights: Sequence[Dict[str, object]], limit: int = 4) -> str:
    parts = []
    for item in insights[:limit]:
        controlled = f", controlled={item.get('controlled_count')}" if item.get("controlled_count") else ""
        confidence = _insight_confidence_label(item)
        parts.append(
            f"{item.get('factor')}={item.get('better_value')} "
            f"[{confidence}] ({item.get('count')} cases{controlled}, mean_delta=+{_fmt_metric(item.get('mean_delta'))})"
        )
    return "; ".join(parts)


def _group_metric_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "mean_mrr_at_k": _mean_present(row.get("mrr_at_k") for row in rows),
        "mean_recall_at_k": _mean_present(row.get("recall_at_k") for row in rows),
        "mean_ndcg_at_k": _mean_present(row.get("ndcg_at_k") for row in rows),
        "mean_context_claim_recall": _mean_present(row.get("context_claim_recall") for row in rows),
        "mean_grounded_claim_ratio": _mean_present(row.get("grounded_claim_ratio") for row in rows),
        "mean_hallucinated_claim_ratio": _mean_present(row.get("hallucinated_claim_ratio") for row in rows),
        "mean_kg_graph_gain_at_k": _mean_present(row.get("kg_graph_gain_at_k") for row in rows),
        "mean_kg_graph_noise_at_k": _mean_present(row.get("kg_graph_noise_at_k") for row in rows),
        "mean_evidence_attribution_f1": _mean_present(row.get("evidence_attribution_f1") for row in rows),
        "low_mrr_rate": _rate(rows, "mrr_at_k", lambda value: value < 0.35),
        "low_recall_rate": _rate(rows, "recall_at_k", lambda value: value < 0.5),
        "low_context_claim_recall_rate": _rate(rows, "context_claim_recall", lambda value: value < 0.5),
        "high_hallucination_rate": _rate(rows, "hallucinated_claim_ratio", lambda value: value > 0.25),
        "high_kg_noise_rate": _rate(rows, "kg_graph_noise_at_k", lambda value: value > 0.25),
    }


def _group_evidence_sentence(metrics: Dict[str, object]) -> str:
    parts = []
    for label, key in [
        ("mean MRR@K", "mean_mrr_at_k"),
        ("mean Recall@K", "mean_recall_at_k"),
        ("mean NDCG@K", "mean_ndcg_at_k"),
        ("mean context-claim recall", "mean_context_claim_recall"),
        ("mean grounded-claim ratio", "mean_grounded_claim_ratio"),
        ("mean hallucinated-claim ratio", "mean_hallucinated_claim_ratio"),
        ("mean KG noise", "mean_kg_graph_noise_at_k"),
    ]:
        value = _safe_float(metrics.get(key))
        if value is not None:
            parts.append(f"{label}={_fmt_metric(value)}")
    for label, key in [
        ("low-MRR cases", "low_mrr_rate"),
        ("low-recall cases", "low_recall_rate"),
        ("low-context-recall cases", "low_context_claim_recall_rate"),
        ("high-hallucination cases", "high_hallucination_rate"),
        ("high-KG-noise cases", "high_kg_noise_rate"),
    ]:
        value = _safe_float(metrics.get(key))
        if value is not None and value > 0:
            parts.append(f"{label}={_fmt_metric(value)}")
    return ", ".join(parts) if parts else "no strong aggregate metric signal"


def _insight_value(insights: Sequence[Dict[str, object]], factor: str) -> str:
    for item in insights:
        if item.get("factor") == factor:
            return _safe_str(item.get("better_value"))
    return ""


def _insight_entry(insights: Sequence[Dict[str, object]], factor: str) -> Dict[str, object]:
    for item in insights:
        if item.get("factor") == factor:
            return dict(item)
    return {}


def _insight_confidence_label(item: Dict[str, object]) -> str:
    controlled = int(item.get("controlled_count") or 0)
    count = int(item.get("count") or 0)
    if controlled >= 2 or (controlled >= 1 and count >= 5) or count >= 8:
        return "strong"
    if controlled >= 1 or count >= 3:
        return "moderate"
    return "weak"


def _has_truthy(rows: Sequence[Dict[str, object]], key: str) -> bool:
    return any(_safe_str(row.get(key)).lower() in {"true", "1", "yes"} for row in rows)


def _group_design_causes(
    component: str,
    rows: Sequence[Dict[str, object]],
    metrics: Dict[str, object],
    insights: Sequence[Dict[str, object]],
) -> List[str]:
    causes: List[str] = []
    mean_mrr = _safe_float(metrics.get("mean_mrr_at_k"))
    mean_recall = _safe_float(metrics.get("mean_recall_at_k"))
    mean_ndcg = _safe_float(metrics.get("mean_ndcg_at_k"))
    mean_context = _safe_float(metrics.get("mean_context_claim_recall"))
    mean_grounded = _safe_float(metrics.get("mean_grounded_claim_ratio"))
    mean_hallucinated = _safe_float(metrics.get("mean_hallucinated_claim_ratio"))
    low_recall_rate = _safe_float(metrics.get("low_recall_rate"))
    low_mrr_rate = _safe_float(metrics.get("low_mrr_rate"))
    low_context_rate = _safe_float(metrics.get("low_context_claim_recall_rate"))
    high_hallucination_rate = _safe_float(metrics.get("high_hallucination_rate"))
    high_kg_noise_rate = _safe_float(metrics.get("high_kg_noise_rate"))
    question_type = _safe_str(rows[0].get("question_type") if rows else "")
    dominant_retriever = _dominant_value(rows, "retriever")
    dominant_chunking = _dominant_value(rows, "chunking_strategy")
    dominant_top_k = _dominant_value(rows, "top_k")
    dominant_query_aug = _dominant_value(rows, "query_augmentation")
    dominant_context_mode = _dominant_value(rows, "context_mode")
    dominant_answer_mode = _dominant_value(rows, "answer_mode")
    answer_generation_enabled_share = _truthy_share(rows, "answer_generation_enabled")
    dominant_llm_model = _dominant_value(rows, "llm_model")
    dominant_kg_profile = _dominant_value(rows, "kg_profile")
    dominant_rerank_top_n = _dominant_value(rows, "rerank_top_n")
    kg_enabled_share = _truthy_share(rows, "kg_enabled")
    retry_enabled_share = _truthy_share(rows, "self_rag_retry_on_weak_evidence")
    critique_enabled_share = _truthy_share(rows, "self_rag_critique")
    retrieval_looks_sufficient = (
        mean_context is not None and mean_context >= 0.85
        and mean_grounded is not None and mean_grounded >= 0.85
        and mean_hallucinated is not None and mean_hallucinated <= 0.1
    )
    retriever_insight = _insight_entry(insights, "retriever")
    better_retriever = _safe_str(retriever_insight.get("better_value"))
    retriever_confidence = _insight_confidence_label(retriever_insight) if retriever_insight else ""
    better_chunking = _insight_value(insights, "chunking_strategy")
    better_top_k = _insight_value(insights, "top_k")
    better_hybrid_alpha = _insight_value(insights, "hybrid_alpha")
    better_rerank_top_n = _insight_value(insights, "rerank_top_n")
    better_rerank_weight = _insight_value(insights, "rerank_weight")
    better_context = _insight_value(insights, "context_mode")
    better_query_aug = _insight_value(insights, "query_augmentation")
    better_answer_mode = _insight_value(insights, "answer_mode")
    better_llm_model = _insight_value(insights, "llm_model")
    better_kg_profile = _insight_value(insights, "kg_profile")
    better_kg_enabled = _insight_value(insights, "kg_enabled")
    better_kg_algorithm = _insight_value(insights, "kg_algorithm")
    if component == "answer_generation" and retrieval_looks_sufficient:
        if dominant_answer_mode in {"", "extractive", "extractive_answer"} or answer_generation_enabled_share == 0.0:
            causes.append("Retrieved evidence already looks sufficient, and answer generation is effectively disabled by design for this group; incomplete final answers are more likely caused by extractive output limits than by retrieval.")
        elif better_answer_mode:
            causes.append(f"Retrieved evidence already looks sufficient, and matched comparisons favor answer_mode={better_answer_mode}; this points to answer-side prompting/synthesis rather than retrieval.")
        elif better_llm_model:
            causes.append(f"Retrieved evidence already looks sufficient, and matched comparisons favor llm_model={better_llm_model}; current answer quality is more likely model-limited than retrieval-limited.")
        elif not dominant_llm_model and answer_generation_enabled_share > 0.0:
            causes.append("Retrieved evidence already looks sufficient, and answer generation appears expected, but the answer-side model/config is missing or not surfaced correctly; verify the LLM setup for this experiment.")
        else:
            causes.append("Retrieved evidence already looks sufficient, so the remaining bottleneck is more likely answer synthesis or prompt completeness than retrieval coverage.")
    if better_retriever and better_retriever != dominant_retriever and not (component == "answer_generation" and retrieval_looks_sufficient):
        current = f" over the current retriever={dominant_retriever}" if dominant_retriever and dominant_retriever != better_retriever else ""
        causes.append(f"Retriever type is a {retriever_confidence}-confidence mismatch hypothesis for this case group: matched comparisons favor retriever={better_retriever}{current}.")
    elif dominant_retriever in {"bm25", "tfidf"} and low_recall_rate is not None and low_recall_rate > 0.6 and question_type in {"multi_hop", "relation", "summary", "global"} and not (component == "answer_generation" and retrieval_looks_sufficient):
        causes.append(f"The current retriever={dominant_retriever} is purely lexical, while this group has widespread low recall on {question_type} questions; hybrid or dense retrieval is a plausible missing capability.")
    elif dominant_retriever == "dense" and component in {"taxonomy_disambiguation", "candidate_generation"} and low_recall_rate is not None and low_recall_rate > 0.5 and not (component == "answer_generation" and retrieval_looks_sufficient):
        causes.append("The current dense retriever may miss exact domain wording or labels; compare BM25/TF-IDF or hybrid retrieval before blaming the generator.")
    if better_chunking and better_chunking != dominant_chunking and not (component == "answer_generation" and retrieval_looks_sufficient):
        causes.append(f"Chunking/segmentation is implicated: chunking_strategy={better_chunking} improves related cases, so current chunks may split or dilute required evidence.")
    elif dominant_chunking in {"fixed_words", "fixed_tokens"} and component in {"candidate_generation", "context_selection"} and low_recall_rate is not None and low_recall_rate > 0.5 and not (component == "answer_generation" and retrieval_looks_sufficient):
        causes.append(f"Chunking is a plausible bottleneck: the current chunking_strategy={dominant_chunking} can separate headings from evidence, which often hurts retrieval coverage compared with document-aware chunking.")
    if better_top_k and better_top_k != dominant_top_k and not (component == "answer_generation" and retrieval_looks_sufficient):
        causes.append(f"Retrieval depth is likely too shallow: matched comparisons favor top_k={better_top_k} over the current top_k={dominant_top_k or 'unknown'}.")
    elif dominant_top_k and dominant_top_k.isdigit() and int(dominant_top_k) <= 5 and low_recall_rate is not None and low_recall_rate > 0.5 and not (component == "answer_generation" and retrieval_looks_sufficient):
        causes.append(f"The current top_k={dominant_top_k} is probably too small for this failure group; low recall persists before the answer stage even starts.")
    if better_hybrid_alpha and not (component == "answer_generation" and retrieval_looks_sufficient):
        causes.append(f"Fusion balance matters here: matched comparisons favor hybrid_alpha={better_hybrid_alpha}, which suggests the current lexical/semantic weighting is off for this case mix.")
    if better_context:
        causes.append(f"Context organization matters for this group: context_mode={better_context} is a better observed alternative.")
    elif dominant_context_mode in {"ranked", "dedupe_section"} and mean_context is not None and mean_context < 0.7 and component == "context_selection":
        causes.append(f"The current context_mode={dominant_context_mode} may be dropping or under-organizing supporting evidence, so the generator never sees enough complete context.")
    if better_query_aug:
        causes.append(f"Query augmentation is likely underpowered: matched comparisons favor query_augmentation={better_query_aug} over the current setting={dominant_query_aug or 'none'}.")
    elif dominant_query_aug in {"", "none"} and component in {"candidate_generation", "context_selection"} and question_type in {"multi_hop", "relation", "summary", "global"}:
        causes.append("LLM query augmentation is disabled for a question type that often needs reformulation or bridge terms; HyDE or LLM expansion is a strong missing component.")
    if better_answer_mode:
        causes.append(f"Answer mode/prompting is implicated: answer_mode={better_answer_mode} is a better observed alternative.")
    elif dominant_answer_mode in {"", "extractive", "extractive_answer"} and mean_context is not None and mean_context >= 0.8 and mean_grounded is not None and mean_grounded >= 0.8 and not (component == "answer_generation" and retrieval_looks_sufficient):
        causes.append("The answer side is too weak for the available evidence: the current answer mode is extractive/no-LLM, so complete synthesis is likely being lost even when context is good.")
    if better_llm_model:
        causes.append(f"Model choice appears relevant: matched comparisons favor llm_model={better_llm_model}, so current generation quality may be model-limited.")
    elif not dominant_llm_model and dominant_answer_mode not in {"", "extractive", "extractive_answer"}:
        causes.append("An answer-side LLM mode is configured without a visible model identifier; verify that the intended model is actually enabled for this experiment.")
    if better_kg_enabled.lower() in {"true", "1"}:
        causes.append("KG retrieval is a likely missing component: matched comparisons improve when KG is enabled, so the current no-KG setup is leaving useful relational evidence out.")
    if better_kg_profile:
        causes.append(f"KG traversal/profile is a likely factor: matched comparisons favor kg_profile={better_kg_profile} over the current profile={dominant_kg_profile or 'none'}.")
    if better_kg_algorithm:
        causes.append(f"KG traversal algorithm matters here: matched comparisons favor kg_algorithm={better_kg_algorithm}, so the current graph walk may be selecting the wrong neighbors.")
    if kg_enabled_share < 0.5 and question_type in {"multi_hop", "relation"} and component in {"candidate_generation", "context_selection"} and low_recall_rate is not None and low_recall_rate > 0.5:
        causes.append("KG is effectively disabled on a multi-hop/relation-heavy failure group; missing graph expansion is a credible reason why bridge evidence never enters context.")
    if mean_mrr is not None and mean_mrr >= 0.7 and mean_recall is not None and mean_recall < 0.5 and not (component == "answer_generation" and retrieval_looks_sufficient):
        causes.append("The first useful item appears early, but coverage is low; this points to missing additional evidence rather than a pure top-rank problem.")
    if mean_mrr is not None and mean_mrr < 0.45 and mean_recall is not None and mean_recall >= 0.7:
        causes.append("Relevant evidence is often present but poorly ordered; reranking or score fusion is a stronger hypothesis than changing chunking first.")
    if component == "candidate_ranking" and dominant_rerank_top_n in {"", "0"}:
        causes.append("Reranking is effectively disabled for a group where evidence is already present; add lexical or cross-encoder reranking before changing the retriever.")
    if better_rerank_top_n or better_rerank_weight:
        parts = []
        if better_rerank_top_n:
            parts.append(f"rerank_top_n={better_rerank_top_n}")
        if better_rerank_weight:
            parts.append(f"rerank_weight={better_rerank_weight}")
        causes.append(f"Ranking control knobs matter here: matched comparisons favor {', '.join(parts)}.")
    if mean_context is not None and mean_context >= 0.8 and mean_grounded is not None and mean_grounded >= 0.8:
        modes = Counter(_safe_str(row.get("answer_mode")) for row in rows if _safe_str(row.get("answer_mode")))
        top_mode = modes.most_common(1)[0][0] if modes else ""
        if (top_mode in {"extractive", "extractive_answer"} or not top_mode) and not (component == "answer_generation" and retrieval_looks_sufficient):
            causes.append("Context and grounding look good, but answers are incomplete; LLM generation or a stronger answer_mode is likely needed instead of extractive synthesis.")
        elif not (component == "answer_generation" and retrieval_looks_sufficient):
            causes.append("Context and grounding look good, so failures are more likely in answer synthesis/prompt instructions than retrieval.")
    if question_type in {"summary", "global"} and mean_context is not None and mean_context >= 0.7 and dominant_context_mode not in {"kg_organized", "community_ordered"}:
        causes.append("This looks like a global/summarization failure under a non-structured context order; better evidence organization may help the generator use the retrieved facts.")
    if mean_hallucinated is not None and mean_hallucinated > 0.25:
        causes.append("Hallucinated-claim rate is high; tighten answer grounding, citations, and abstention rather than expanding retrieval blindly.")
    if high_hallucination_rate is not None and high_hallucination_rate > 0.4 and retry_enabled_share == 0.0 and critique_enabled_share == 0.0:
        causes.append("There is no critique/retry safety layer on a group with frequent unsupported answers; Self-RAG retry or critique is a plausible missing guardrail.")
    if component in {"kg_expansion", "kg_noise"} or (high_kg_noise_rate is not None and high_kg_noise_rate > 0.25):
        causes.append("KG evidence is noisy or unhelpful for this group; compare conservative/direct profiles, graph weight, and quality threshold.")
    if kg_enabled_share >= 0.5 and component == "kg_expansion" and mean_recall is not None and mean_recall < 0.5:
        causes.append("KG is enabled but does not recover the missing evidence; this points to a weak graph, poor edge coverage, or the wrong traversal profile rather than a general answer-model issue.")
    if component in {"candidate_generation", "context_selection"} and low_recall_rate is not None and low_recall_rate > 0.6 and not any("Retriever type" in cause or "Chunking" in cause for cause in causes):
        causes.append("Low recall is widespread, but no single better retriever/chunking setting dominates; run a controlled retrieval-depth/chunking ablation.")
    if component == "answer_generation" and (mean_recall is not None and mean_recall < 0.5) and (mean_context is None or mean_context < 0.8):
        causes.append("Although this group was labelled answer generation, retrieval/context coverage is still weak; do not claim prompt/model failure until retrieval is controlled.")
    if component == "calibration":
        causes.append("Confidence is not trustworthy on this group; the system needs calibration or review thresholds, not just accuracy tuning.")
        if mean_ndcg is not None and mean_ndcg < 0.5:
            causes.append("Calibration is not the whole story here: ranking quality is also weak, so confidence should not hide a retrieval/selection problem.")
    if low_context_rate is not None and low_context_rate > 0.5 and component == "context_selection":
        causes.append("The main loss happens before answer synthesis: many cases still miss enough supporting context, so retrieval assembly should be fixed before prompt tuning.")
    if _has_truthy(rows, "self_rag_retry_on_weak_evidence"):
        causes.append("Self-RAG retry was already enabled in some cases; inspect whether retry queries changed retrieval coverage before adding more attempts.")
    if not causes:
        causes.append("No dominant design cause is visible from current comparisons; add controlled one-factor ablations for retriever, chunking, top_k, KG, and answer_mode.")
    deduped: List[str] = []
    seen = set()
    for cause in causes:
        if cause in seen:
            continue
        seen.add(cause)
        deduped.append(cause)
    return deduped[:8]


def _group_explanation(
    component: str,
    rows: Sequence[Dict[str, object]],
    metrics: Dict[str, object],
    insights: Sequence[Dict[str, object]],
    causes: Sequence[str],
) -> str:
    taxonomy = _taxonomy_details(component)
    n_cases = len(rows)
    evidence = _group_evidence_sentence(metrics)
    comparison_clause = f" Repeated observed alternatives: {_format_pair_insights(insights)}." if insights else " No repeated same-question alternative dominates yet."
    cause_clause = " Main likely causes: " + " ".join(f"{index + 1}) {cause}" for index, cause in enumerate(causes[:4]))
    return f"{taxonomy['label']} covers {n_cases} similar cases. Aggregate indicators: {evidence}.{comparison_clause}{cause_clause}"


def build_group_failure_explanations(failure_rows: Sequence[Dict[str, object]], paired_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in failure_rows:
        component = _safe_str(row.get("likely_component") or "unknown")
        reason = _safe_str(row.get("primary_error_reason") or "unknown")
        question_type = _safe_str(row.get("question_type") or "unknown")
        grouped[(component, reason, question_type)].append(row)
    explanations: List[Dict[str, object]] = []
    for (component, reason, question_type), rows in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        taxonomy = _taxonomy_details(component)
        metrics = _group_metric_summary(rows)
        fixes = _top_observed_fixes(rows)
        insights = _paired_group_insights(rows, paired_rows)
        insight_text = _format_pair_insights(insights)
        causes = _group_design_causes(component, rows, metrics, insights)
        explanations.append(
            {
                "likely_component": component,
                "component_label": taxonomy["label"],
                "component_explanation": taxonomy["explanation"],
                "primary_error_reason": reason,
                "question_type": question_type,
                "n_cases": len(rows),
                "example_question_ids": "; ".join(_safe_str(row.get("question_id")) for row in rows[:5]),
                "top_retrievers": _top_values(rows, "retriever"),
                "top_chunking_strategies": _top_values(rows, "chunking_strategy"),
                "top_context_modes": _top_values(rows, "context_mode"),
                "top_observed_fixes": fixes,
                "top_pairwise_improvements": insight_text,
                "likely_design_causes": " | ".join(causes),
                "group_causal_explanation": _group_explanation(component, rows, metrics, insights, causes),
                "advisor_recommendation": taxonomy["recommendation"],
                "advisor_next_experiment": taxonomy["next_experiment"],
                **metrics,
            }
        )
    return explanations


def build_cost_latency_attribution(trace_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    by_experiment: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in trace_rows:
        by_experiment[_safe_str(row.get("experiment") or "single_config")].append(row)
    for experiment, items in sorted(by_experiment.items()):
        llm_calls = [_safe_float(row.get("estimated_llm_call_count")) or 0.0 for row in items]
        cost_units = [_safe_float(row.get("estimated_extra_cost_units")) or 0.0 for row in items]
        runtimes = [
            value for value in (_safe_float(row.get("runtime_seconds")) for row in items)
            if value is not None
        ]
        exemplar = items[0] if items else {}
        payload = {
            "experiment": experiment,
            "n_questions": len(items),
            "total_estimated_llm_call_count": sum(llm_calls),
            "mean_estimated_llm_call_count": sum(llm_calls) / len(llm_calls) if llm_calls else None,
            "total_estimated_extra_cost_units": sum(cost_units),
            "mean_estimated_extra_cost_units": sum(cost_units) / len(cost_units) if cost_units else None,
            "runtime_seconds": runtimes[0] if runtimes else None,
            "runtime_seconds_per_question": (runtimes[0] / len(items)) if runtimes and items else None,
        }
        for factor in DESIGN_FACTORS:
            payload[factor] = exemplar.get(factor)
        rows.append(payload)
    return rows


def build_component_attribution(trace_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    by_component: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in trace_rows:
        component = _likely_component(row) if _is_failure(row) else "success"
        by_component[component].append(row)
    score_columns = [
        "design_quality_score",
        "retrieval_score",
        "generation_score",
        "grounding_score",
        "kg_score",
        "calibration_score",
        "cost_efficiency_score",
    ]
    for component, items in sorted(by_component.items()):
        taxonomy = _taxonomy_details(component)
        payload = {
            "component": component,
            "component_label": taxonomy["label"],
            "component_explanation": taxonomy["explanation"],
            "advisor_component": taxonomy["advisor_component"],
            "advisor_issue": taxonomy["advisor_issue"],
            "advisor_recommendation": taxonomy["recommendation"],
            "advisor_next_experiment": taxonomy["next_experiment"],
            "n_rows": len(items),
            "n_failures": sum(1 for row in items if _is_failure(row)),
            "failure_rate": sum(1 for row in items if _is_failure(row)) / len(items) if items else None,
            "top_failure_reason": _top_reason(row.get("primary_error_reason") for row in items),
            "top_question_type": _top_reason(row.get("question_type") for row in items),
            "top_retriever": _top_reason(row.get("retriever") for row in items),
            "top_context_mode": _top_reason(row.get("context_mode") for row in items),
            "top_kg_profile": _top_reason(row.get("kg_profile") for row in items),
        }
        for score in score_columns:
            payload[f"mean_{score}"] = _mean_present(row.get(score) for row in items)
        rows.append(payload)
    return rows


def build_failure_taxonomy_reference() -> List[Dict[str, object]]:
    return [
        {
            "component": component,
            **details,
        }
        for component, details in sorted(FAILURE_TAXONOMY.items())
    ]


def build_task_type_attribution(trace_rows: Sequence[Dict[str, object]], paired_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    grouped: Dict[tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in trace_rows:
        question_type = _safe_str(row.get("question_type") or "unknown")
        for factor in DESIGN_FACTORS:
            value = _safe_str(row.get(factor))
            if value:
                grouped[(question_type, factor, value)].append(row)
    for (question_type, factor, value), items in sorted(grouped.items()):
        relevant_pairs = [
            row for row in paired_rows
            if row.get("question_type") == question_type
            and row.get("factor") == factor
            and (row.get("left_value") == value or row.get("right_value") == value)
        ]
        controlled_deltas = []
        for row in relevant_pairs:
            if row.get("comparison_type") != "controlled":
                continue
            delta = _safe_float(row.get("score_delta_right_minus_left"))
            if delta is None:
                continue
            controlled_deltas.append(delta if row.get("right_value") == value else -delta)
        stats = _delta_stats(controlled_deltas)
        rows.append(
            {
                "question_type": question_type,
                "factor": factor,
                "value": value,
                "n_rows": len(items),
                "n_questions": len({_safe_str(row.get("question_id")) for row in items}),
                "failure_rate": sum(1 for row in items if _is_failure(row)) / len(items) if items else None,
                "mean_design_quality_score": _mean_present(row.get("design_quality_score") for row in items),
                "mean_retrieval_score": _mean_present(row.get("retrieval_score") for row in items),
                "mean_generation_score": _mean_present(row.get("generation_score") for row in items),
                "mean_grounding_score": _mean_present(row.get("grounding_score") for row in items),
                "mean_kg_score": _mean_present(row.get("kg_score") for row in items),
                "mean_controlled_design_quality_delta": stats["mean"],
                "controlled_delta_ci95_low": stats["ci95_low"],
                "controlled_delta_ci95_high": stats["ci95_high"],
                "controlled_win_rate": stats["win_rate"],
                "n_controlled_comparisons": len(controlled_deltas),
                "top_failure_reason": _top_reason(row.get("primary_error_reason") for row in items),
            }
        )
    return rows


def build_pareto_frontier(trace_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in trace_rows:
        grouped[_safe_str(row.get("experiment") or "single_config")].append(row)
    candidates = []
    for experiment, items in sorted(grouped.items()):
        quality = _mean_present(row.get("design_quality_score") for row in items)
        cost = _mean_present(row.get("estimated_extra_cost_units") for row in items)
        if quality is None or cost is None:
            continue
        exemplar = items[0]
        candidates.append(
            {
                "experiment": experiment,
                "mean_design_quality_score": quality,
                "mean_estimated_extra_cost_units": cost,
                "mean_estimated_llm_call_count": _mean_present(row.get("estimated_llm_call_count") for row in items),
                "n_questions": len(items),
                "retriever": exemplar.get("retriever"),
                "chunking_strategy": exemplar.get("chunking_strategy"),
                "top_k": exemplar.get("top_k"),
                "query_augmentation": exemplar.get("query_augmentation"),
                "context_mode": exemplar.get("context_mode"),
                "kg_enabled": exemplar.get("kg_enabled"),
                "kg_profile": exemplar.get("kg_profile"),
                "self_rag_retry_on_weak_evidence": exemplar.get("self_rag_retry_on_weak_evidence"),
            }
        )
    for candidate in candidates:
        candidate["is_pareto_optimal"] = not any(
            other["mean_design_quality_score"] >= candidate["mean_design_quality_score"]
            and other["mean_estimated_extra_cost_units"] <= candidate["mean_estimated_extra_cost_units"]
            and (
                other["mean_design_quality_score"] > candidate["mean_design_quality_score"]
                or other["mean_estimated_extra_cost_units"] < candidate["mean_estimated_extra_cost_units"]
            )
            for other in candidates
        )
    return sorted(candidates, key=lambda row: (not row["is_pareto_optimal"], row["mean_estimated_extra_cost_units"], -row["mean_design_quality_score"]))


def build_parsing_diagnostics(run_summary: Dict[str, object]) -> List[Dict[str, object]]:
    outputs = run_summary.get("outputs", {}) if isinstance(run_summary.get("outputs"), dict) else {}
    sections = _read_csv(outputs.get("sections_csv"))
    paragraphs = _read_csv(outputs.get("paragraphs_csv"))
    rows: List[Dict[str, object]] = []
    if sections is not None and not sections.empty:
        doc_col = "doc_id" if "doc_id" in sections.columns else sections.columns[0]
        text_col = "text" if "text" in sections.columns else None
        for doc_id, group in sections.groupby(doc_col):
            text_lengths = [len(_safe_str(value)) for value in group[text_col]] if text_col else []
            rows.append(
                {
                    "doc_id": doc_id,
                    "n_sections": len(group),
                    "n_empty_sections": sum(1 for length in text_lengths if length == 0),
                    "empty_section_rate": (sum(1 for length in text_lengths if length == 0) / len(text_lengths)) if text_lengths else None,
                    "mean_section_chars": sum(text_lengths) / len(text_lengths) if text_lengths else None,
                    "min_section_chars": min(text_lengths) if text_lengths else None,
                    "max_section_chars": max(text_lengths) if text_lengths else None,
                    "n_paragraphs": None,
                }
            )
    if paragraphs is not None and not paragraphs.empty and rows:
        doc_col = "doc_id" if "doc_id" in paragraphs.columns else paragraphs.columns[0]
        paragraph_counts = paragraphs.groupby(doc_col).size().to_dict()
        for row in rows:
            row["n_paragraphs"] = paragraph_counts.get(row["doc_id"], 0)
    return rows


def build_design_report(
    *,
    trace_rows: Sequence[Dict[str, object]],
    factor_rows: Sequence[Dict[str, object]],
    failure_rows: Sequence[Dict[str, object]],
    group_failure_rows: Sequence[Dict[str, object]],
    component_rows: Sequence[Dict[str, object]],
    task_type_rows: Sequence[Dict[str, object]],
    pareto_rows: Sequence[Dict[str, object]],
    cost_latency_rows: Sequence[Dict[str, object]],
    parsing_rows: Sequence[Dict[str, object]],
) -> str:
    lines = ["# Design Attribution Report", ""]
    lines.append("This report links RAG quality failures to design choices such as chunking, retrieval, top-k, KG expansion, prompting, and calibration.")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Trace rows: {len(trace_rows)}")
    lines.append(f"- Failure attribution rows: {len(failure_rows)}")
    lines.append(f"- Group failure explanation rows: {len(group_failure_rows)}")
    lines.append(f"- Component attribution rows: {len(component_rows)}")
    lines.append(f"- Task-type attribution rows: {len(task_type_rows)}")
    lines.append(f"- Pareto frontier rows: {len(pareto_rows)}")
    lines.append(f"- Cost/latency rows: {len(cost_latency_rows)}")
    lines.append(f"- Design factors compared: {len({row.get('factor') for row in factor_rows})}")
    lines.append("")
    if factor_rows:
        lines.append("## Strongest Paired Factor Effects")
        ranked = sorted(
            [row for row in factor_rows if _safe_float(row.get("mean_paired_design_quality_delta")) is not None],
            key=lambda row: abs(float(row.get("mean_paired_design_quality_delta") or 0.0)),
            reverse=True,
        )[:10]
        for row in ranked:
            lines.append(
                f"- {row.get('factor')}={row.get('value')}: paired delta={row.get('mean_paired_design_quality_delta')}, "
                f"failures={row.get('failure_count')}, top_failure={row.get('top_failure_reason')}"
            )
        lines.append("")
    if component_rows:
        lines.append("## Component Attribution")
        for row in sorted(component_rows, key=lambda item: int(item.get("n_failures") or 0), reverse=True)[:10]:
            lines.append(
                f"- {row.get('component')} ({row.get('component_label')}): failures={row.get('n_failures')}, "
                f"failure_rate={row.get('failure_rate')}, top_reason={row.get('top_failure_reason')}"
            )
            lines.append(f"  - Meaning: {row.get('component_explanation')}")
            lines.append(f"  - Recommendation: {row.get('advisor_recommendation')}")
        lines.append("")
    if group_failure_rows:
        lines.append("## Group-Level Causal Failure Explanations")
        for row in group_failure_rows[:10]:
            lines.append(
                f"- {row.get('component_label')} / {row.get('primary_error_reason')} / {row.get('question_type')}: "
                f"{row.get('group_causal_explanation')}"
            )
            if row.get("top_observed_fixes"):
                lines.append(f"  - Observed fixes: {row.get('top_observed_fixes')}")
        lines.append("")
    if task_type_rows:
        lines.append("## Task-Type Attribution")
        ranked_task_rows = sorted(
            [row for row in task_type_rows if _safe_float(row.get("mean_controlled_design_quality_delta")) is not None],
            key=lambda row: abs(float(row.get("mean_controlled_design_quality_delta") or 0.0)),
            reverse=True,
        )[:10]
        for row in ranked_task_rows:
            lines.append(
                f"- {row.get('question_type')} / {row.get('factor')}={row.get('value')}: "
                f"controlled_delta={row.get('mean_controlled_design_quality_delta')}, "
                f"n={row.get('n_controlled_comparisons')}"
            )
        lines.append("")
    if pareto_rows:
        lines.append("## Quality-Cost Pareto Frontier")
        for row in [item for item in pareto_rows if item.get("is_pareto_optimal")][:10]:
            lines.append(
                f"- {row.get('experiment')}: quality={row.get('mean_design_quality_score')}, "
                f"cost={row.get('mean_estimated_extra_cost_units')}"
            )
        lines.append("")
    if failure_rows:
        lines.append("## Most Common Suspected Components")
        components = Counter(row.get("likely_component") for row in failure_rows)
        for component, count in components.most_common(10):
            lines.append(f"- {component}: {count}")
        lines.append("")
        lines.append("## Example Failure Attributions")
        for row in failure_rows[:10]:
            lines.append(f"- {row.get('question_id')} / {row.get('experiment')}: {row.get('causal_explanation')}")
            lines.append(f"  - Suspected factors: {row.get('design_factor_suspects')}")
            lines.append(f"  - Next ablation: {row.get('recommended_next_ablation')}")
        lines.append("")
    if cost_latency_rows:
        lines.append("## Cost And Latency Attribution")
        for row in cost_latency_rows[:10]:
            lines.append(
                f"- {row.get('experiment')}: estimated_llm_calls={row.get('total_estimated_llm_call_count')}, "
                f"estimated_cost_units={row.get('total_estimated_extra_cost_units')}, "
                f"runtime_seconds={row.get('runtime_seconds')}"
            )
        lines.append("")
    if parsing_rows:
        lines.append("## Parsing Diagnostics")
        high_empty = [
            row for row in parsing_rows
            if _safe_float(row.get("empty_section_rate")) is not None and float(row.get("empty_section_rate") or 0.0) > 0.2
        ]
        lines.append(f"- Documents inspected: {len(parsing_rows)}")
        lines.append(f"- Documents with >20% empty sections: {len(high_empty)}")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("Use `failure_attribution.csv` for row-level error causes and `design_factor_effects.csv` for aggregate design-choice effects. Strong claims should use `comparison_type=controlled`; partially controlled and confounded comparisons are exploratory.")
    return "\n".join(lines) + "\n"


def _select_target_experiment_name(run_summary: Dict[str, object]) -> str:
    best = run_summary.get("best_experiment")
    if isinstance(best, dict) and _safe_str(best.get("experiment")):
        return _safe_str(best.get("experiment"))
    classifier_summary = run_summary.get("classifier_summary")
    if isinstance(classifier_summary, dict):
        classifier = classifier_summary.get("classifier", {}) if isinstance(classifier_summary.get("classifier"), dict) else {}
        return _safe_str(classifier_summary.get("experiment") or classifier.get("type"))
    experiments = _iter_experiment_summaries(run_summary)
    if len(experiments) == 1:
        return _safe_str(experiments[0].get("experiment"))
    return ""


def _root_cause_items(
    group_failure_rows: Sequence[Dict[str, object]],
    target_experiment: str,
    target_trace_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows = list(group_failure_rows)
    items: List[Dict[str, object]] = []
    seen_components = set()
    for row in rows:
        component = _safe_str(row.get("likely_component")) or "general"
        if component in seen_components:
            continue
        seen_components.add(component)
        evidence_parts = []
        for label, key in [
            ("mean MRR@K", "mean_mrr_at_k"),
            ("mean Recall@K", "mean_recall_at_k"),
            ("mean NDCG@K", "mean_ndcg_at_k"),
            ("mean context-claim recall", "mean_context_claim_recall"),
            ("mean grounded-claim ratio", "mean_grounded_claim_ratio"),
            ("mean hallucinated-claim ratio", "mean_hallucinated_claim_ratio"),
        ]:
            value = _safe_float(row.get(key))
            if value is not None:
                evidence_parts.append(f"{label}={_fmt_metric(value)}")
        if _safe_str(row.get("top_pairwise_improvements")):
            evidence_parts.append(f"comparisons: {_safe_str(row.get('top_pairwise_improvements'))}")
        likely_causes = [
            part.strip()
            for part in _safe_str(row.get("likely_design_causes")).split("|")
            if part.strip()
        ]
        recommendation = _safe_str(row.get("advisor_recommendation"))
        if likely_causes:
            recommendation = f"{likely_causes[0]} {recommendation}".strip()
        items.append(
            {
                "priority": f"P{min(len(items) + 1, 3)}",
                "component": component,
                "issue": f"{_safe_str(row.get('component_label'))} · {_safe_str(row.get('primary_error_reason'))}",
                "n_cases": int(row.get("n_cases") or 0),
                "recommendation": recommendation,
                "evidence": "; ".join(evidence_parts),
                "next_experiment": _safe_str(row.get("advisor_next_experiment")),
                "likely_causes": " | ".join(likely_causes[:4]),
                "experiment": target_experiment,
            }
        )
        if len(items) >= 4:
            break
    failed_rows = [row for row in target_trace_rows if _is_failure(row)]
    high_conf_wrong_rate = (
        sum(1 for row in failed_rows if (_safe_float(row.get("prediction_confidence")) or 0.0) > 0.8) / len(failed_rows)
        if failed_rows else 0.0
    )
    mean_failed_confidence = _mean_present(row.get("prediction_confidence") for row in failed_rows)
    if "calibration" not in seen_components and failed_rows and high_conf_wrong_rate > 0.5:
        items.append(
            {
                "priority": f"P{min(len(items) + 1, 3)}",
                "component": "calibration",
                "issue": "Confidence calibration · overconfident failures",
                "n_cases": len(failed_rows),
                "recommendation": "A large share of failed rows are still high-confidence, so confidence is not safe for automatic acceptance. Add calibration or review thresholds before trusting the score output.",
                "evidence": f"high-confidence failed rows={_fmt_metric(high_conf_wrong_rate)}" + (f"; mean failed-row confidence={_fmt_metric(mean_failed_confidence)}" if mean_failed_confidence is not None else ""),
                "next_experiment": "Create reliability bins and test auto-accept/manual-review thresholds using confidence, margin, and entropy.",
                "likely_causes": "Confidence is being interpreted as correctness even when ranking quality is weak. | Post-hoc calibration or review thresholds are missing.",
                "experiment": target_experiment,
            }
        )
    return items[:4]


def write_design_attribution_artifacts(run_summary: Dict[str, object], run_dir: str) -> Dict[str, object]:
    import pandas as pd

    start = time.perf_counter()
    outputs = dict(run_summary.get("outputs", {}) if isinstance(run_summary.get("outputs"), dict) else {})
    trace_rows = build_design_trace(run_summary)
    paired_rows = build_paired_comparisons(trace_rows)
    factor_rows = build_factor_effects(trace_rows, paired_rows)
    failure_rows = build_failure_attribution(trace_rows, paired_rows)
    target_experiment = _select_target_experiment_name(run_summary)
    target_trace_rows = [
        row for row in trace_rows
        if not target_experiment or _safe_str(row.get("experiment")) == target_experiment
    ]
    target_failure_rows = [
        row for row in failure_rows
        if not target_experiment or _safe_str(row.get("experiment")) == target_experiment
    ]
    group_failure_rows = build_group_failure_explanations(target_failure_rows, paired_rows)
    component_rows = build_component_attribution(trace_rows)
    taxonomy_rows = build_failure_taxonomy_reference()
    task_type_rows = build_task_type_attribution(trace_rows, paired_rows)
    pareto_rows = build_pareto_frontier(trace_rows)
    cost_latency_rows = build_cost_latency_attribution(trace_rows)
    parsing_rows = build_parsing_diagnostics(run_summary)

    paths = {
        "design_trace_csv": os.path.join(run_dir, "design_trace.csv"),
        "paired_design_comparisons_csv": os.path.join(run_dir, "paired_design_comparisons.csv"),
        "design_factor_effects_csv": os.path.join(run_dir, "design_factor_effects.csv"),
        "failure_attribution_csv": os.path.join(run_dir, "failure_attribution.csv"),
        "group_failure_explanations_csv": os.path.join(run_dir, "group_failure_explanations.csv"),
        "component_attribution_csv": os.path.join(run_dir, "component_attribution.csv"),
        "failure_taxonomy_csv": os.path.join(run_dir, "failure_taxonomy.csv"),
        "task_type_attribution_csv": os.path.join(run_dir, "task_type_attribution.csv"),
        "design_pareto_frontier_csv": os.path.join(run_dir, "design_pareto_frontier.csv"),
        "cost_latency_attribution_csv": os.path.join(run_dir, "cost_latency_attribution.csv"),
        "parsing_diagnostics_csv": os.path.join(run_dir, "parsing_diagnostics.csv"),
        "design_attribution_report_md": os.path.join(run_dir, "design_attribution_report.md"),
        "design_attribution_summary_json": os.path.join(run_dir, "design_attribution_summary.json"),
    }
    pd.DataFrame(trace_rows).to_csv(paths["design_trace_csv"], index=False)
    pd.DataFrame(paired_rows).to_csv(paths["paired_design_comparisons_csv"], index=False)
    pd.DataFrame(factor_rows).to_csv(paths["design_factor_effects_csv"], index=False)
    pd.DataFrame(failure_rows).to_csv(paths["failure_attribution_csv"], index=False)
    pd.DataFrame(group_failure_rows).to_csv(paths["group_failure_explanations_csv"], index=False)
    pd.DataFrame(component_rows).to_csv(paths["component_attribution_csv"], index=False)
    pd.DataFrame(taxonomy_rows).to_csv(paths["failure_taxonomy_csv"], index=False)
    pd.DataFrame(task_type_rows).to_csv(paths["task_type_attribution_csv"], index=False)
    pd.DataFrame(pareto_rows).to_csv(paths["design_pareto_frontier_csv"], index=False)
    pd.DataFrame(cost_latency_rows).to_csv(paths["cost_latency_attribution_csv"], index=False)
    pd.DataFrame(parsing_rows).to_csv(paths["parsing_diagnostics_csv"], index=False)
    with open(paths["design_attribution_report_md"], "w", encoding="utf-8") as f:
        f.write(
            build_design_report(
                trace_rows=trace_rows,
                factor_rows=factor_rows,
                failure_rows=failure_rows,
                group_failure_rows=group_failure_rows,
                component_rows=component_rows,
                task_type_rows=task_type_rows,
                pareto_rows=pareto_rows,
                cost_latency_rows=cost_latency_rows,
                parsing_rows=parsing_rows,
            )
        )
    summary_payload = {
        "n_trace_rows": len(trace_rows),
        "n_paired_comparisons": len(paired_rows),
        "n_factor_effect_rows": len(factor_rows),
        "n_failure_attribution_rows": len(failure_rows),
        "n_group_failure_explanation_rows": len(group_failure_rows),
        "n_component_attribution_rows": len(component_rows),
        "n_failure_taxonomy_rows": len(taxonomy_rows),
        "n_task_type_attribution_rows": len(task_type_rows),
        "n_pareto_frontier_rows": len(pareto_rows),
        "n_controlled_comparisons": sum(1 for row in paired_rows if row.get("comparison_type") == "controlled"),
        "n_confounded_comparisons": sum(1 for row in paired_rows if row.get("comparison_type") == "confounded"),
        "n_cost_latency_rows": len(cost_latency_rows),
        "n_parsing_diagnostic_rows": len(parsing_rows),
        "total_estimated_llm_call_count": sum(_safe_float(row.get("total_estimated_llm_call_count")) or 0.0 for row in cost_latency_rows),
        "total_estimated_extra_cost_units": sum(_safe_float(row.get("total_estimated_extra_cost_units")) or 0.0 for row in cost_latency_rows),
        "top_suspected_components": Counter(row.get("likely_component") for row in failure_rows).most_common(10),
        "top_suspected_component_details": [
            {
                "component": component,
                "count": count,
                **_taxonomy_details(component),
            }
            for component, count in Counter(row.get("likely_component") for row in failure_rows).most_common(10)
        ],
        "target_experiment": target_experiment,
        "root_cause_recommendations": _root_cause_items(group_failure_rows, target_experiment, target_trace_rows),
        "runtime_seconds": time.perf_counter() - start,
        "outputs": paths,
    }
    with open(paths["design_attribution_summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    outputs.update(paths)
    updated = dict(run_summary)
    updated["outputs"] = outputs
    updated["design_attribution"] = {
        key: value for key, value in summary_payload.items() if key != "outputs"
    }
    if isinstance(updated.get("classifier_summary"), dict):
        classifier_summary = dict(updated["classifier_summary"])
        classifier_outputs = dict(classifier_summary.get("outputs", {}))
        classifier_outputs.update(paths)
        classifier_summary["outputs"] = classifier_outputs
        classifier_summary["design_attribution"] = updated["design_attribution"]
        updated["classifier_summary"] = classifier_summary
    return updated
