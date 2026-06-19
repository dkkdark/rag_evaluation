from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().casefold()
    return normalized in {"true", "1", "yes", "y"}


def _as_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_code(value: object) -> str:
    text = "".join(char for char in str(value or "") if char.isdigit())
    return text[:8] if len(text) >= 8 else text


def _top_items(counter: Counter, *, limit: int = 10, key_names: Sequence[str] = ("key", "count")) -> List[Dict[str, object]]:
    first_key = key_names[0]
    second_key = key_names[1] if len(key_names) > 1 else "count"
    rows: List[Dict[str, object]] = []
    for item, count in counter.most_common(limit):
        if isinstance(item, tuple):
            payload = {name: value for name, value in zip(key_names, item)}
            payload["count"] = count
        else:
            payload = {first_key: item, second_key: count}
        rows.append(payload)
    return rows


def build_cpv_evidence_graph_summary(
    *,
    result_rows: Sequence[Dict[str, object]],
    ranking_rows: Sequence[Dict[str, object]],
    parent_lookup: Dict[str, str],
) -> Dict[str, object]:
    total = len(result_rows)
    if total == 0:
        return {
            "enabled": True,
            "n_questions": 0,
            "n_errors": 0,
            "component_signals": {},
        }

    ranking_by_qid: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in ranking_rows:
        qid = str(row.get("question_id") or "").strip()
        if qid:
            ranking_by_qid[qid].append(row)
    for rows in ranking_by_qid.values():
        rows.sort(key=lambda row: int(float(row.get("rank") or 0)))

    incorrect_rows = [row for row in result_rows if str(row.get("auto_flag") or "") != "correct"]
    n_errors = len(incorrect_rows)

    parent_child_errors = 0
    sibling_errors = 0
    same_division_errors = 0
    low_margin_same_branch_errors = 0
    short_query_gold_missing = 0
    low_diversity_gold_missing = 0
    duplicate_pressure_errors = 0
    high_conf_wrong_errors = 0
    notice_example_supported_errors = 0
    one_branch_top3_errors = 0

    confusion_pairs: Counter = Counter()
    wrong_top1_codes: Counter = Counter()
    wrong_gold_codes: Counter = Counter()
    branch_prefix_pairs: Counter = Counter()

    for row in incorrect_rows:
        qid = str(row.get("question_id") or "").strip()
        predicted = _normalize_code(row.get("top1_predicted_cpv"))
        gold = _normalize_code(row.get("expected_cpv_codes"))
        if predicted:
            wrong_top1_codes[predicted] += 1
        if gold:
            wrong_gold_codes[gold] += 1
        if predicted and gold:
            confusion_pairs[(predicted, gold)] += 1
            branch_prefix_pairs[(predicted[:2], gold[:2])] += 1
            pred_parent = parent_lookup.get(predicted, "")
            gold_parent = parent_lookup.get(gold, "")
            if pred_parent == gold and gold:
                parent_child_errors += 1
            elif gold_parent == predicted and predicted:
                parent_child_errors += 1
            elif pred_parent and pred_parent == gold_parent and predicted != gold:
                sibling_errors += 1
            if predicted[:2] and predicted[:2] == gold[:2]:
                same_division_errors += 1

        gold_present = _as_bool(row.get("gold_present_at_k"))
        low_margin = _as_bool(row.get("low_margin_decision"))
        short_query = _as_bool(row.get("short_or_ambiguous_query"))
        low_diversity = _as_bool(row.get("low_diversity_at_k"))
        duplicate_pressure = _as_bool(row.get("duplicate_candidate_pressure"))
        high_conf_wrong = _as_bool(row.get("high_confidence_wrong"))

        if duplicate_pressure:
            duplicate_pressure_errors += 1
        if high_conf_wrong:
            high_conf_wrong_errors += 1
        if short_query and not gold_present:
            short_query_gold_missing += 1
        if low_diversity and not gold_present:
            low_diversity_gold_missing += 1

        ranked = ranking_by_qid.get(qid, [])
        top3 = ranked[:3]
        codes = [_normalize_code(item.get("chunk_id") or item.get("cpv_code") or item.get("title")) for item in top3]
        codes = [code for code in codes if code]
        if len(codes) >= 2:
            parents = [parent_lookup.get(code, "") for code in codes]
            branch_prefixes = {code[:5] for code in codes if len(code) >= 5}
            if len(branch_prefixes) <= 1:
                one_branch_top3_errors += 1
            same_parent_cluster = len({parent for parent in parents if parent}) == 1 and any(parent for parent in parents)
            if low_margin and (same_parent_cluster or len(branch_prefixes) <= 1):
                low_margin_same_branch_errors += 1
        if any(float(item.get("notice_examples_channel_score") or 0.0) > 0.0 for item in top3):
            notice_example_supported_errors += 1

    top1_wrong_rate = n_errors / total if total else 0.0
    graph_summary = {
        "enabled": True,
        "n_questions": total,
        "n_errors": n_errors,
        "top1_wrong_rate": top1_wrong_rate,
        "error_patterns": {
            "parent_child_errors": parent_child_errors,
            "sibling_errors": sibling_errors,
            "same_division_errors": same_division_errors,
            "low_margin_same_branch_errors": low_margin_same_branch_errors,
            "one_branch_top3_errors": one_branch_top3_errors,
            "duplicate_pressure_errors": duplicate_pressure_errors,
            "high_confidence_wrong_errors": high_conf_wrong_errors,
            "short_query_gold_missing": short_query_gold_missing,
            "low_diversity_gold_missing": low_diversity_gold_missing,
            "notice_example_supported_errors": notice_example_supported_errors,
        },
        "rates": {
            "parent_child_error_rate": (parent_child_errors / n_errors) if n_errors else 0.0,
            "sibling_error_rate": (sibling_errors / n_errors) if n_errors else 0.0,
            "same_division_error_rate": (same_division_errors / n_errors) if n_errors else 0.0,
            "low_margin_same_branch_error_rate": (low_margin_same_branch_errors / n_errors) if n_errors else 0.0,
            "one_branch_top3_error_rate": (one_branch_top3_errors / n_errors) if n_errors else 0.0,
            "duplicate_pressure_error_rate": (duplicate_pressure_errors / n_errors) if n_errors else 0.0,
            "high_confidence_wrong_error_rate": (high_conf_wrong_errors / n_errors) if n_errors else 0.0,
            "short_query_gold_missing_rate": (short_query_gold_missing / n_errors) if n_errors else 0.0,
            "low_diversity_gold_missing_rate": (low_diversity_gold_missing / n_errors) if n_errors else 0.0,
            "notice_example_supported_error_rate": (notice_example_supported_errors / n_errors) if n_errors else 0.0,
        },
        "top_confusion_pairs": _top_items(confusion_pairs, limit=10, key_names=("predicted_code", "gold_code")),
        "top_wrong_predicted_codes": _top_items(wrong_top1_codes, limit=10, key_names=("cpv_code", "n_errors")),
        "top_missed_gold_codes": _top_items(wrong_gold_codes, limit=10, key_names=("cpv_code", "n_errors")),
        "top_cross_division_confusions": _top_items(branch_prefix_pairs, limit=10, key_names=("predicted_division", "gold_division")),
    }

    graph_summary["component_signals"] = {
        "retriever": float(graph_summary["rates"]["low_diversity_gold_missing_rate"]),
        "query_enrichment": float(graph_summary["rates"]["short_query_gold_missing_rate"]),
        "hierarchy": max(
            float(graph_summary["rates"]["parent_child_error_rate"]),
            float(graph_summary["rates"]["sibling_error_rate"]),
        ),
        "selection": float(graph_summary["rates"]["low_margin_same_branch_error_rate"]),
        "examples": float(graph_summary["rates"]["notice_example_supported_error_rate"]),
        "deduplication": float(graph_summary["rates"]["duplicate_pressure_error_rate"]),
        "calibration": float(graph_summary["rates"]["high_confidence_wrong_error_rate"]),
    }
    return graph_summary


def build_document_evidence_graph_summary(
    *,
    result_rows: Sequence[Dict[str, object]],
    ranking_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    total = len(result_rows)
    if total == 0:
        return {
            "enabled": True,
            "n_questions": 0,
            "n_errors": 0,
            "component_signals": {},
        }

    ranking_by_qid: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in ranking_rows:
        qid = str(row.get("question_id") or "").strip()
        if qid:
            ranking_by_qid[qid].append(row)
    for rows in ranking_by_qid.values():
        rows.sort(key=lambda row: int(float(row.get("rank") or 0)))

    incorrect_rows = [row for row in result_rows if str(row.get("auto_flag") or "") != "correct"]
    n_errors = len(incorrect_rows)

    missing_relation_evidence_errors = 0
    graph_noise_errors = 0
    graph_gain_but_wrong_errors = 0
    graph_supported_false_refusals = 0
    graph_supported_incomplete_errors = 0
    graph_supported_contradiction_errors = 0
    unsupported_without_graph_path_errors = 0
    cross_doc_pressure_errors = 0
    graph_only_addition_errors = 0

    primary_reasons: Counter = Counter()
    claim_diagnostics: Counter = Counter()
    question_types: Counter = Counter()
    programs: Counter = Counter()

    for row in incorrect_rows:
        qid = str(row.get("question_id") or "").strip()
        primary_reasons[str(row.get("primary_error_reason") or "")] += 1
        claim_diagnostics[str(row.get("claim_diagnostic") or "")] += 1
        question_types[str(row.get("question_type") or "unknown")] += 1
        programs[str(row.get("program_name") or "")] += 1

        relation_recall = _as_float(row.get("gold_kg_relation_evidence_recall"))
        graph_noise = _as_float(row.get("kg_graph_noise_at_k"))
        graph_gain = _as_float(row.get("kg_graph_gain_at_k"))
        path_availability = _as_float(row.get("kg_path_availability"))
        path_correctness = _as_float(row.get("kg_path_correctness"))
        context_claim_recall = _as_float(row.get("context_claim_recall"))
        unsupported_missing_path = _as_float(row.get("unsupported_claim_missing_kg_path_rate"))

        graph_support_strong = (
            (relation_recall is not None and relation_recall >= 0.8)
            or (path_availability is not None and path_availability >= 0.5)
            or (path_correctness is not None and path_correctness >= 0.5)
            or (context_claim_recall is not None and context_claim_recall >= 0.8)
        )

        if relation_recall is not None and relation_recall < 0.8:
            missing_relation_evidence_errors += 1
        if graph_noise is not None and graph_noise > 0.25:
            graph_noise_errors += 1
        if graph_gain is not None and graph_gain > 0.0:
            graph_gain_but_wrong_errors += 1
        if _as_bool(row.get("false_refusal")) and graph_support_strong:
            graph_supported_false_refusals += 1

        claim_diag = str(row.get("claim_diagnostic") or "")
        if claim_diag in {"answer_incomplete", "answer_incomplete_from_available_context"} and graph_support_strong:
            graph_supported_incomplete_errors += 1
        if claim_diag in {"claim_contradiction", "unsupported_generated_claims"} and graph_support_strong:
            graph_supported_contradiction_errors += 1
        if unsupported_missing_path is not None and unsupported_missing_path >= 0.5:
            unsupported_without_graph_path_errors += 1

        top_rows = ranking_by_qid.get(qid, [])
        top_doc_paths = {
            str(item.get("doc_path") or item.get("doc_id") or "").strip()
            for item in top_rows[:10]
            if str(item.get("doc_path") or item.get("doc_id") or "").strip()
        }
        graph_only_rows = [
            item
            for item in top_rows[:10]
            if str(item.get("retrieval_source") or "").strip() == "graph"
        ]
        if len(top_doc_paths) >= 3:
            cross_doc_pressure_errors += 1
        if graph_only_rows:
            graph_only_addition_errors += 1

    graph_summary = {
        "enabled": True,
        "n_questions": total,
        "n_errors": n_errors,
        "top1_wrong_rate": (n_errors / total) if total else 0.0,
        "error_patterns": {
            "missing_relation_evidence_errors": missing_relation_evidence_errors,
            "graph_noise_errors": graph_noise_errors,
            "graph_gain_but_wrong_errors": graph_gain_but_wrong_errors,
            "graph_supported_false_refusals": graph_supported_false_refusals,
            "graph_supported_incomplete_errors": graph_supported_incomplete_errors,
            "graph_supported_contradiction_errors": graph_supported_contradiction_errors,
            "unsupported_without_graph_path_errors": unsupported_without_graph_path_errors,
            "cross_doc_pressure_errors": cross_doc_pressure_errors,
            "graph_only_addition_errors": graph_only_addition_errors,
        },
        "rates": {
            "missing_relation_evidence_error_rate": (missing_relation_evidence_errors / n_errors) if n_errors else 0.0,
            "graph_noise_error_rate": (graph_noise_errors / n_errors) if n_errors else 0.0,
            "graph_gain_but_wrong_error_rate": (graph_gain_but_wrong_errors / n_errors) if n_errors else 0.0,
            "graph_supported_false_refusal_rate": (graph_supported_false_refusals / n_errors) if n_errors else 0.0,
            "graph_supported_incomplete_error_rate": (graph_supported_incomplete_errors / n_errors) if n_errors else 0.0,
            "graph_supported_contradiction_error_rate": (graph_supported_contradiction_errors / n_errors) if n_errors else 0.0,
            "unsupported_without_graph_path_error_rate": (unsupported_without_graph_path_errors / n_errors) if n_errors else 0.0,
            "cross_doc_pressure_error_rate": (cross_doc_pressure_errors / n_errors) if n_errors else 0.0,
            "graph_only_addition_error_rate": (graph_only_addition_errors / n_errors) if n_errors else 0.0,
        },
        "top_primary_error_reasons": _top_items(primary_reasons, limit=10, key_names=("reason", "count")),
        "top_claim_diagnostics": _top_items(claim_diagnostics, limit=10, key_names=("claim_diagnostic", "count")),
        "top_question_types": _top_items(question_types, limit=10, key_names=("question_type", "count")),
        "top_programs": _top_items(programs, limit=10, key_names=("program_name", "count")),
    }
    graph_summary["component_signals"] = {
        "graph_coverage": float(graph_summary["rates"]["missing_relation_evidence_error_rate"]),
        "graph_noise": float(graph_summary["rates"]["graph_noise_error_rate"]),
        "answer_synthesis": max(
            float(graph_summary["rates"]["graph_supported_incomplete_error_rate"]),
            float(graph_summary["rates"]["graph_supported_contradiction_error_rate"]),
        ),
        "refusal_policy": float(graph_summary["rates"]["graph_supported_false_refusal_rate"]),
        "context_selection": float(graph_summary["rates"]["cross_doc_pressure_error_rate"]),
    }
    return graph_summary
