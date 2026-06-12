from __future__ import annotations

from collections import Counter
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List

from rag_eval.core.models import DiagnosticResult, LLMCallResult, LLMConfig
from rag_eval.evaluation.advisor import apply_question_recommendations, build_run_advisor, write_quality_report
from rag_eval.classifiers.cpv_baseline import (
    build_cpv_chunks,
    build_cpv_chunks_from_db,
    build_parent_lookup,
    load_cpv_catalog_from_ted_corpus_export,
    load_cpv_catalog_from_db,
    load_queries,
    sync_cpv_profiles_to_db,
)
from rag_eval.classifiers.cpv_kg import (
    build_cpv_knowledge_graph,
    cpv_kg_metrics,
    graph_expand_and_rerank_cpv,
)
from rag_eval.evaluation.adapters import build_items_from_cpv_queries
from rag_eval.evaluation.core import (
    EvaluationItem,
    PredictionRecord,
    RankedCandidate,
    best_cpv_hierarchy_match,
    cpv_common_prefix_length,
    cpv_structural_distance,
    cpv_structural_similarity,
    evaluate_ranked_predictions,
)
from rag_eval.evaluation.metrics import (
    diagnose_failure,
    evaluate_answer_metrics,
    evaluate_retrieval_metrics,
    is_relevant_grade,
    retrieval_relevance_grade,
    runtime_retrieval_evaluation,
    summarize_diagnostics,
    summarize_retrieval_metrics,
    summarize_confidence_calibration,
)
from rag_eval.evaluation.llm import LLM_QUERY_AUGMENTATION_MODES, augment_query_with_llm
from rag_eval.evaluation.llm import rerank_candidates_with_llm
from rag_eval.retrieval.engines import (
    build_cpv_multi_retriever,
    lexical_overlap_score,
    retrieve_top_k_cpv_multi,
    rerank_with_lexical_signal,
)
from rag_eval.retrieval.cross_encoder import (
    DEFAULT_CROSS_ENCODER_MODEL,
    rerank_with_cross_encoder,
)
from rag_eval.retrieval.local_search import lexical_search as sqlite_lexical_search
from rag_eval.reporting.visualization import write_classifier_showcase_bundle


DUPLICATE_METRIC_COLUMNS = [
    "gold_hit_at_k",
    "gold_first_rank",
    "reciprocal_rank",
    "hierarchical_distance_top1",
    "normalized_hierarchical_distance_top1",
    "cpv_hierarchy_distance_top1",
    "cpv_common_prefix_length_top1",
    "hierarchy_match_level_top1",
    "hierarchy_match_label_top1",
    "hierarchy_score_top1",
    "same_division_top1",
    "same_group_top1",
    "same_class_top1",
    "same_category_top1",
    "same_branch_top1",
    "ancestor_hit_at_k",
    "abstention_correct",
    "runtime_retrieval_score",
    "answer_gold_support",
    "answer_has_gold_substring",
    "gold_claim_count",
    "answer_claim_count",
    "answer_claim_recall",
    "answer_claim_precision",
    "factual_correctness_precision",
    "factual_correctness_recall",
    "hallucinated_claim_ratio",
    "noise_sensitivity_irrelevant",
    "evidence_attribution_precision",
    "evidence_attribution_recall",
    "attributed_answer_claim_count",
    "attributed_gold_claim_count",
    "invalid_attribution_count",
    "unsupported_claim_count",
    "missing_gold_claim_count",
    "contradicted_claim_count",
    "first_relevant_rank",
    "n_relevant_chunks",
    "n_retrieved_relevant_chunks",
    "target_doc_retrieved_at_k",
    "first_target_doc_rank",
    "n_retrieved_target_doc_chunks",
    "partial_correct",
]


def _apply_showcase_bundle(summary: Dict[str, object], showcase_bundle: Dict[str, object]) -> None:
    summary["showcase"] = showcase_bundle
    summary["visualization"]["enabled"] = True
    summary["visualization"]["strategy_score_profile_svg"] = showcase_bundle["score_profile_svg"]
    summary["visualization"]["strategy_chunk_alignment_svg"] = showcase_bundle["chunk_alignment_svg"]
    summary["visualization"]["strategy_unique_chunk_alignment_svg"] = showcase_bundle.get("unique_chunk_alignment_svg")
    summary["outputs"]["strategy_score_profile_svg"] = showcase_bundle["score_profile_svg"]
    summary["outputs"]["strategy_chunk_alignment_svg"] = showcase_bundle["chunk_alignment_svg"]
    summary["outputs"]["strategy_unique_chunk_alignment_svg"] = showcase_bundle.get("unique_chunk_alignment_svg")
    summary["outputs"]["strategy_metric_overview_svg"] = showcase_bundle["metric_overview_svg"]
    summary["outputs"]["strategy_diagnostics_svg"] = showcase_bundle["diagnostics_svg"]
    summary["outputs"]["strategy_showcase_md"] = showcase_bundle["showcase_md"]


def _normalize_prediction_label(candidate: Dict[str, object]) -> str:
    for key in [
        "label",
        "cpv_code",
        "cpv",
        "answer",
        "candidate",
        "prediction",
        "predicted",
        "predicted_answer",
        "predicted_cpv",
        "id",
        "code",
    ]:
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalized_prediction_confidence(score: object) -> float | None:
    try:
        if score is None or score == "":
            return None
        value = float(score)
    except (TypeError, ValueError):
        return None
    if 0.0 <= value <= 1.0:
        return value
    if value > 0.0:
        return value / (1.0 + value)
    return 0.0


def _normalize_prediction_score(candidate: Dict[str, object], *, fallback: float) -> float:
    for key in [
        "score",
        "confidence",
        "probability",
        "retrieval_score",
        "vector_score",
        "rrf_score",
        "reranker_score",
        "final_score",
    ]:
        value = candidate.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return fallback


_QUERY_NOISE_PATTERNS = [
    r"\b(?:procedure|procedura|procedimiento|proc[ée]dure|verhandlungsverfahren|concurso|march[eé]|contrato|framework agreement|dynamic purchasing system|dps)\b",
    r"\b(?:lot(?:s)?|n[. ]?[0-9]+/[0-9]+|nr[. ]?[0-9]+|no[. ]?[0-9]+|ref[. ]?[A-Z0-9/-]+)\b",
    r"\b(?:relance|avviso di consultazione di mercato|consultation de march[eé]|public procurement|service procurement)\b",
]

_QUERY_SPLIT_RE = re.compile(r"\s*(?:[-:;|]|[\u2013\u2014])\s*")
_QUERY_SPACE_RE = re.compile(r"\s+")
_QUERY_NUMERIC_RE = re.compile(r"\b[0-9][0-9./%-]*\b")
_QUERY_NONWORD_RE = re.compile(r"[^\w\s/&+]", flags=re.UNICODE)
_QUERY_SECONDARY_SPLIT_RE = re.compile(
    r"\b(?:with|including|plus|along with|associated with|optional|mit|avec|con|incl\.?|including|samt|und optional|et services associ[ée]s|servizi associati)\b",
    flags=re.IGNORECASE,
)
_PROCUREMENT_ACTION_PATTERNS = [
    "supply and installation",
    "delivery and installation",
    "operation and maintenance",
    "repair and maintenance",
    "maintenance service",
    "installation",
    "delivery",
    "supply",
    "operation",
    "maintenance",
    "repair",
    "consultancy",
    "service",
    "lieferung",
    "montage",
    "wartung",
    "suministro",
    "instalacion",
    "mantenimiento",
    "fourniture",
    "maintenance",
    "entretien",
]

_CONTRACT_TYPE_PATTERNS = {
    "maintenance_repair": [
        "maintenance", "maintain", "repair", "servicing", "service and maintenance",
        "wartung", "instandhaltung", "reparatur", "mantenimiento", "reparacion",
        "manutenzione", "reparation", "entretien", "exploitation des installations",
        "exploitation", "operation", "operating", "technical assistance",
        "assistencia tecnica", "assistencia", "assistenza tecnica",
    ],
    "installation_work": [
        "installation work", "installation works", "install and commission", "installation of",
        "construction", "adaptation works", "instalacion", "installazione", "travaux",
        "obra", "obras", "montaz", "montage",
    ],
    "supply": [
        "supply", "supplies", "procurement", "delivery", "purchase", "acquisition",
        "fourniture", "suministro", "lieferung", "fornitura", "aquisição", "προμήθεια",
    ],
    "consultancy": [
        "consultancy", "consulting", "advisory", "study", "evaluation consultancy",
        "beratung", "conseil", "consultoria", "estudio",
    ],
    "software_it_service": [
        "software", "digital", "it service", "development service", "network service",
        "telecommunication service", "support service", "mdm", "internet access",
    ],
    "security_service": [
        "security service", "private security", "guarding", "surveillance service",
    ],
}


def _normalize_free_text(value: str) -> str:
    return _QUERY_SPACE_RE.sub(" ", _QUERY_NONWORD_RE.sub(" ", str(value or "").casefold())).strip()


def _extract_procurement_object(query_text: str) -> Dict[str, object]:
    original = str(query_text or "").strip()
    lowered = original.casefold()
    cleaned = lowered
    removed_patterns: List[str] = []
    for pattern in _QUERY_NOISE_PATTERNS:
        updated = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        if updated != cleaned:
            removed_patterns.append(pattern)
        cleaned = updated
    cleaned = _QUERY_NUMERIC_RE.sub(" ", cleaned)
    full_query = _normalize_free_text(cleaned)
    raw_segments = [segment.strip() for segment in _QUERY_SPLIT_RE.split(cleaned) if segment.strip()]
    left_primary = raw_segments[0] if raw_segments else cleaned
    secondary_parts = [segment for segment in raw_segments[1:] if segment]

    split_match = _QUERY_SECONDARY_SPLIT_RE.search(left_primary)
    if split_match:
        left_part = left_primary[: split_match.start()].strip()
        right_part = left_primary[split_match.end() :].strip()
        left_primary = left_part or left_primary
        if right_part:
            secondary_parts.insert(0, right_part)

    action_terms = [term for term in _PROCUREMENT_ACTION_PATTERNS if term in full_query]
    action_terms = sorted(set(action_terms), key=lambda term: (-len(term), term))
    action_pattern = re.compile(
        r"\b(?:"
        + "|".join(re.escape(term) for term in action_terms[:8] if term)
        + r"|and|und|et|y|e|de|des|del|di|of|for|a|an|the|eine?r?)\b",
        flags=re.IGNORECASE,
    ) if action_terms else None
    left_primary_core = left_primary
    if action_pattern is not None:
        left_primary_core = _QUERY_SPACE_RE.sub(" ", action_pattern.sub(" ", left_primary)).strip()

    object_candidates: List[str] = []
    for connector in [" of ", " for ", " de ", " des ", " del ", " di ", " για ", " για την ", " για το "]:
        if connector in left_primary:
            object_candidates.append(left_primary.split(connector, 1)[1].strip())
    if left_primary_core and left_primary_core != left_primary:
        object_candidates.insert(0, left_primary_core)
    object_candidates.append(left_primary)
    object_candidates.extend(raw_segments)

    contract_like_tokens = {
        "maintenance", "repair", "service", "services", "installation", "operation", "delivery", "supply",
        "lieferung", "montage", "wartung", "suministro", "instalacion", "mantenimiento", "fourniture", "entretien",
    }
    scored_candidates = []
    for candidate in object_candidates:
        normalized = _normalize_free_text(candidate)
        if not normalized:
            continue
        tokens = [token for token in normalized.split() if len(token) > 2]
        if not tokens:
            continue
        contract_hits = sum(1 for token in tokens if token in contract_like_tokens)
        object_hits = sum(1 for token in tokens if token not in (contract_like_tokens | {"optional", "associated"}))
        scored_candidates.append((normalized, object_hits - contract_hits, contract_hits, len(tokens)))
    scored_candidates.sort(key=lambda item: (-item[1], item[2], item[3], len(item[0])))
    preferred_object_query = _normalize_free_text(left_primary_core) if left_primary_core and left_primary_core != left_primary else ""
    object_query = preferred_object_query or (scored_candidates[0][0] if scored_candidates else full_query)
    full_query = _normalize_free_text(cleaned)
    if len(object_query.split()) < 2 and len(object_query) < 10:
        object_query = _normalize_free_text(original)
    if len(full_query.split()) < 3:
        full_query = _normalize_free_text(original)
    procurement_action = _normalize_free_text(" ".join(action_terms[:3]))
    secondary_context = _normalize_free_text(" ".join(secondary_parts[:4]))
    exclude_as_primary = [
        token
        for token in _normalize_free_text(" ".join(secondary_parts + action_terms)).split()
        if token in {"maintenance", "repair", "service", "services", "operation", "optional", "associated"}
    ]
    return {
        "original_query": original,
        "cleaned_query": full_query,
        "object_query": object_query,
        "main_object": object_query,
        "procurement_action": procurement_action,
        "secondary_context": secondary_context,
        "exclude_as_primary": exclude_as_primary,
        "removed_noise_patterns": removed_patterns,
    }


def _infer_contract_types(text: str) -> List[str]:
    normalized = _normalize_free_text(text)
    types: List[str] = []
    for contract_type, patterns in _CONTRACT_TYPE_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            types.append(contract_type)
    return types


def _contract_type_bonus(query_types: List[str], candidate_types: List[str]) -> float:
    if not query_types:
        return 0.0
    query_set = set(query_types)
    candidate_set = set(candidate_types)
    specific_types = {"maintenance_repair", "installation_work", "consultancy", "security_service", "software_it_service"}
    penalties = {
        frozenset({"maintenance_repair", "installation_work"}): -0.14,
        frozenset({"maintenance_repair", "supply"}): -0.10,
        frozenset({"installation_work", "consultancy"}): -0.08,
        frozenset({"supply", "consultancy"}): -0.08,
        frozenset({"security_service", "supply"}): -0.10,
    }
    for query_type in query_set:
        for candidate_type in candidate_set:
            penalty = penalties.get(frozenset({query_type, candidate_type}))
            if penalty is not None:
                return penalty
    specific_overlap = (query_set & specific_types) & (candidate_set & specific_types)
    if specific_overlap:
        return 0.18
    if query_set & candidate_set:
        return 0.06
    if "maintenance_repair" in query_set and "maintenance_repair" not in candidate_set:
        return -0.12
    if "installation_work" in query_set and "installation_work" not in candidate_set:
        return -0.10
    if "consultancy" in query_set and "consultancy" not in candidate_set:
        return -0.08
    return -0.04 if candidate_set else 0.0


def _apply_contract_type_rerank(rows: List[Dict[str, object]], query_text: str) -> List[Dict[str, object]]:
    if not rows:
        return []
    query_types = _infer_contract_types(query_text)
    reranked: List[Dict[str, object]] = []
    for row in rows:
        updated = dict(row)
        candidate_text = " ".join(
            str(updated.get(key) or "")
            for key in ["cpv_label", "title", "text", "description_en", "keywords_en", "cpv_parent_label"]
        )
        candidate_types = _infer_contract_types(candidate_text)
        bonus = _contract_type_bonus(query_types, candidate_types)
        updated["query_contract_types"] = ",".join(query_types)
        updated["candidate_contract_types"] = ",".join(candidate_types)
        updated["contract_type_bonus"] = bonus
        updated["score_before_contract_type"] = float(updated.get("score") or 0.0)
        updated["score"] = float(updated["score_before_contract_type"]) + bonus
        if abs(bonus) > 1e-12:
            updated["reranker"] = "contract_type"
        reranked.append(updated)
    reranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return reranked


def _apply_object_focus_rerank(
    rows: List[Dict[str, object]],
    *,
    main_object: str,
    procurement_action: str,
    secondary_context: str,
    exclude_as_primary: List[str],
) -> List[Dict[str, object]]:
    if not rows:
        return []
    reranked: List[Dict[str, object]] = []
    exclusion_text = " ".join(exclude_as_primary)
    for row in rows:
        updated = dict(row)
        candidate_text = " ".join(
            str(updated.get(key) or "")
            for key in [
                "cpv_label",
                "description_en",
                "use_when_text",
                "do_not_use_when_text",
                "children_labels",
                "sibling_labels",
                "search_text_en",
                "search_text_multilingual",
            ]
        )
        main_overlap = lexical_overlap_score(main_object, candidate_text)
        action_overlap = lexical_overlap_score(procurement_action, candidate_text)
        secondary_overlap = lexical_overlap_score(secondary_context, candidate_text)
        exclude_overlap = lexical_overlap_score(exclusion_text, candidate_text)
        focus_bonus = (0.18 * main_overlap) + (0.06 * action_overlap) + (0.02 * secondary_overlap)
        if exclude_overlap > main_overlap:
            focus_bonus -= 0.10 * exclude_overlap
        updated["main_object_overlap"] = main_overlap
        updated["procurement_action_overlap"] = action_overlap
        updated["secondary_context_overlap"] = secondary_overlap
        updated["exclude_as_primary_overlap"] = exclude_overlap
        updated["score_before_object_focus"] = float(updated.get("score") or 0.0)
        updated["object_focus_bonus"] = focus_bonus
        updated["score"] = float(updated["score_before_object_focus"]) + focus_bonus
        reranked.append(updated)
    reranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return reranked


def _compact_hint_text(value: str, *, max_tokens: int = 6) -> str:
    tokens = [token for token in _normalize_free_text(value).split() if len(token) > 2]
    return " ".join(tokens[:max_tokens])


def _augment_query_with_cpv_db_hints(
    *,
    sqlite_path: str | None,
    main_object: str,
    translated_query: str,
    procurement_action: str,
    max_hints: int = 3,
) -> Dict[str, object]:
    if not sqlite_path or not os.path.exists(sqlite_path):
        return {"status": "no_sqlite_index", "english_hints": [], "multilingual_hints": []}
    query_en = translated_query.strip() or main_object.strip()
    query_local = main_object.strip() or translated_query.strip()
    if not query_en and not query_local:
        return {"status": "empty_query", "english_hints": [], "multilingual_hints": []}

    probe_rows: List[Dict[str, object]] = []
    if query_en:
        probe_rows.extend(
            sqlite_lexical_search(
                sqlite_path=sqlite_path,
                query=query_en,
                k=8,
                fields=["search_text_en"],
            )
        )
    if query_local:
        probe_rows.extend(
            sqlite_lexical_search(
                sqlite_path=sqlite_path,
                query=query_local,
                k=8,
                fields=["search_text_multilingual"],
            )
        )

    english_hints: List[str] = []
    multilingual_hints: List[str] = []
    seen = set()
    for row in probe_rows:
        code = str(row.get("cpv_code") or row.get("chunk_id") or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        label = _compact_hint_text(str(row.get("cpv_label") or ""), max_tokens=5)
        keywords = _compact_hint_text(str(row.get("keywords_en") or row.get("generated_keywords_en") or ""), max_tokens=5)
        procurement_type = _compact_hint_text(str(row.get("procurement_type") or ""), max_tokens=3)
        object_overlap = lexical_overlap_score(main_object, " ".join([label, keywords]))
        if object_overlap < 0.15 and label:
            continue
        if label:
            english_hints.append(label)
        if keywords and keywords != label:
            english_hints.append(keywords)
        if procurement_type and procurement_action and lexical_overlap_score(procurement_action, procurement_type) > 0.2:
            english_hints.append(procurement_type)
        aliases = _compact_hint_text(str(row.get("description_multilingual_aliases") or ""), max_tokens=5)
        if aliases:
            multilingual_hints.append(aliases)
        if len(english_hints) >= max_hints * 2 and len(multilingual_hints) >= max_hints:
            break

    dedup_en = []
    seen_en = set()
    for hint in english_hints:
        key = hint.casefold()
        if hint and key not in seen_en:
            seen_en.add(key)
            dedup_en.append(hint)
    dedup_local = []
    seen_local = set()
    for hint in multilingual_hints:
        key = hint.casefold()
        if hint and key not in seen_local:
            seen_local.add(key)
            dedup_local.append(hint)
    return {
        "status": "ok" if (dedup_en or dedup_local) else "no_hints",
        "english_hints": dedup_en[: max_hints * 2],
        "multilingual_hints": dedup_local[:max_hints],
    }


def _should_apply_llm_rerank(rows: List[Dict[str, object]], query_text: str, *, top_k: int) -> tuple[bool, str]:
    if len(rows) < 2:
        return False, "not_enough_candidates"
    head = rows[: min(max(top_k, 3), len(rows))]
    scores = [float(row.get("score") or 0.0) for row in head[:3]]
    margin = scores[0] - scores[1] if len(scores) >= 2 else 1.0
    top_branch_prefixes = {str(row.get("cpv_code") or "")[:5] for row in head[:3] if str(row.get("cpv_code") or "").strip()}
    query_types = set(_infer_contract_types(query_text))
    top_candidate_types = [set(str(row.get("candidate_contract_types") or "").split(",")) - {""} for row in head[:3]]
    specific_types = {"maintenance_repair", "installation_work", "consultancy", "security_service", "software_it_service"}
    query_specific = query_types & specific_types
    if margin <= 0.03:
        return True, "close_score_margin"
    if len(top_branch_prefixes) <= 2:
        return True, "same_branch_cluster"
    if query_specific and any(query_specific.isdisjoint(candidate_types & specific_types) for candidate_types in top_candidate_types[:2]):
        return True, "specific_contract_type_conflict"
    if query_types and top_candidate_types and any(query_types.isdisjoint(candidate_types) for candidate_types in top_candidate_types[:2]):
        return True, "contract_type_conflict"
    if len(_extract_procurement_object(query_text)["object_query"].split()) <= 4:
        return True, "short_object_query"
    return False, "high_confidence_no_llm"


def _calibrated_prediction_confidence(rows: List[Dict[str, object]]) -> float | None:
    if not rows:
        return None
    base_confidence = _normalized_prediction_confidence(rows[0].get("score"))
    if base_confidence is None:
        return None
    if len(rows) < 2:
        return base_confidence
    top_score = float(rows[0].get("score") or 0.0)
    second_score = float(rows[1].get("score") or 0.0)
    margin = max(0.0, top_score - second_score)
    margin_factor = min(1.0, margin / 0.12)
    top_branches = {
        str(row.get("cpv_code") or "")[:5]
        for row in rows[:3]
        if str(row.get("cpv_code") or "").strip()
    }
    branch_factor = 1.0 if len(top_branches) <= 1 else 0.72 if len(top_branches) == 2 else 0.55
    return max(0.05, min(1.0, base_confidence * ((0.45 + (0.55 * margin_factor)) * branch_factor)))


def _normalize_header(value: object) -> str:
    return "".join(char for char in str(value or "").casefold() if char.isalnum())


def _first_present(row: Dict[str, object], keys: List[str]) -> object:
    for key in keys:
        normalized_key = _normalize_header(key)
        for candidate in dict.fromkeys([key, normalized_key]):
            if candidate in row and row[candidate] not in {None, ""}:
                return row[candidate]
    return ""


def _ranked_aliases(rank: int, bases: List[str]) -> List[str]:
    aliases: List[str] = []
    for base in bases:
        compact = _normalize_header(base)
        aliases.extend(
            [
                f"{compact}{rank}",
                f"{rank}{compact}",
                f"{compact}rank{rank}",
                f"rank{rank}{compact}",
                f"{base} {rank}",
                f"{base} #{rank}",
                f"{rank} {base}",
            ]
        )
        for marker in ("cpv", "chunk", "vector", "score", "reasoning", "title", "id"):
            marker_index = compact.find(marker)
            if marker_index > 0:
                aliases.append(f"{compact[:marker_index]}rank{rank}{compact[marker_index:]}")
                break
    return list(dict.fromkeys(aliases))


ID_FIELD_ALIASES = [
    "Record ID",
    "id",
    "query_id",
    "question_id",
    "banf_id",
]
QUERY_FIELD_ALIASES = [
    "Query Text",
    "query",
    "question",
    "user_query",
    "query_banf",
    "banf",
    "Notice Description",
]
ANSWER_FIELD_ALIASES = ["LLM Answer", "Answer", "Generated Answer", "Final Answer", "RAG Answer"]
RETRIEVED_PREDICTION_FIELD_ALIASES = ["Retrieved CPV", "Retrieved CPV Code"]
RETRIEVED_SCORE_FIELD_ALIASES = ["Retrieved Vector Score", "Retrieved Score"]
RETRIEVED_CHUNK_TEXT_FIELD_ALIASES = ["Retrieved Chunk Text", "Retrieved Context", "Retrieved Chunks"]
PREDICTION_FIELD_ALIASES = [
    "Predicted CPV",
    "Predicted CPV Code",
    "Predicted answer",
    "Prediction",
    "Predicted",
    "Candidate",
    "Answer CPV",
    "Answer",
    "CPV",
    "Label",
    "Code",
]
SCORE_FIELD_ALIASES = [
    "Predicted Vector Score",
    "Retrieved Vector Score",
    "Vector Score",
    "RRF Score",
    "Score",
    "Confidence",
    "Probability",
    "Retrieval Score",
    "Reranker Score",
    "Final Score",
]
CHUNK_ID_FIELD_ALIASES = ["Chunk ID", "Chunk IDs", "Retrieved Chunk ID", "Retrieved Chunk IDs", "Candidate ID"]
CHUNK_TITLE_FIELD_ALIASES = ["Chunk Titel", "Chunk Title", "Title", "Retrieved Chunk Title", "Candidate Title"]
CHUNK_TEXT_FIELD_ALIASES = [
    "Chunk Text",
    "Chunk",
    "Chunks",
    "Retrieved Chunk Text",
    "Retrieved Chunks",
    "Retrieved Context",
    "Context",
    "Contexts",
    "Candidate Text",
]
EXPECTED_FIELD_ALIASES = [
    "Ground Truth CPV",
    "Expected CPV",
    "Gold CPV",
    "Reference CPV",
    "Expected Answer",
    "Expected",
    "Gold",
    "Reference",
    "Target",
    "Label",
]


def _input_contract_fields() -> Dict[str, object]:
    return {
        "description": "One row per query or one row per ranked candidate. Multiple rows with the same id are treated as one ranked list.",
        "fields": [
            {
                "name": "id",
                "required": True,
                "aliases": [_normalize_header(alias) for alias in ID_FIELD_ALIASES],
                "description": "Stable record id used to group candidates belonging to one query.",
            },
            {
                "name": "query",
                "required": True,
                "aliases": [_normalize_header(alias) for alias in QUERY_FIELD_ALIASES],
                "description": "User request or source text to classify.",
            },
            {
                "name": "expected_answer",
                "required": True,
                "aliases": [_normalize_header(alias) for alias in EXPECTED_FIELD_ALIASES],
                "description": "Reference answer used for evaluation.",
            },
            {
                "name": "predicted_answer",
                "required": True,
                "aliases": ["prediction", "predicted", "predicted_answer", "predicted_cpv", "answer", "candidate", "cpv", "label", "code"],
                "description": "Predicted answer/candidate. For wide top-k files use ranked aliases such as predicted_answer_1, predicted_cpv_2, candidate_3.",
            },
            {
                "name": "score",
                "required": False,
                "aliases": ["score", "confidence", "probability", "retrieval_score", "vector_score", "rrf_score", "reranker_score", "final_score"],
                "description": "Candidate score. Raw scores are preserved and normalized only in diagnostics where needed.",
            },
            {
                "name": "answer",
                "required": False,
                "aliases": [_normalize_header(alias) for alias in ANSWER_FIELD_ALIASES],
                "description": "Optional final answer text if the classifier/RAG system already generated one.",
            },
            {
                "name": "candidate_text",
                "required": False,
                "aliases": ["chunk", "chunks", "chunk_text", "retrieved_chunks", "retrieved_context", "context", "candidate_text"],
                "description": "Optional retrieved text or candidate description used for answer/retrieval diagnostics.",
            },
            {
                "name": "candidate_title",
                "required": False,
                "aliases": ["title", "chunk_title", "retrieved_title", "candidate_title"],
                "description": "Optional candidate title shown in reports and used for retrieval diagnostics.",
            },
            {
                "name": "chunk_id",
                "required": False,
                "aliases": ["chunk_id", "chunk_ids", "retrieved_chunk_id", "retrieved_chunk_ids", "candidate_id"],
                "description": "Optional source/candidate id for traceability.",
            },
        ],
        "ranked_alias_rule": "Append rank numbers to candidate fields, e.g. predicted_answer_1 / score_1 / chunk_text_1 or Predicted CPV #1 / Vector Score #1.",
    }


def _classifier_payload_excluded_keys() -> set[str]:
    keys = set()
    for alias in PREDICTION_FIELD_ALIASES + SCORE_FIELD_ALIASES + ["id", "query", "user_query"]:
        keys.add(alias)
        keys.add(_normalize_header(alias))
    return keys


def _classifier_next_steps(classifier_type: str) -> List[Dict[str, str]]:
    common = [
        {
            "area": "ranked_output",
            "step": "Return a ranked list with stable scores and candidate ids, not only the top answer.",
            "why": "This makes top-1 accuracy, candidate coverage, MRR, calibration, and reranking failures separable.",
        },
        {
            "area": "calibration",
            "step": "Keep raw confidence/retrieval/reranker scores for every candidate.",
            "why": "Reliability curves and review thresholds need the original score signal.",
        },
    ]
    if classifier_type == "prepared_rag_results":
        return [
            *common,
            {
                "area": "input_schema",
                "step": "Use shared aliases for expected answer, predictions, scores, chunk text, and chunk ids.",
                "why": "A consistent schema lets the same evaluator compare prepared files, API output, and live chat checks.",
            },
        ]
    if classifier_type == "api_classifier":
        return [
            *common,
            {
                "area": "api_contract",
                "step": "Expose top-k candidates, scores, optional explanation, and source/context metadata in one response.",
                "why": "The evaluator can then distinguish candidate generation failures from selection or prompt failures.",
            },
        ]
    return common


def _cell_ref_to_column_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return max(index - 1, 0)


def _read_xlsx_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    try:
        raw = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values: List[str] = []
    for item in root.findall("x:si", namespace):
        parts = [node.text or "" for node in item.findall(".//x:t", namespace)]
        values.append("".join(parts))
    return values


def _read_xlsx_table(path: str) -> List[Dict[str, object]]:
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_strings = _read_xlsx_shared_strings(archive)
        sheet_names = [name for name in archive.namelist() if name.startswith("xl/worksheets/sheet")]
        if not sheet_names:
            raise ValueError(f"No worksheets found in {path}.")
        sheet_name = sorted(sheet_names)[0]
        root = ET.fromstring(archive.read(sheet_name))

    rows: List[List[object]] = []
    for row_node in root.findall(".//x:sheetData/x:row", namespace):
        cells: List[object] = []
        for cell_node in row_node.findall("x:c", namespace):
            ref = str(cell_node.attrib.get("r", ""))
            column_index = _cell_ref_to_column_index(ref)
            while len(cells) <= column_index:
                cells.append("")
            cell_type = cell_node.attrib.get("t")
            value_node = cell_node.find("x:v", namespace)
            if cell_type == "s" and value_node is not None:
                try:
                    value = shared_strings[int(value_node.text or "0")]
                except (ValueError, IndexError):
                    value = ""
            elif cell_type == "inlineStr":
                value = "".join(
                    text_node.text or ""
                    for text_node in cell_node.findall(".//x:t", namespace)
                )
            elif value_node is not None:
                raw = value_node.text or ""
                try:
                    number = float(raw)
                    value = int(number) if number.is_integer() else number
                except ValueError:
                    value = raw
            else:
                value = ""
            cells[column_index] = value
        if any(str(cell).strip() for cell in cells):
            rows.append(cells)
    if not rows:
        return []
    headers = [str(value).strip() for value in rows[0]]
    normalized_headers = [_normalize_header(header) or f"column{index}" for index, header in enumerate(headers)]
    records: List[Dict[str, object]] = []
    for values in rows[1:]:
        record: Dict[str, object] = {}
        for index, key in enumerate(normalized_headers):
            record[key] = values[index] if index < len(values) else ""
        record["_raw"] = {
            headers[index] if index < len(headers) else f"Column {index + 1}": values[index]
            for index in range(len(values))
        }
        records.append(record)
    return records


def _extract_cpv_codes(value: object) -> List[str]:
    if value is None:
        return []
    text = str(value)
    codes = re.findall(r"\b\d{8}\b", text)
    if codes:
        return codes
    stripped = text.strip()
    return [stripped] if stripped else []


def _parse_float(value: object, fallback: float | None = None) -> float | None:
    if value in {None, ""}:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        text = str(value).strip().replace(",", ".")
        try:
            return float(text)
        except ValueError:
            return fallback


def _ranked_candidates_from_row(
    row: Dict[str, object],
    *,
    row_index: int,
    prediction_aliases: List[str],
    score_aliases: List[str],
    chunk_text_aliases: List[str],
    chunk_title_aliases: List[str] | None = None,
    chunk_id_aliases: List[str] | None = None,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    prediction_bases = {_normalize_header(base) for base in prediction_aliases}
    chunk_title_aliases = chunk_title_aliases or []
    chunk_id_aliases = chunk_id_aliases or []
    numbered_ranks = set()
    for key in row:
        normalized_key = _normalize_header(key)
        trailing_match = re.search(r"(\d+)$", normalized_key)
        if trailing_match and normalized_key[: trailing_match.start(1)] in prediction_bases:
            numbered_ranks.add(int(trailing_match.group(1)))
            continue
        middle_match = re.search(r"rank(\d+)", normalized_key)
        if not middle_match:
            continue
        collapsed_key = normalized_key[: middle_match.start()] + normalized_key[middle_match.end() :]
        if collapsed_key in prediction_bases:
            numbered_ranks.add(int(middle_match.group(1)))
    for rank in sorted(numbered_ranks):
        predicted_codes = _extract_cpv_codes(_first_present(row, _ranked_aliases(rank, prediction_aliases)))
        if not predicted_codes:
            continue
        score_raw = _first_present(row, _ranked_aliases(rank, score_aliases))
        base_score = _parse_float(score_raw, fallback=max(0.0, 1.0 - rank * 0.001))
        chunk_id = _first_present(row, _ranked_aliases(rank, chunk_id_aliases))
        chunk_title = _first_present(row, _ranked_aliases(rank, chunk_title_aliases))
        chunk_text = _first_present(row, _ranked_aliases(rank, chunk_text_aliases))
        for offset, predicted_code in enumerate(predicted_codes):
            candidates.append(
                {
                    "label": predicted_code,
                    "score": float(base_score or 0.0) - (offset * 0.000001),
                    "rank": rank,
                    "chunk_id": str(chunk_id).strip() if chunk_id else "",
                    "chunk_title": str(chunk_title).strip() if chunk_title else "",
                    "chunk_text": str(chunk_text).strip() if chunk_text else "",
                    "source_row": row,
                    "source_row_index": row_index,
                }
            )
    if candidates:
        return candidates

    predicted_codes = _extract_cpv_codes(_first_present(row, prediction_aliases))
    if not predicted_codes:
        predicted_codes = [""]
    score_raw = _first_present(row, score_aliases)
    base_score = _parse_float(score_raw, fallback=max(0.0, 1.0 - row_index * 0.001))
    rank_raw = _first_present(row, ["rank", "position", "candidate_rank", "topkrank"])
    try:
        rank_value = int(rank_raw) if rank_raw not in {None, ""} else None
    except (TypeError, ValueError):
        rank_value = None
    return [
        {
            "label": predicted_code,
            "score": float(base_score or 0.0) - (offset * 0.000001),
            "rank": rank_value,
            "chunk_id": str(_first_present(row, chunk_id_aliases)).strip(),
            "chunk_title": str(_first_present(row, chunk_title_aliases)).strip(),
            "chunk_text": str(_first_present(row, chunk_text_aliases)).strip(),
            "source_row": row,
            "source_row_index": row_index,
        }
        for offset, predicted_code in enumerate(predicted_codes)
    ]


def _prepared_candidates_from_row(row: Dict[str, object], *, row_index: int) -> List[Dict[str, object]]:
    return _ranked_candidates_from_row(
        row,
        row_index=row_index,
        prediction_aliases=PREDICTION_FIELD_ALIASES,
        score_aliases=SCORE_FIELD_ALIASES,
        chunk_text_aliases=CHUNK_TEXT_FIELD_ALIASES,
        chunk_title_aliases=CHUNK_TITLE_FIELD_ALIASES,
        chunk_id_aliases=CHUNK_ID_FIELD_ALIASES,
    )


def _prepared_retrieved_candidates_from_row(row: Dict[str, object], *, row_index: int) -> List[Dict[str, object]]:
    return _ranked_candidates_from_row(
        row,
        row_index=row_index,
        prediction_aliases=RETRIEVED_PREDICTION_FIELD_ALIASES,
        score_aliases=RETRIEVED_SCORE_FIELD_ALIASES,
        chunk_text_aliases=RETRIEVED_CHUNK_TEXT_FIELD_ALIASES,
    )


def _rank_prepared_candidates(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        [candidate for candidate in candidates if str(candidate.get("label") or "").strip()],
        key=lambda candidate: (
            int(candidate.get("rank") or 999999),
            -float(candidate.get("score") or 0.0),
            int(candidate.get("source_row_index") or 0),
        ),
    )


def _dedupe_ranked_candidates(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    best_by_label: Dict[str, Dict[str, object]] = {}
    for candidate in candidates:
        label = str(candidate.get("label") or "").strip()
        if not label:
            continue
        existing = best_by_label.get(label)
        if existing is None:
            best_by_label[label] = candidate
            continue
        candidate_rank = int(candidate.get("rank") or 999999)
        existing_rank = int(existing.get("rank") or 999999)
        candidate_score = float(candidate.get("score") or 0.0)
        existing_score = float(existing.get("score") or 0.0)
        if (candidate_rank, -candidate_score) < (existing_rank, -existing_score):
            best_by_label[label] = candidate
    return sorted(
        best_by_label.values(),
        key=lambda candidate: (
            int(candidate.get("rank") or 999999),
            -float(candidate.get("score") or 0.0),
            int(candidate.get("source_row_index") or 0),
        ),
    )


def _prepared_candidate_row(
    *,
    query_id: str,
    candidate_label: str,
    score: float,
    rank: int,
    query_text: str,
    catalog_by_code: Dict[str, object],
    candidate: Dict[str, object],
) -> Dict[str, object]:
    row = _catalog_row_from_prediction(
        predicted_label=candidate_label,
        score=score,
        rank=rank,
        query_text=query_text,
        catalog_by_code=catalog_by_code,
    )
    row["retriever"] = "prepared_rag_results"
    row["chunking_strategy"] = "prepared_result"
    row["source_type"] = "prepared_rag_result"
    source_row = candidate["source_row"]
    chunk_id = _first_present(source_row, CHUNK_ID_FIELD_ALIASES)
    chunk_text = _first_present(source_row, CHUNK_TEXT_FIELD_ALIASES)
    chunk_title = _first_present(source_row, CHUNK_TITLE_FIELD_ALIASES)
    if candidate.get("chunk_id"):
        chunk_id = candidate["chunk_id"]
    if candidate.get("chunk_text"):
        chunk_text = candidate["chunk_text"]
    if candidate.get("chunk_title"):
        chunk_title = candidate["chunk_title"]
    if chunk_id:
        row["chunk_id"] = str(chunk_id).strip()
    elif chunk_text:
        row["chunk_id"] = f"{query_id}|rank_{rank}|{candidate_label}"
    if chunk_text:
        row["text"] = str(chunk_text)
    if chunk_title:
        row["title"] = str(chunk_title)
    return row


def _reorder_metric_columns(df, *, trailing_columns: List[str]):
    if df.empty:
        return df
    trailing = [column for column in trailing_columns if column in df.columns]
    leading = [column for column in df.columns if column not in trailing]
    return df[leading + trailing]


def _drop_duplicate_metric_columns(df):
    if df.empty:
        return df
    return df.drop(columns=[column for column in DUPLICATE_METRIC_COLUMNS if column in df.columns])


CPV_IRRELEVANT_COLUMNS = [
    "program_id",
    "program_name",
    "doc_id",
    "doc_path",
    "section_id",
    "chunking_strategy",
    "source_type",
    "answer_scope",
    "evaluation_scope",
    "base_retrieved_chunk_ids",
    "llm_used",
    "llm_status",
    "llm_error",
    "answer_mode",
    "manual_flag",
    "manual_comment",
    "expected_answerable",
    "abstained",
    "abstention_correct",
    "over_answered",
    "false_refusal",
    "ragas_recall_at_k",
    "mean_ragas_recall_at_k",
    "gold_answer",
    "expected_keywords",
    "answer",
    "gold_answer_overlap",
    "answer_gold_support",
    "proxy_faithfulness",
    "proxy_context_relevance",
    "answer_has_gold_substring",
    "answerability_confidence",
    "runtime_retrieval_status",
    "runtime_retrieval_action",
    "runtime_retrieval_reason",
    "runtime_retrieval_score",
    "gold_claim_count",
    "answer_claim_count",
    "context_claim_recall",
    "answer_claim_recall",
    "answer_claim_precision",
    "answer_claim_f1",
    "factual_correctness_precision",
    "factual_correctness_recall",
    "factual_correctness_f1",
    "grounded_claim_ratio",
    "hallucinated_claim_ratio",
    "noise_sensitivity_relevant",
    "noise_sensitivity_irrelevant",
    "context_utilization",
    "context_entities_recall",
    "answer_entity_precision",
    "evidence_attribution_precision",
    "evidence_attribution_recall",
    "evidence_attribution_f1",
    "evidence_coverage",
    "attributed_answer_claim_count",
    "attributed_gold_claim_count",
    "invalid_attribution_count",
    "unsupported_claim_count",
    "missing_gold_claim_count",
    "contradicted_claim_count",
    "claim_diagnostic",
    "claim_judge_used",
    "claim_judge_status",
    "claim_judge_error",
    "claim_judge_model",
    "recommendations",
    "recommendation_count",
    "recommended_component",
    "recommendation_priority",
    "recommended_action",
    "recommendation_reason",
    "recommended_experiment",
    "needs_manual_review",
]


def _drop_irrelevant_classifier_columns(df, *, classifier_type: str):
    if df.empty:
        return df
    if classifier_type not in {"ted_cpv", "api_classifier", "prepared_rag_results"}:
        return df
    return df.drop(columns=[column for column in CPV_IRRELEVANT_COLUMNS if column in df.columns])


def _clean_cpv_rows(rows: List[Dict[str, object]], *, keep_recommendations: bool = False) -> List[Dict[str, object]]:
    dropped_columns = set(CPV_IRRELEVANT_COLUMNS)
    if keep_recommendations:
        dropped_columns -= {
            "recommendations",
            "recommendation_count",
            "recommended_component",
            "recommendation_priority",
            "recommended_action",
            "recommendation_reason",
            "recommended_experiment",
            "needs_manual_review",
        }
    return [
        {key: value for key, value in row.items() if key not in dropped_columns}
        for row in rows
    ]


def _cpv_outcome_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    n_items = len(rows)
    n_correct = sum(1 for row in rows if row.get("auto_flag") == "correct")
    n_incorrect = sum(1 for row in rows if row.get("auto_flag") == "incorrect")
    return {
        "n_items": n_items,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "accuracy": (n_correct / n_items) if n_items else None,
    }


def _cpv_retrieval_summary(metric_rows: List[Dict[str, object]]) -> Dict[str, object]:
    summary = summarize_retrieval_metrics(metric_rows)
    summary.pop("mean_ragas_recall_at_k", None)
    return summary


def _standardize_rag_results_columns(df):
    if df.empty:
        return df
    df = _drop_duplicate_metric_columns(df)
    leading_columns = [
        "question_id",
        "question",
        "retrieval_query",
        "query_augmentation_mode",
        "query_augmentation_status",
        "gold_answer",
        "expected_keywords",
        "retrieved_chunks",
    ]
    trailing_columns = [
        "program_id",
        "program_name",
        "doc_id",
        "answer_scope",
        "evaluation_scope",
        "gold_cpv_label",
        "gold_cpv_description",
        "retrieved_chunk_ids",
        "base_retrieved_chunk_ids",
        "answer",
        "answer_mode",
        "llm_status",
        "llm_error",
        "runtime_retrieval_reason",
        "primary_error_reason",
        "secondary_error_reason",
        "diagnostic_explanation",
        "manual_flag",
        "manual_comment",
        "recommendations",
        "recommendation_count",
        "recommended_component",
        "recommendation_priority",
        "recommended_action",
        "recommendation_reason",
        "recommended_experiment",
        "needs_manual_review",
    ]
    leading = [column for column in leading_columns if column in df.columns]
    trailing = [column for column in trailing_columns if column in df.columns and column not in leading]
    middle = [column for column in df.columns if column not in set(leading + trailing)]
    return df[leading + middle + trailing]


def _retrieved_chunks_payload(rows: List[Dict[str, object]]) -> str:
    return json.dumps(
        [
            {
                "rank": rank,
                "chunk_id": row.get("chunk_id", ""),
                "score": row.get("score"),
                "base_retrieval_score": row.get("base_retrieval_score"),
                "kg_graph_score": row.get("kg_graph_score"),
                "retrieval_source": row.get("retrieval_source", "vector"),
                "kg_candidate_reason": row.get("kg_candidate_reason", ""),
                "kg_path": row.get("kg_path", ""),
                "doc_id": row.get("doc_id", ""),
                "title": row.get("title", ""),
                "text": row.get("text", ""),
            }
            for rank, row in enumerate(rows, start=1)
        ],
        ensure_ascii=False,
    )


def _cpv_classification_metrics(
    *,
    expected_codes: List[str],
    ranked_labels: List[str],
) -> Dict[str, object]:
    gold_set = {str(code) for code in expected_codes if str(code).strip()}
    top1_label = ranked_labels[0] if ranked_labels else ""
    exact_top1_match = bool(top1_label and top1_label in gold_set)
    similarities = [
        similarity
        for gold_code in gold_set
        for similarity in [cpv_structural_similarity(top1_label, gold_code)]
        if similarity is not None
    ]
    prefix_lengths = [
        prefix_length
        for gold_code in gold_set
        for prefix_length in [cpv_common_prefix_length(top1_label, gold_code)]
        if prefix_length is not None
    ]
    hierarchy_similarity = max(similarities) if similarities else None
    common_prefix_length = max(prefix_lengths) if prefix_lengths else None
    hierarchy_match = best_cpv_hierarchy_match(top1_label, list(gold_set)) if top1_label else None
    hierarchy_match_level = int(hierarchy_match["level"]) if hierarchy_match else None
    hierarchy_match_label = str(hierarchy_match["label"]) if hierarchy_match else None
    hierarchy_score = float(hierarchy_match["score"]) if hierarchy_match else None
    same_division_top1 = bool(hierarchy_match_level is not None and hierarchy_match_level >= 2)
    same_group_top1 = bool(hierarchy_match_level is not None and hierarchy_match_level >= 4)
    same_class_top1 = bool(hierarchy_match_level is not None and hierarchy_match_level >= 6)
    same_category_top1 = bool(hierarchy_match_level is not None and hierarchy_match_level >= 8)
    return {
        "exact_top1_match": exact_top1_match,
        "cpv_hierarchy_similarity_top1": hierarchy_similarity,
        "cpv_common_prefix_length_top1": common_prefix_length,
        "hierarchy_match_level_top1": hierarchy_match_level,
        "hierarchy_match_label_top1": hierarchy_match_label,
        "hierarchy_score_top1": hierarchy_score,
        "same_division_top1": same_division_top1,
        "same_group_top1": same_group_top1,
        "same_class_top1": same_class_top1,
        "same_category_top1": same_category_top1,
        "top1_predicted_cpv": top1_label,
        "expected_cpv_codes": json.dumps(list(gold_set), ensure_ascii=False),
    }


def _cpv_auto_flag(classification_metrics: Dict[str, object]) -> str:
    if classification_metrics.get("exact_top1_match") is True:
        return "correct"
    return "incorrect"


def _cpv_diagnostics(
    classification_metrics: Dict[str, object],
    retrieval_metrics: Dict[str, object],
) -> DiagnosticResult:
    if classification_metrics.get("exact_top1_match") is True:
        return DiagnosticResult(
            primary_error_reason="ok",
            secondary_error_reason="exact_match",
            explanation="The top-1 answer matches the expected answer.",
        )
    hierarchy_score = _parse_float(classification_metrics.get("hierarchy_score_top1"), fallback=0.0)
    if hierarchy_score and hierarchy_score > 0.0:
        return DiagnosticResult(
            primary_error_reason="hierarchy_near_miss",
            secondary_error_reason=str(classification_metrics.get("hierarchy_match_label_top1") or "partial_hierarchy_match"),
            explanation="The top-1 answer is not exact, but it lands in the same hierarchy branch.",
        )
    if retrieval_metrics.get("target_doc_retrieved_at_k") is True:
        return DiagnosticResult(
            primary_error_reason="gold_present_but_not_ranked_first",
            secondary_error_reason="reranking_error",
            explanation="The expected answer appears in top-k, but it was not selected as top-1.",
        )
    return DiagnosticResult(
        primary_error_reason="gold_missing_from_top_k",
        secondary_error_reason="candidate_generation_error",
        explanation="The expected answer is missing from the top-k candidates.",
    )


def _all_answer_metric_fields(answer_metrics) -> Dict[str, object]:
    return {
        "expected_answerable": answer_metrics.expected_answerable,
        "abstained": answer_metrics.abstained,
        "over_answered": answer_metrics.over_answered,
        "false_refusal": answer_metrics.false_refusal,
        "answerability_confidence": answer_metrics.answerability_confidence,
        "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
        "runtime_retrieval_action": answer_metrics.runtime_retrieval_action,
        "runtime_retrieval_reason": answer_metrics.runtime_retrieval_reason,
        "gold_answer_overlap": answer_metrics.gold_answer_overlap,
        "proxy_faithfulness": answer_metrics.proxy_faithfulness,
        "proxy_context_relevance": answer_metrics.proxy_context_relevance,
        "context_claim_recall": answer_metrics.context_claim_recall,
        "answer_claim_f1": answer_metrics.answer_claim_f1,
        "factual_correctness_f1": answer_metrics.factual_correctness_f1,
        "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
        "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
        "context_utilization": answer_metrics.context_utilization,
        "context_entities_recall": answer_metrics.context_entities_recall,
        "answer_entity_precision": answer_metrics.answer_entity_precision,
        "evidence_attribution_f1": answer_metrics.evidence_attribution_f1,
        "evidence_coverage": answer_metrics.evidence_coverage,
        "claim_diagnostic": answer_metrics.claim_diagnostic,
    }


def _all_retrieval_metric_fields(retrieval_metrics: Dict[str, object]) -> Dict[str, object]:
    return {
        "mrr_at_k": retrieval_metrics["mrr_at_k"],
        "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
        "recall_at_k": retrieval_metrics["recall_at_k"],
    }


def _cpv_division(code: str) -> str:
    normalized = re.sub(r"\D", "", str(code or ""))
    return normalized[:2] if len(normalized) >= 2 else ""


def _score_entropy(scores: List[float]) -> float | None:
    usable = [max(float(score), 0.0) for score in scores if score is not None]
    if len(usable) <= 1:
        return None
    total = sum(usable)
    if total <= 0:
        return None
    probabilities = [score / total for score in usable if score > 0]
    if not probabilities:
        return None
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return entropy / math.log(len(usable)) if len(usable) > 1 else 0.0


def _cpv_rank_diagnostics(
    *,
    expected_codes: List[str],
    ranked_labels: List[str],
    scores: List[float],
    query_text: str,
    top_k: int,
    prediction_confidence: float | None,
) -> Dict[str, object]:
    gold_set = {str(code).strip() for code in expected_codes if str(code).strip()}
    labels = [str(label).strip() for label in ranked_labels[:top_k] if str(label).strip()]
    top1 = labels[0] if labels else ""
    exact_top1 = bool(top1 and top1 in gold_set)
    gold_rank = next((rank for rank, label in enumerate(labels, start=1) if label in gold_set), None)
    unique_labels = list(dict.fromkeys(labels))
    unique_divisions = list(dict.fromkeys(_cpv_division(label) for label in labels if _cpv_division(label)))
    duplicate_count = max(0, len(labels) - len(unique_labels))
    duplicate_rate = duplicate_count / len(labels) if labels else 0.0
    score_margin = None
    if len(scores) >= 2 and scores[0] is not None and scores[1] is not None:
        score_margin = float(scores[0]) - float(scores[1])
    entropy = _score_entropy(scores[:top_k])
    match_scores = [
        float(match["score"])
        for label in labels
        for match in [best_cpv_hierarchy_match(label, list(gold_set))]
        if match is not None
    ]
    best_hierarchy_score = max(match_scores) if match_scores else 0.0
    query_tokens = re.findall(r"\w+", query_text or "", flags=re.UNICODE)

    if exact_top1:
        failure_mode = "ok"
        likely_bottleneck = "none"
    elif gold_rank is not None:
        failure_mode = "gold_present_but_not_ranked_first"
        likely_bottleneck = "reranker_or_prompt_selection"
    elif best_hierarchy_score >= 0.75:
        failure_mode = "same_class_wrong_code"
        likely_bottleneck = "sibling_disambiguation"
    elif best_hierarchy_score >= 0.25:
        failure_mode = "same_branch_wrong_code"
        likely_bottleneck = "hierarchy_disambiguation"
    else:
        failure_mode = "gold_missing_from_top_k"
        likely_bottleneck = "candidate_generation_or_retriever"

    high_confidence_wrong = bool(not exact_top1 and prediction_confidence is not None and prediction_confidence >= 0.85)
    low_margin_decision = bool(score_margin is not None and score_margin <= 0.05)
    duplicate_candidate_pressure = bool(duplicate_rate >= 0.34)
    low_diversity_at_k = bool(len(labels) > 1 and len(unique_divisions) <= 1)
    short_or_ambiguous_query = bool(len(query_tokens) <= 3)

    if high_confidence_wrong:
        likely_bottleneck = "confidence_calibration"
    elif not exact_top1 and low_margin_decision and gold_rank is not None:
        likely_bottleneck = "reranker_or_prompt_selection"

    return {
        "failure_mode": failure_mode,
        "likely_bottleneck": likely_bottleneck,
        "gold_rank": gold_rank,
        "gold_present_at_k": gold_rank is not None,
        "score_margin_top1_top2": score_margin,
        "score_entropy_at_k": entropy,
        "unique_cpv_at_k": len(unique_labels),
        "unique_division_at_k": len(unique_divisions),
        "duplicate_cpv_at_k": duplicate_count,
        "duplicate_cpv_rate_at_k": duplicate_rate,
        "best_hierarchy_score_at_k": best_hierarchy_score,
        "high_confidence_wrong": high_confidence_wrong,
        "low_margin_decision": low_margin_decision,
        "duplicate_candidate_pressure": duplicate_candidate_pressure,
        "low_diversity_at_k": low_diversity_at_k,
        "short_or_ambiguous_query": short_or_ambiguous_query,
        "query_token_count": len(query_tokens),
        "gold_division": _cpv_division(next(iter(gold_set), "")),
        "top1_division": _cpv_division(top1),
    }


def _json_list(value: object) -> str:
    if value is None:
        return "[]"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _average_numeric(rows: List[Dict[str, object]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and row.get(key) != ""
    ]
    return sum(values) / len(values) if values else None


def _rate(rows: List[Dict[str, object]], key: str) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row.get(key) is True) / len(rows)


def _summarize_cpv_diagnostics(rows: List[Dict[str, object]], *, top_k: int) -> Dict[str, object]:
    total = len(rows)
    incorrect = [row for row in rows if row.get("auto_flag") != "correct"]
    failure_counts = Counter(str(row.get("failure_mode") or "unknown") for row in rows)
    bottleneck_counts = Counter(str(row.get("likely_bottleneck") or "unknown") for row in rows)
    error_failure_counts = Counter(str(row.get("failure_mode") or "unknown") for row in incorrect)
    error_bottleneck_counts = Counter(str(row.get("likely_bottleneck") or "unknown") for row in incorrect)
    confusion_pairs = Counter(
        f"{row.get('gold_division') or '?'}->{row.get('top1_division') or '?'}"
        for row in incorrect
        if row.get("gold_division") or row.get("top1_division")
    )
    data_needs: List[Dict[str, object]] = []
    if top_k < 10:
        data_needs.append(
            {
                "data": "larger ranked lists",
                "why": "Top-3 shows whether the expected answer is nearby, but top-10/top-20 is much better for separating retriever coverage from reranker selection.",
            }
        )
    if rows and not any(str(row.get("classifier_explanation") or "").strip() for row in rows):
        data_needs.append(
            {
                "data": "candidate-level explanations",
                "why": "A short reason for top-1 and contrast against rank-2 would make prompt/selection failures easier to detect.",
            }
        )
    if rows and not any(str(row.get("alternative_gold_cpv_codes") or "").strip() for row in rows):
        data_needs.append(
            {
                "data": "accepted alternative answers",
                "why": "Some records can be ambiguous; alternate acceptable labels prevent marking plausible answers as pure system errors.",
            }
        )
    if rows and not any(str(row.get("manual_error_type") or "").strip() for row in rows):
        data_needs.append(
            {
                "data": "manual error labels for a small sample",
                "why": "A 30-50 row audit with labels like retriever miss, prompt issue, ambiguous reference, and catalog gap would calibrate the automatic diagnosis.",
            }
        )
    if rows and not any(str(row.get("prompt_version") or "").strip() for row in rows):
        data_needs.append(
            {
                "data": "prompt/retriever configuration metadata",
                "why": "Prompt version, retriever type, embedding model, and index version let the evaluator tie failures to concrete system changes.",
            }
        )

    return {
        "n_items": total,
        "n_errors": len(incorrect),
        "failure_mode_counts": dict(failure_counts),
        "likely_bottleneck_counts": dict(bottleneck_counts),
        "dominant_failure_mode": error_failure_counts.most_common(1)[0][0] if error_failure_counts else None,
        "dominant_bottleneck": error_bottleneck_counts.most_common(1)[0][0] if error_bottleneck_counts else None,
        "gold_present_at_k_rate": _rate(rows, "gold_present_at_k"),
        "high_confidence_wrong_rate": _rate(rows, "high_confidence_wrong"),
        "low_margin_decision_rate": _rate(rows, "low_margin_decision"),
        "duplicate_candidate_pressure_rate": _rate(rows, "duplicate_candidate_pressure"),
        "low_diversity_at_k_rate": _rate(rows, "low_diversity_at_k"),
        "short_or_ambiguous_query_rate": _rate(rows, "short_or_ambiguous_query"),
        "error_gold_present_at_k_rate": _rate(incorrect, "gold_present_at_k"),
        "error_high_confidence_wrong_rate": _rate(incorrect, "high_confidence_wrong"),
        "error_low_margin_decision_rate": _rate(incorrect, "low_margin_decision"),
        "error_duplicate_candidate_pressure_rate": _rate(incorrect, "duplicate_candidate_pressure"),
        "error_low_diversity_at_k_rate": _rate(incorrect, "low_diversity_at_k"),
        "error_short_or_ambiguous_query_rate": _rate(incorrect, "short_or_ambiguous_query"),
        "mean_score_margin_top1_top2": _average_numeric(rows, "score_margin_top1_top2"),
        "mean_score_entropy_at_k": _average_numeric(rows, "score_entropy_at_k"),
        "mean_unique_cpv_at_k": _average_numeric(rows, "unique_cpv_at_k"),
        "mean_unique_division_at_k": _average_numeric(rows, "unique_division_at_k"),
        "top_division_confusions": [
            {"pair": pair, "count": count}
            for pair, count in confusion_pairs.most_common(10)
        ],
        "kg": {
            "enabled_rate": _rate(rows, "kg_enabled"),
            "mean_candidate_pool_size": _average_numeric(rows, "kg_candidate_pool_size"),
            "mean_added_candidate_count": _average_numeric(rows, "kg_added_candidate_count"),
            "mean_useful_added_candidate_count": _average_numeric(rows, "kg_useful_added_candidate_count"),
            "strict_gold_delta_rate": _rate(rows, "kg_strict_gold_delta"),
            "expansion_gold_added_rate": _rate(rows, "kg_expansion_gold_added"),
            "expansion_gold_available_rate": _rate(rows, "kg_expansion_gold_available"),
            "mean_expansion_noise_rate": _average_numeric(rows, "kg_expansion_noise_rate"),
            "branch_recall_at_k": _rate(rows, "branch_recall_at_k"),
            "class_recall_at_k": _rate(rows, "class_recall_at_k"),
            "sibling_disambiguation_success_rate": _rate(rows, "sibling_disambiguation_success"),
            "oracle_rerank_ceiling_rate": _rate(rows, "oracle_rerank_ceiling"),
            "oracle_pool_gap_rate": _rate(rows, "kg_oracle_pool_gap"),
            "path_explanation_coverage": _average_numeric(rows, "path_explanation_coverage"),
            "path_coverage_at_k": _average_numeric(rows, "kg_path_coverage_at_k"),
            "ppr_path_share": _average_numeric(rows, "kg_ppr_path_share"),
            "sibling_path_share": _average_numeric(rows, "kg_sibling_path_share"),
            "hierarchy_path_share": _average_numeric(rows, "kg_hierarchy_path_share"),
        },
        "additional_data_that_would_help": data_needs,
    }


def _post_classifier_request(
    *,
    api_url: str,
    payload: Dict[str, object],
    auth_token_env: str,
    extra_headers: Dict[str, object],
    timeout_seconds: float,
) -> Dict[str, object]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    token = os.environ.get(auth_token_env, "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for key, value in extra_headers.items():
        headers[str(key)] = str(value)

    request = urllib.request.Request(api_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Classifier API returned HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Classifier API request failed: {exc}") from exc

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Classifier API returned non-JSON payload: {raw[:400]}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("Classifier API response must be a JSON object.")
    return decoded


def _catalog_row_from_prediction(
    *,
    predicted_label: str,
    score: float,
    rank: int,
    query_text: str,
    catalog_by_code: Dict[str, object],
) -> Dict[str, object]:
    record = catalog_by_code.get(predicted_label)
    if record is None:
        return {
            "chunk_id": predicted_label or f"prediction_{rank}",
            "chunk_index": rank - 1,
            "doc_id": predicted_label,
            "doc_path": "api_classifier",
            "program_id": "cpv",
            "program_name": "CPV",
            "section_id": predicted_label,
            "title": predicted_label,
            "text": predicted_label,
            "chunking_strategy": "api_prediction",
            "source_type": "api_classifier_prediction",
            "cpv_code": predicted_label,
            "cpv_label": predicted_label,
            "cpv_parent_code": "",
            "score": score,
            "retriever": "api_classifier",
        }

    text_parts = [record.label, record.description]
    if record.examples:
        text_parts.append("Examples: " + " | ".join(record.examples))
    return {
        "chunk_id": record.code,
        "chunk_index": rank - 1,
        "doc_id": record.code,
        "doc_path": "cpv_catalog",
        "program_id": "cpv",
        "program_name": "CPV",
        "section_id": record.code,
        "title": record.label,
        "text": "\n".join(part for part in text_parts if part.strip()),
        "chunking_strategy": "api_prediction",
        "source_type": "api_classifier_prediction",
        "cpv_code": record.code,
        "cpv_label": record.label,
        "cpv_parent_code": record.parent_code,
        "score": score,
        "retriever": "api_classifier",
        "query": query_text,
    }


def evaluate_local_ted_cpv_classifier(
    *,
    ted_corpus_export_path: str,
    queries_path: str,
    retriever: str,
    embedding_model: str,
    top_k: int,
    use_examples: bool,
    classifier_label: str,
    run_dir: str,
    create_visualization: bool,
    create_showcase: bool,
    rerank_top_n: int = 0,
    rerank_weight: float = 0.25,
    cross_encoder_rerank: bool = False,
    cross_encoder_model: str | None = None,
    cross_encoder_top_n: int = 10,
    llm_rerank: bool = False,
    llm_rerank_top_n: int = 10,
    llm_rerank_weight: float = 0.4,
    kg_enabled: bool = False,
    kg_graph_weight: float = 0.35,
    kg_profile: str = "balanced",
    kg_algorithm: str | None = None,
    llm_config: LLMConfig | None = None,
    query_augmentation: str | None = None,
    query_augmentation_max_terms: int = 8,
    search_backend_config: Dict[str, object] | None = None,
    search_index_name: str | None = None,
) -> Dict[str, object]:
    import pandas as pd

    ted_notice_db_path = os.path.join(
        str((search_backend_config or {}).get("index_dir") or ".rag_eval_indices"),
        "ted_notices.sqlite",
    )
    bootstrap_catalog = load_cpv_catalog_from_ted_corpus_export(ted_corpus_export_path)
    sync_cpv_profiles_to_db(
        bootstrap_catalog,
        use_examples=use_examples,
        ted_notice_db_path=ted_notice_db_path,
    )
    catalog = load_cpv_catalog_from_db(ted_notice_db_path) or bootstrap_catalog
    queries = load_queries(queries_path)
    chunks = build_cpv_chunks_from_db(ted_notice_db_path) or build_cpv_chunks(
        catalog,
        use_examples=use_examples,
        ted_notice_db_path=ted_notice_db_path,
    )
    retriever_state = build_cpv_multi_retriever(
        chunks,
        embedding_model,
        retriever,
        search_backend_config=search_backend_config,
        index_name=search_index_name,
    )
    cpv_graph = build_cpv_knowledge_graph(catalog) if kg_enabled else None
    chunks_by_code = {str(chunk["cpv_code"]): chunk for chunk in chunks}
    # mean_cpv_hierarchy_similarity_top1 shows how close top-1 is to the reference label.
    parent_lookup = build_parent_lookup(catalog)
    label_by_code = {record.code: record.label for record in catalog}
    description_by_code = {record.code: record.description for record in catalog}

    prediction_records: List[PredictionRecord] = []
    ranking_rows: List[Dict[str, object]] = []
    result_rows: List[Dict[str, object]] = []
    answer_metric_rows: List[Dict[str, object]] = []
    retrieval_metric_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []

    resolved_cross_encoder_model = cross_encoder_model or DEFAULT_CROSS_ENCODER_MODEL
    ce_top_n = max(1, int(cross_encoder_top_n)) if cross_encoder_rerank else 0
    llm_top_n = max(1, int(llm_rerank_top_n)) if llm_rerank else 0
    expanded_selector_pool = 500 if (kg_enabled or rerank_top_n > top_k or cross_encoder_rerank or llm_rerank) else max(top_k, 200)
    pool_size = min(
        max(top_k, rerank_top_n or top_k, ce_top_n, llm_top_n, expanded_selector_pool),
        len(chunks),
    )
    candidate_k = min(
        len(chunks),
        max(top_k, ce_top_n, llm_top_n, pool_size, 200 if (cross_encoder_rerank or llm_rerank or kg_enabled) else top_k),
    )
    effective_rerank_top_n = rerank_top_n if rerank_top_n else (10 if kg_enabled else 0)
    effective_rerank_weight = rerank_weight if rerank_top_n else (0.3 if kg_enabled else 0.25)
    resolved_query_augmentation = query_augmentation or "none"
    base_llm_config = llm_config or LLMConfig(False, "gpt-4.1-mini", "OPENAI_API_KEY", 0.0)

    for query in queries:
        llm_rerank_result = LLMCallResult(answer=None, used=False, status="disabled", error=None)
        query_object = _extract_procurement_object(query.query)
        cleaned_query = str(query_object["cleaned_query"])
        object_query = str(query_object["object_query"])
        procurement_action = str(query_object.get("procurement_action") or "")
        secondary_context = str(query_object.get("secondary_context") or "")
        exclude_as_primary = [str(item) for item in query_object.get("exclude_as_primary", []) if str(item).strip()]
        query_augmentation_config = (
            LLMConfig(True, base_llm_config.model, base_llm_config.api_key_env, 0.0)
            if resolved_query_augmentation in LLM_QUERY_AUGMENTATION_MODES
            else base_llm_config
        )
        query_augmentation_result = augment_query_with_llm(
            object_query,
            query_augmentation_config,
            mode=resolved_query_augmentation,
            max_terms=query_augmentation_max_terms,
        )
        retrieval_query = query_augmentation_result.answer or object_query
        translated_query = retrieval_query
        if resolved_query_augmentation == "translate_en" and translated_query.startswith(object_query):
            translated_query = translated_query[len(object_query):].strip() or retrieval_query
        elif resolved_query_augmentation == "translate_en" and translated_query.startswith(query.query):
            translated_query = translated_query[len(query.query):].strip() or retrieval_query
        object_translation = translated_query if translated_query.strip() else retrieval_query
        db_query_hints = _augment_query_with_cpv_db_hints(
            sqlite_path=str(retriever_state.get("sqlite_path") or ""),
            main_object=object_query,
            translated_query=object_translation,
            procurement_action=procurement_action,
        )
        english_hint_suffix = " ".join(str(item) for item in db_query_hints.get("english_hints", []) if str(item).strip())
        multilingual_hint_suffix = " ".join(str(item) for item in db_query_hints.get("multilingual_hints", []) if str(item).strip())
        if english_hint_suffix:
            object_translation = f"{object_translation} {english_hint_suffix}".strip()
            translated_query = f"{translated_query} {english_hint_suffix}".strip()
        if multilingual_hint_suffix:
            cleaned_query = f"{cleaned_query} {multilingual_hint_suffix}".strip()

        base_retrieved = retrieve_top_k_cpv_multi(
            query_original=cleaned_query or query.query,
            query_translated_to_en=translated_query,
            query_object=object_query,
            query_object_translated=object_translation,
            procurement_action=procurement_action,
            secondary_context=secondary_context,
            exclude_as_primary=exclude_as_primary,
            retriever_state=retriever_state,
            chunks=chunks,
            k=pool_size,
        )
        base_retrieved = rerank_with_lexical_signal(
            query=retrieval_query,
            rows=base_retrieved,
            top_k=min(max(top_k, 25 if kg_enabled else top_k), len(base_retrieved)),
            rerank_top_n=effective_rerank_top_n,
            rerank_weight=effective_rerank_weight,
        )
        base_retrieved = _apply_object_focus_rerank(
            base_retrieved,
            main_object=object_query,
            procurement_action=procurement_action,
            secondary_context=secondary_context,
            exclude_as_primary=exclude_as_primary,
        )
        kg_retrieval = {
            "enabled": False,
            "base_codes": [str(row["cpv_code"]) for row in base_retrieved[:top_k]],
            "candidate_pool_codes": [str(row["cpv_code"]) for row in base_retrieved],
            "added_codes": [],
            "paths": {},
        }
        if kg_enabled and cpv_graph is not None:
            retrieved, kg_retrieval = graph_expand_and_rerank_cpv(
                query=retrieval_query,
                base_rows=base_retrieved,
                chunks_by_code=chunks_by_code,
                graph=cpv_graph,
                top_k=min(candidate_k, len(chunks)),
                graph_weight=kg_graph_weight,
                kg_profile=kg_profile,
                graph_algorithm=kg_algorithm,
            )
        else:
            retrieved = base_retrieved[: min(candidate_k, len(base_retrieved))]

        if cross_encoder_rerank:
            ce_source = [
                dict(row)
                for row in (
                    kg_retrieval.get("pool_rows")
                    if kg_enabled and kg_retrieval.get("pool_rows")
                    else retrieved
                )
            ]
            ce_source.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
            retrieved = rerank_with_cross_encoder(
                query=retrieval_query,
                rows=ce_source[: min(ce_top_n, len(ce_source))],
                top_k=min(max(top_k, llm_top_n), len(chunks)),
                rerank_top_n=min(max(ce_top_n, 200), len(ce_source)),
                model_name=resolved_cross_encoder_model,
                fusion_weight=0.55,
            )
        else:
            retrieved = retrieved[: min(max(top_k, llm_top_n), len(retrieved))]

        retrieved = _apply_object_focus_rerank(
            retrieved,
            main_object=object_query,
            procurement_action=procurement_action,
            secondary_context=secondary_context,
            exclude_as_primary=exclude_as_primary,
        )
        retrieved = _apply_contract_type_rerank(retrieved, f"{object_query} {procurement_action}".strip() or retrieval_query)
        apply_llm, llm_reason = _should_apply_llm_rerank(retrieved, retrieval_query, top_k=top_k)
        if llm_rerank and apply_llm:
            llm_rerank_result_config = LLMConfig(
                True,
                base_llm_config.model,
                base_llm_config.api_key_env,
                0.0,
            )
            retrieved, llm_rerank_result = rerank_candidates_with_llm(
                question=f"{query.query}\nNormalized procurement object: {object_query}",
                rows=retrieved,
                llm_config=llm_rerank_result_config,
                top_k=min(top_k, len(retrieved)),
                rerank_top_n=llm_top_n,
                rerank_weight=llm_rerank_weight,
            )
            llm_rerank_result.status = f"{llm_rerank_result.status}:{llm_reason}"
        elif llm_rerank:
            llm_rerank_result = LLMCallResult(answer=None, used=False, status=f"skipped:{llm_reason}", error=None)
        else:
            retrieved = retrieved[: min(top_k, len(retrieved))]

        prediction_records.append(
            PredictionRecord(
                id=query.id,
                candidates=[
                    RankedCandidate(label=str(row["cpv_code"]), score=float(row["score"]))
                    for row in retrieved
                ],
                metadata={
                    "query": query.query,
                    "cleaned_query": cleaned_query,
                    "object_query": object_query,
                    "procurement_action": procurement_action,
                    "secondary_context": secondary_context,
                    "retrieval_query": retrieval_query,
                    "db_query_hint_status": db_query_hints.get("status"),
                    "db_query_hints_en": " | ".join(str(item) for item in db_query_hints.get("english_hints", [])),
                    "db_query_hints_local": " | ".join(str(item) for item in db_query_hints.get("multilingual_hints", [])),
                    "query_augmentation_mode": resolved_query_augmentation,
                    "query_augmentation_status": query_augmentation_result.status,
                    "llm_rerank_status": llm_rerank_result.status,
                },
            )
        )

        gold_label = label_by_code.get(query.gold_cpv_code, query.gold_cpv_code)
        gold_description = description_by_code.get(query.gold_cpv_code, "")
        item = {
            "id": query.id,
            "question": query.query,
            "gold_answer": f"{query.gold_cpv_code} {gold_label}".strip(),
            "expected_keywords": [query.gold_cpv_code, gold_label],
            "doc_id": query.gold_cpv_code,
        }

        answer = ""
        if retrieved:
            top1 = retrieved[0]
            answer = f"{top1['cpv_code']} {top1['cpv_label']}".strip()

        runtime_retrieval_result = runtime_retrieval_evaluation(
            question=query.query,
            retrieved=retrieved,
        )
        answer_metrics = evaluate_answer_metrics(
            item,
            answer,
            retrieved,
            runtime_retrieval_result=runtime_retrieval_result,
        )
        retrieval_metrics = evaluate_retrieval_metrics(
            item=item,
            retrieved=retrieved,
            candidate_chunks=chunks,
            k=min(top_k, len(chunks)),
        )
        diagnostics = diagnose_failure(
            answer_metrics=answer_metrics,
            retrieval_metrics=retrieval_metrics,
            llm_status="disabled",
            answer_mode="classifier_label",
        )
        classification_metrics = _cpv_classification_metrics(
            expected_codes=[query.gold_cpv_code],
            ranked_labels=[str(row["cpv_code"]) for row in retrieved[:top_k]],
        )
        classifier_auto_flag = _cpv_auto_flag(classification_metrics)
        diagnostics = _cpv_diagnostics(classification_metrics, retrieval_metrics)
        prediction_confidence = _calibrated_prediction_confidence(retrieved)
        cpv_rank_diagnostics = _cpv_rank_diagnostics(
            expected_codes=[query.gold_cpv_code],
            ranked_labels=[str(row["cpv_code"]) for row in retrieved[:top_k]],
            scores=[float(row["score"]) for row in retrieved[:top_k]],
            query_text=query.query,
            top_k=top_k,
            prediction_confidence=prediction_confidence,
        )
        cpv_graph_metrics = cpv_kg_metrics(
            gold_code=query.gold_cpv_code,
            base_codes=kg_retrieval["base_codes"],
            final_codes=[str(row["cpv_code"]) for row in retrieved[:top_k]],
            candidate_pool_codes=kg_retrieval["candidate_pool_codes"],
            added_codes=kg_retrieval["added_codes"],
            paths=kg_retrieval["paths"],
            top_k=top_k,
        ) if kg_enabled else {}

        for rank, row in enumerate(retrieved, start=1):
            relevance_grade = retrieval_relevance_grade(item, row)
            ranking_rows.append(
                {
                    "question_id": query.id,
                    "question": query.query,
                    "cleaned_query": cleaned_query,
                    "object_query": object_query,
                    "retrieval_query": retrieval_query,
                    "query_augmentation_mode": resolved_query_augmentation,
                    "query_augmentation_status": query_augmentation_result.status,
                    "llm_rerank_status": llm_rerank_result.status,
                    "rank": rank,
                    "auto_flag": classifier_auto_flag,
                    "retriever": retriever,
                    "chunk_id": row["chunk_id"],
                    "score": row["score"],
                    "base_retrieval_score": row.get("base_retrieval_score", row["score"]),
                    "cross_encoder_score": row.get("cross_encoder_score", 0.0),
                    "llm_rerank_score": row.get("llm_rerank_score", 0.0),
                    "contract_type_bonus": row.get("contract_type_bonus", 0.0),
                    "query_contract_types": row.get("query_contract_types", ""),
                    "candidate_contract_types": row.get("candidate_contract_types", ""),
                    "kg_graph_score": row.get("kg_graph_score", 0.0),
                    "kg_path_score": row.get("kg_path_score", 0.0),
                    "retrieval_source": row.get("retrieval_source", "vector"),
                    "kg_candidate_reason": row.get("kg_candidate_reason", ""),
                    "kg_path": row.get("kg_path", ""),
                    "doc_id": row["doc_id"],
                    "section_id": row["section_id"],
                    "title": row["title"],
                    "chunking_strategy": row["chunking_strategy"],
                    "source_type": row["source_type"],
                    "text": row["text"],
                    "relevance_grade": relevance_grade,
                    "is_relevant": is_relevant_grade(relevance_grade),
                }
            )

        answer_metric_rows.append(
            {
                "question_id": query.id,
                "question": query.query,
                "retrieval_query": retrieval_query,
                "query_augmentation_mode": resolved_query_augmentation,
                "query_augmentation_status": query_augmentation_result.status,
                "llm_rerank_status": llm_rerank_result.status,
                "llm_rerank_error": llm_rerank_result.error,
                **classification_metrics,
                "answer_accuracy_label": classifier_auto_flag,
                "expected_answerable": answer_metrics.expected_answerable,
                "abstained": answer_metrics.abstained,
                "abstention_correct": answer_metrics.abstention_correct,
                "over_answered": answer_metrics.over_answered,
                "false_refusal": answer_metrics.false_refusal,
                "llm_used": False,
                "llm_status": "disabled",
                "llm_error": None,
                "gold_answer_overlap": answer_metrics.gold_answer_overlap,
                "answer_gold_support": answer_metrics.answer_gold_support,
                "proxy_faithfulness": answer_metrics.proxy_faithfulness,
                "proxy_context_relevance": answer_metrics.proxy_context_relevance,
                "answer_has_gold_substring": answer_metrics.answer_has_gold_substring,
                "answerability_confidence": answer_metrics.answerability_confidence,
                "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
                "factual_correctness_precision": answer_metrics.factual_correctness_precision,
                "factual_correctness_recall": answer_metrics.factual_correctness_recall,
                "factual_correctness_f1": answer_metrics.factual_correctness_f1,
                "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
                "noise_sensitivity_irrelevant": answer_metrics.noise_sensitivity_irrelevant,
                "context_entities_recall": answer_metrics.context_entities_recall,
                "answer_entity_precision": answer_metrics.answer_entity_precision,
                "context_claim_recall": answer_metrics.context_claim_recall,
                "answer_claim_precision": answer_metrics.answer_claim_precision,
                "answer_claim_recall": answer_metrics.answer_claim_recall,
                "answer_claim_f1": answer_metrics.answer_claim_f1,
                "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
                "evidence_attribution_f1": answer_metrics.evidence_attribution_f1,
                "evidence_attribution_precision": answer_metrics.evidence_attribution_precision,
                "evidence_attribution_recall": answer_metrics.evidence_attribution_recall,
                "evidence_coverage": answer_metrics.evidence_coverage,
                "attributed_answer_claim_count": answer_metrics.attributed_answer_claim_count,
                "attributed_gold_claim_count": answer_metrics.attributed_gold_claim_count,
                "invalid_attribution_count": answer_metrics.invalid_attribution_count,
                "unsupported_claim_count": answer_metrics.unsupported_claim_count,
                "missing_gold_claim_count": answer_metrics.missing_gold_claim_count,
                "contradicted_claim_count": answer_metrics.contradicted_claim_count,
                "claim_diagnostic": answer_metrics.claim_diagnostic,
            }
        )
        retrieval_metric_rows.append(
            {
                "question_id": query.id,
                "question": query.query,
                "retrieval_query": retrieval_query,
                "query_augmentation_mode": resolved_query_augmentation,
                "query_augmentation_status": query_augmentation_result.status,
                "llm_rerank_status": llm_rerank_result.status,
                **classification_metrics,
                "program_id": "cpv",
                "program_name": "CPV",
                "doc_id": query.gold_cpv_code,
                "answer_scope": json.dumps({}, ensure_ascii=False),
                "evaluation_scope": json.dumps({"doc_id": [query.gold_cpv_code]}, ensure_ascii=False),
                "mrr_at_k": retrieval_metrics["mrr_at_k"],
                "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
                "recall_at_k": retrieval_metrics["recall_at_k"],
                "first_relevant_rank": retrieval_metrics["first_relevant_rank"],
                "n_relevant_chunks": retrieval_metrics["n_relevant_chunks"],
                "n_retrieved_relevant_chunks": retrieval_metrics["n_retrieved_relevant_chunks"],
                "target_doc_retrieved_at_k": retrieval_metrics["target_doc_retrieved_at_k"],
                "first_target_doc_rank": retrieval_metrics["first_target_doc_rank"],
                "n_retrieved_target_doc_chunks": retrieval_metrics["n_retrieved_target_doc_chunks"],
            }
        )
        diagnostic_rows.append(
            {
                "question_id": query.id,
                "question": query.query,
                "cleaned_query": cleaned_query,
                "object_query": object_query,
                "retrieval_query": retrieval_query,
                "query_augmentation_mode": resolved_query_augmentation,
                "query_augmentation_status": query_augmentation_result.status,
                "llm_rerank_status": llm_rerank_result.status,
                "program_id": "cpv",
                "program_name": "CPV",
                "doc_id": query.gold_cpv_code,
                "answer_scope": json.dumps({}, ensure_ascii=False),
                "evaluation_scope": json.dumps({"doc_id": [query.gold_cpv_code]}, ensure_ascii=False),
                "primary_error_reason": diagnostics.primary_error_reason,
                "secondary_error_reason": diagnostics.secondary_error_reason,
                **cpv_rank_diagnostics,
                **cpv_graph_metrics,
                "kg_candidate_added_codes": _json_list(cpv_graph_metrics.get("kg_candidate_added_codes")),
                "kg_candidate_pool_codes": _json_list(cpv_graph_metrics.get("kg_candidate_pool_codes")),
                "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
                "context_claim_recall": answer_metrics.context_claim_recall,
                "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
                "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
                "context_entities_recall": answer_metrics.context_entities_recall,
                "explanation": diagnostics.explanation,
            }
        )

        result_rows.append(
            apply_question_recommendations(
                {
                "question_id": query.id,
                "question": query.query,
                "cleaned_query": cleaned_query,
                "object_query": object_query,
                "retrieval_query": retrieval_query,
                "query_augmentation_mode": resolved_query_augmentation,
                "query_augmentation_status": query_augmentation_result.status,
                "llm_rerank_status": llm_rerank_result.status,
                "program_id": "cpv",
                "program_name": "CPV",
                "doc_id": query.gold_cpv_code,
                "answer_scope": json.dumps({}, ensure_ascii=False),
                "evaluation_scope": json.dumps({"doc_id": [query.gold_cpv_code]}, ensure_ascii=False),
                "gold_answer": item["gold_answer"],
                "expected_keywords": json.dumps(item["expected_keywords"], ensure_ascii=False),
                **classification_metrics,
                **cpv_rank_diagnostics,
                **cpv_graph_metrics,
                **_all_retrieval_metric_fields(retrieval_metrics),
                **_all_answer_metric_fields(answer_metrics),
                "expected_answerable": answer_metrics.expected_answerable,
                "abstained": answer_metrics.abstained,
                "abstention_correct": answer_metrics.abstention_correct,
                "over_answered": answer_metrics.over_answered,
                "false_refusal": answer_metrics.false_refusal,
                "prediction_confidence": prediction_confidence,
                "retrieved_chunk_ids": json.dumps([row["chunk_id"] for row in retrieved], ensure_ascii=False),
                "retrieved_chunks": _retrieved_chunks_payload(retrieved),
                "base_retrieved_chunk_ids": json.dumps(
                    [row["chunk_id"] for row in base_retrieved[:top_k]], ensure_ascii=False
                ),
                "kg_candidate_added_codes": _json_list(cpv_graph_metrics.get("kg_candidate_added_codes")),
                "kg_candidate_pool_codes": _json_list(cpv_graph_metrics.get("kg_candidate_pool_codes")),
                "kg_top1_path": cpv_graph_metrics.get("kg_top1_path", ""),
                "mrr_at_k": retrieval_metrics["mrr_at_k"],
                "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
                "recall_at_k": retrieval_metrics["recall_at_k"],
                "target_doc_retrieved_at_k": retrieval_metrics["target_doc_retrieved_at_k"],
                "first_target_doc_rank": retrieval_metrics["first_target_doc_rank"],
                "n_retrieved_target_doc_chunks": retrieval_metrics["n_retrieved_target_doc_chunks"],
                "gold_answer_overlap": answer_metrics.gold_answer_overlap,
                "answer_gold_support": answer_metrics.answer_gold_support,
                "proxy_faithfulness": answer_metrics.proxy_faithfulness,
                "proxy_context_relevance": answer_metrics.proxy_context_relevance,
                "answerability_confidence": answer_metrics.answerability_confidence,
                "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
                "runtime_retrieval_action": answer_metrics.runtime_retrieval_action,
                "runtime_retrieval_reason": answer_metrics.runtime_retrieval_reason,
                "factual_correctness_precision": answer_metrics.factual_correctness_precision,
                "factual_correctness_recall": answer_metrics.factual_correctness_recall,
                "factual_correctness_f1": answer_metrics.factual_correctness_f1,
                "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
                "noise_sensitivity_irrelevant": answer_metrics.noise_sensitivity_irrelevant,
                "context_entities_recall": answer_metrics.context_entities_recall,
                "answer_entity_precision": answer_metrics.answer_entity_precision,
                "context_claim_recall": answer_metrics.context_claim_recall,
                "answer_claim_precision": answer_metrics.answer_claim_precision,
                "answer_claim_recall": answer_metrics.answer_claim_recall,
                "answer_claim_f1": answer_metrics.answer_claim_f1,
                "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
                "evidence_attribution_f1": answer_metrics.evidence_attribution_f1,
                "claim_diagnostic": answer_metrics.claim_diagnostic,
                "llm_status": "disabled",
                "llm_error": None,
                "answer": answer,
                "answer_mode": "classifier_label",
                "auto_flag": classifier_auto_flag,
                "primary_error_reason": diagnostics.primary_error_reason,
                "secondary_error_reason": diagnostics.secondary_error_reason,
                "diagnostic_explanation": diagnostics.explanation,
                "manual_flag": "",
                "manual_comment": "",
                "gold_cpv_label": gold_label,
                "gold_cpv_description": gold_description,
            }
            )
        )

    results_df = _drop_irrelevant_classifier_columns(
        _standardize_rag_results_columns(pd.DataFrame(result_rows)),
        classifier_type="ted_cpv",
    )

    rag_results_csv = os.path.join(run_dir, "rag_results.csv")
    retrieved_csv = os.path.join(run_dir, "retrieved_chunks.csv")
    retrieval_metrics_csv = os.path.join(run_dir, "retrieval_metrics.csv")
    diagnostics_csv = os.path.join(run_dir, "diagnostics.csv")
    results_df.to_csv(rag_results_csv, index=False)
    _drop_irrelevant_classifier_columns(
        pd.DataFrame(ranking_rows),
        classifier_type="ted_cpv",
    ).to_csv(retrieved_csv, index=False)
    _drop_irrelevant_classifier_columns(
        _drop_duplicate_metric_columns(pd.DataFrame(retrieval_metric_rows)),
        classifier_type="ted_cpv",
    ).to_csv(retrieval_metrics_csv, index=False)
    _drop_irrelevant_classifier_columns(
        pd.DataFrame(diagnostic_rows),
        classifier_type="ted_cpv",
    ).to_csv(diagnostics_csv, index=False)

    metric_rows = _clean_cpv_rows(result_rows)
    advisor_rows = _clean_cpv_rows(result_rows, keep_recommendations=True)
    aggregate_answer_metrics = _cpv_outcome_summary(metric_rows)
    aggregate_retrieval_metrics = _cpv_retrieval_summary(retrieval_metric_rows)
    aggregate_diagnostics = summarize_diagnostics(diagnostic_rows)
    aggregate_cpv_diagnostics = _summarize_cpv_diagnostics(metric_rows, top_k=top_k)
    classifier_calibration = summarize_confidence_calibration(
        result_rows,
        confidence_key="prediction_confidence",
        correct_fn=lambda row: row.get("auto_flag") == "correct",
    )

    retrieval_metrics_json = os.path.join(run_dir, "retrieval_metrics_summary.json")
    diagnostics_json = os.path.join(run_dir, "diagnostics_summary.json")
    with open(retrieval_metrics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_retrieval_metrics, f, ensure_ascii=False, indent=2)
    with open(diagnostics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_diagnostics, f, ensure_ascii=False, indent=2)

    ranking_metrics = evaluate_ranked_predictions(
        build_items_from_cpv_queries(queries_path),
        prediction_records,
        top_k=top_k,
        distance_fn=cpv_structural_distance,
    )
    top1_top2_margins = []
    for record in prediction_records:
        if len(record.candidates) < 2:
            continue
        first = record.candidates[0].score
        second = record.candidates[1].score
        if first is None or second is None:
            continue
        top1_top2_margins.append(float(first) - float(second))

    summary = {
        "experiment": classifier_label,
        "chunking_strategy": "cpv_profile_db",
        "retriever": retriever,
        "search_backend": retriever_state.get("search_backend", {"backend": "local", "index_name": None}),
        "chunk_size": 0,
        "chunk_overlap": 0,
        "hybrid_alpha": None,
        "top_k": top_k,
        "reranker": {
            "enabled": rerank_top_n > 1 and rerank_weight > 0.0,
            "type": "lexical_overlap",
            "top_n": rerank_top_n,
            "weight": rerank_weight,
        },
        "cross_encoder_reranker": {
            "enabled": cross_encoder_rerank,
            "type": "cross_encoder",
            "model": resolved_cross_encoder_model if cross_encoder_rerank else None,
            "top_n": ce_top_n if cross_encoder_rerank else 0,
            "fusion_weight": 0.55 if cross_encoder_rerank else 0.0,
        },
        "llm_reranker": {
            "enabled": llm_rerank,
            "type": "llm",
            "model": base_llm_config.model if llm_rerank else None,
            "top_n": llm_top_n if llm_rerank else 0,
            "weight": llm_rerank_weight if llm_rerank else 0.0,
            "conditional": True if llm_rerank else False,
        },
        "n_chunks": len(chunks),
        "n_questions": len(queries),
        "n_correct": int((results_df["auto_flag"] == "correct").sum()),
        "n_incorrect": int((results_df["auto_flag"] == "incorrect").sum()),
        "answer_metrics": aggregate_answer_metrics,
        "retrieval_metrics": aggregate_retrieval_metrics,
        "diagnostics": aggregate_diagnostics,
        "llm": {
            "enabled": base_llm_config.enabled or resolved_query_augmentation in LLM_QUERY_AUGMENTATION_MODES or llm_rerank,
            "model": base_llm_config.model if (base_llm_config.enabled or resolved_query_augmentation in LLM_QUERY_AUGMENTATION_MODES or llm_rerank) else None,
            "answer_generation": False,
        },
        "evaluation_settings": {
            "query_augmentation": resolved_query_augmentation,
            "query_augmentation_max_terms": query_augmentation_max_terms,
            "query_cleaning": True,
            "contract_type_aware_rerank": True,
        },
        "classifier": {
            "type": "ted_cpv",
            "label": classifier_label,
            "use_examples": use_examples,
            "kg": {
                "enabled": kg_enabled,
                "type": "cpv_taxonomy_graph",
                "graph_weight": kg_graph_weight if kg_enabled else None,
                "profile": kg_profile if kg_enabled else None,
                "algorithm": kg_algorithm if kg_enabled else None,
                "n_nodes": len(cpv_graph.nodes) if cpv_graph is not None else 0,
            },
            "avg_top1_top2_margin": (
                sum(top1_top2_margins) / len(top1_top2_margins) if top1_top2_margins else None
            ),
            "explanation_coverage": 0.0,
            "ranking_metrics": ranking_metrics,
            "calibration": classifier_calibration,
            "cpv_diagnostics": aggregate_cpv_diagnostics,
        },
        "visualization": {
            "enabled": create_visualization or create_showcase,
            "strategy_score_profile_svg": None,
            "strategy_chunk_alignment_svg": None,
        },
        "showcase": {
            "enabled": False,
            "score_profile_svg": None,
            "chunk_alignment_svg": None,
            "metric_overview_svg": None,
            "diagnostics_svg": None,
            "showcase_md": None,
            "improvement_summary": None,
        },
        "outputs": {
            "rag_results_csv": rag_results_csv,
            "retrieved_chunks_csv": retrieved_csv,
            "retrieval_metrics_csv": retrieval_metrics_csv,
            "retrieval_metrics_summary_json": retrieval_metrics_json,
            "diagnostics_csv": diagnostics_csv,
            "diagnostics_summary_json": diagnostics_json,
            "strategy_score_profile_svg": None,
            "strategy_chunk_alignment_svg": None,
            "strategy_metric_overview_svg": None,
            "strategy_diagnostics_svg": None,
            "strategy_showcase_md": None,
        },
    }
    quality_advisor = build_run_advisor(summary, advisor_rows)
    quality_advisor_json = os.path.join(run_dir, "quality_advisor.json")
    with open(quality_advisor_json, "w", encoding="utf-8") as f:
        json.dump(quality_advisor, f, ensure_ascii=False, indent=2)
    quality_report_md = write_quality_report(
        os.path.join(run_dir, "quality_report.md"),
        quality_advisor,
        summary,
    )
    summary["advisor"] = quality_advisor
    summary["outputs"]["quality_advisor_json"] = quality_advisor_json
    summary["outputs"]["quality_report_md"] = quality_report_md
    if create_visualization or create_showcase:
        showcase_bundle = write_classifier_showcase_bundle(
            summary=summary,
            ranking_rows=ranking_rows,
            experiment_dir=run_dir,
        )
        _apply_showcase_bundle(summary, showcase_bundle)

    summary_json = os.path.join(run_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["outputs"]["summary_json"] = summary_json
    return summary


def evaluate_prepared_rag_results_classifier(
    *,
    prepared_results_path: str,
    ted_corpus_export_path: str,
    top_k: int,
    classifier_label: str,
    run_dir: str,
    create_visualization: bool,
    create_showcase: bool,
) -> Dict[str, object]:
    import pandas as pd

    rows = _read_xlsx_table(prepared_results_path)
    if not rows:
        raise ValueError(f"No data rows found in {prepared_results_path}.")

    catalog = load_cpv_catalog_from_ted_corpus_export(ted_corpus_export_path)
    catalog_chunks = build_cpv_chunks(catalog, use_examples=True)
    parent_lookup = build_parent_lookup(catalog)
    catalog_by_code = {record.code: record for record in catalog}
    label_by_code = {record.code: record.label for record in catalog}
    description_by_code = {record.code: record.description for record in catalog}

    grouped: Dict[str, Dict[str, object]] = {}
    for row_index, row in enumerate(rows, start=1):
        query_id = str(_first_present(row, ID_FIELD_ALIASES) or row_index).strip()
        query_text = str(_first_present(row, QUERY_FIELD_ALIASES) or "").strip()
        expected_codes = _extract_cpv_codes(_first_present(row, EXPECTED_FIELD_ALIASES))
        llm_answer = str(_first_present(row, ANSWER_FIELD_ALIASES) or "").strip()

        group = grouped.setdefault(
            query_id,
            {
                "id": query_id,
                "query": query_text,
                "expected_codes": expected_codes,
                "llm_answer": llm_answer,
                "predicted_rows": [],
                "retrieved_rows": [],
            },
        )
        if query_text and not group["query"]:
            group["query"] = query_text
        if expected_codes and not group["expected_codes"]:
            group["expected_codes"] = expected_codes
        if llm_answer and not group["llm_answer"]:
            group["llm_answer"] = llm_answer
        for candidate in _prepared_candidates_from_row(row, row_index=row_index):
            if candidate.get("rank") is None:
                candidate["rank"] = len(group["predicted_rows"]) + 1
            group["predicted_rows"].append(candidate)
        for candidate in _prepared_retrieved_candidates_from_row(row, row_index=row_index):
            if candidate.get("rank") is None:
                candidate["rank"] = len(group["retrieved_rows"]) + 1
            group["retrieved_rows"].append(candidate)

    prediction_records: List[PredictionRecord] = []
    evaluation_items = []
    ranking_rows: List[Dict[str, object]] = []
    result_rows: List[Dict[str, object]] = []
    answer_metric_rows: List[Dict[str, object]] = []
    retrieval_metric_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    effective_predicted_top_k = max(
        (len(_rank_prepared_candidates(group["predicted_rows"])) for group in grouped.values()),
        default=0,
    )
    effective_retrieved_top_k = max(
        (
            len(_rank_prepared_candidates(group["retrieved_rows"]) or _rank_prepared_candidates(group["predicted_rows"]))
            for group in grouped.values()
        ),
        default=0,
    )
    effective_predicted_top_k = min(top_k, effective_predicted_top_k) if effective_predicted_top_k else 0
    effective_retrieved_top_k = max(
        effective_predicted_top_k,
        min(max(top_k, effective_retrieved_top_k), effective_retrieved_top_k) if effective_retrieved_top_k else 0,
    )

    for group in grouped.values():
        query_id = str(group["id"])
        query_text = str(group["query"])
        expected_codes = [str(code) for code in group["expected_codes"] if str(code).strip()]
        candidates_raw = _rank_prepared_candidates(group["predicted_rows"])[:effective_predicted_top_k]
        retrieved_candidates_raw = _rank_prepared_candidates(group["retrieved_rows"])[:effective_retrieved_top_k]
        if not retrieved_candidates_raw:
            retrieved_candidates_raw = candidates_raw
        predicted_rows_for_query = [
            _prepared_candidate_row(
                query_id=query_id,
                candidate_label=str(candidate["label"]),
                score=float(candidate["score"]),
                rank=rank,
                query_text=query_text,
                catalog_by_code=catalog_by_code,
                candidate=candidate,
            )
            for rank, candidate in enumerate(candidates_raw, start=1)
            if str(candidate.get("label", "")).strip()
        ]
        retrieved_rows_for_query = [
            _prepared_candidate_row(
                query_id=query_id,
                candidate_label=str(candidate["label"]),
                score=float(candidate["score"]),
                rank=rank,
                query_text=query_text,
                catalog_by_code=catalog_by_code,
                candidate=candidate,
            )
            for rank, candidate in enumerate(retrieved_candidates_raw, start=1)
            if str(candidate.get("label", "")).strip()
        ]
        normalized_candidates = [
            RankedCandidate(
                label=str(row["cpv_code"]),
                score=float(row["score"]),
                metadata={"source": "prepared_rag_results"},
            )
            for row in predicted_rows_for_query
        ]
        ranked_labels = [candidate.label for candidate in normalized_candidates[:effective_predicted_top_k]]
        classification_metrics = _cpv_classification_metrics(
            expected_codes=expected_codes,
            ranked_labels=ranked_labels,
        )
        prediction_records.append(
            PredictionRecord(
                id=query_id,
                candidates=normalized_candidates,
                metadata={"query": query_text},
            )
        )
        evaluation_items.append(
            EvaluationItem(
                id=query_id,
                query=query_text,
                gold_labels=expected_codes,
                metadata={"source": prepared_results_path},
            )
        )

        primary_gold = expected_codes[0] if expected_codes else ""
        gold_label = label_by_code.get(primary_gold, primary_gold)
        gold_description = description_by_code.get(primary_gold, "")
        item = {
            "id": query_id,
            "question": query_text,
            "gold_answer": f"{primary_gold} {gold_label}".strip(),
            "expected_keywords": [code for code in [primary_gold, gold_label] if code],
            "doc_id": primary_gold,
        }

        answer = str(group.get("llm_answer") or "").strip()
        answer_mode = "prepared_llm_answer" if answer else "prepared_classifier_label"
        if not answer and normalized_candidates:
            top1 = normalized_candidates[0]
            top1_label = label_by_code.get(top1.label, top1.label)
            answer = f"{top1.label} {top1_label}".strip()

        runtime_retrieval_result = runtime_retrieval_evaluation(
            question=query_text,
            retrieved=retrieved_rows_for_query,
        )
        answer_metrics = evaluate_answer_metrics(
            item,
            answer,
            retrieved_rows_for_query,
            runtime_retrieval_result=runtime_retrieval_result,
        )
        retrieval_metrics = evaluate_retrieval_metrics(
            item=item,
            retrieved=retrieved_rows_for_query,
            candidate_chunks=catalog_chunks,
            k=min(effective_retrieved_top_k, len(catalog_chunks)),
        )
        diagnostics = diagnose_failure(
            answer_metrics=answer_metrics,
            retrieval_metrics=retrieval_metrics,
            llm_status="provided" if group.get("llm_answer") else "disabled",
            answer_mode=answer_mode,
        )
        classifier_auto_flag = _cpv_auto_flag(classification_metrics)
        diagnostics = _cpv_diagnostics(classification_metrics, retrieval_metrics)
        prediction_confidence = (
            _normalized_prediction_confidence(normalized_candidates[0].score)
            if normalized_candidates and normalized_candidates[0].score is not None
            else None
        )
        cpv_rank_diagnostics = _cpv_rank_diagnostics(
            expected_codes=expected_codes,
            ranked_labels=ranked_labels,
            scores=[
                float(candidate.score)
                for candidate in normalized_candidates[:effective_predicted_top_k]
                if candidate.score is not None
            ],
            query_text=query_text,
            top_k=effective_predicted_top_k,
            prediction_confidence=prediction_confidence,
        )

        for rank, row in enumerate(retrieved_rows_for_query, start=1):
            relevance_grade = retrieval_relevance_grade(item, row)
            ranking_rows.append(
                {
                    "question_id": query_id,
                    "question": query_text,
                    "rank": rank,
                    "auto_flag": classifier_auto_flag,
                    "retriever": "prepared_rag_results",
                    "chunk_id": row["chunk_id"],
                    "score": row["score"],
                    "doc_id": row["doc_id"],
                    "section_id": row["section_id"],
                    "title": row["title"],
                    "chunking_strategy": row["chunking_strategy"],
                    "source_type": row["source_type"],
                    "text": row["text"],
                    "relevance_grade": relevance_grade,
                    "is_relevant": is_relevant_grade(relevance_grade),
                }
            )

        answer_metric_rows.append(
            {
                "question_id": query_id,
                "question": query_text,
                **classification_metrics,
                "answer_accuracy_label": classifier_auto_flag,
                "expected_answerable": answer_metrics.expected_answerable,
                "abstained": answer_metrics.abstained,
                "abstention_correct": answer_metrics.abstention_correct,
                "over_answered": answer_metrics.over_answered,
                "false_refusal": answer_metrics.false_refusal,
                "llm_used": bool(group.get("llm_answer")),
                "llm_status": "provided" if group.get("llm_answer") else "disabled",
                "llm_error": None,
                "gold_answer_overlap": answer_metrics.gold_answer_overlap,
                "answer_gold_support": answer_metrics.answer_gold_support,
                "proxy_faithfulness": answer_metrics.proxy_faithfulness,
                "proxy_context_relevance": answer_metrics.proxy_context_relevance,
                "answer_has_gold_substring": answer_metrics.answer_has_gold_substring,
                "answerability_confidence": answer_metrics.answerability_confidence,
                "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
                "runtime_retrieval_action": answer_metrics.runtime_retrieval_action,
                "runtime_retrieval_score": answer_metrics.runtime_retrieval_score,
                "runtime_retrieval_reason": answer_metrics.runtime_retrieval_reason,
                "factual_correctness_precision": answer_metrics.factual_correctness_precision,
                "factual_correctness_recall": answer_metrics.factual_correctness_recall,
                "factual_correctness_f1": answer_metrics.factual_correctness_f1,
                "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
                "noise_sensitivity_irrelevant": answer_metrics.noise_sensitivity_irrelevant,
                "context_entities_recall": answer_metrics.context_entities_recall,
                "answer_entity_precision": answer_metrics.answer_entity_precision,
                "context_claim_recall": answer_metrics.context_claim_recall,
                "answer_claim_precision": answer_metrics.answer_claim_precision,
                "answer_claim_recall": answer_metrics.answer_claim_recall,
                "answer_claim_f1": answer_metrics.answer_claim_f1,
                "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
                "evidence_attribution_f1": answer_metrics.evidence_attribution_f1,
                "evidence_attribution_precision": answer_metrics.evidence_attribution_precision,
                "evidence_attribution_recall": answer_metrics.evidence_attribution_recall,
                "evidence_coverage": answer_metrics.evidence_coverage,
                "attributed_answer_claim_count": answer_metrics.attributed_answer_claim_count,
                "attributed_gold_claim_count": answer_metrics.attributed_gold_claim_count,
                "invalid_attribution_count": answer_metrics.invalid_attribution_count,
                "unsupported_claim_count": answer_metrics.unsupported_claim_count,
                "missing_gold_claim_count": answer_metrics.missing_gold_claim_count,
                "contradicted_claim_count": answer_metrics.contradicted_claim_count,
                "claim_diagnostic": answer_metrics.claim_diagnostic,
                "answer": answer,
                "answer_mode": answer_mode,
            }
        )
        retrieval_metric_rows.append(
            {
                "question_id": query_id,
                "question": query_text,
                **classification_metrics,
                "program_id": "cpv",
                "program_name": "CPV",
                "doc_id": primary_gold,
                "answer_scope": json.dumps({}, ensure_ascii=False),
                "evaluation_scope": json.dumps({"doc_id": expected_codes}, ensure_ascii=False),
                "mrr_at_k": retrieval_metrics["mrr_at_k"],
                "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
                "recall_at_k": retrieval_metrics["recall_at_k"],
                "first_relevant_rank": retrieval_metrics["first_relevant_rank"],
                "n_relevant_chunks": retrieval_metrics["n_relevant_chunks"],
                "n_retrieved_relevant_chunks": retrieval_metrics["n_retrieved_relevant_chunks"],
                "target_doc_retrieved_at_k": retrieval_metrics["target_doc_retrieved_at_k"],
                "first_target_doc_rank": retrieval_metrics["first_target_doc_rank"],
                "n_retrieved_target_doc_chunks": retrieval_metrics["n_retrieved_target_doc_chunks"],
            }
        )
        diagnostic_rows.append(
            {
                "question_id": query_id,
                "question": query_text,
                "program_id": "cpv",
                "program_name": "CPV",
                "doc_id": primary_gold,
                "answer_scope": json.dumps({}, ensure_ascii=False),
                "evaluation_scope": json.dumps({"doc_id": expected_codes}, ensure_ascii=False),
                "primary_error_reason": diagnostics.primary_error_reason,
                "secondary_error_reason": diagnostics.secondary_error_reason,
                **cpv_rank_diagnostics,
                "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
                "context_claim_recall": answer_metrics.context_claim_recall,
                "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
                "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
                "context_entities_recall": answer_metrics.context_entities_recall,
                "explanation": diagnostics.explanation,
            }
        )

        result_rows.append(
            apply_question_recommendations(
                {
                    "question_id": query_id,
                    "question": query_text,
                    "program_id": "cpv",
                    "program_name": "CPV",
                    "doc_id": primary_gold,
                    "answer_scope": json.dumps({}, ensure_ascii=False),
                    "evaluation_scope": json.dumps({"doc_id": expected_codes}, ensure_ascii=False),
                    "gold_answer": item["gold_answer"],
                    "expected_keywords": json.dumps(item["expected_keywords"], ensure_ascii=False),
                    **classification_metrics,
                    **cpv_rank_diagnostics,
                    **_all_retrieval_metric_fields(retrieval_metrics),
                    **_all_answer_metric_fields(answer_metrics),
                    "expected_answerable": answer_metrics.expected_answerable,
                    "abstained": answer_metrics.abstained,
                    "abstention_correct": answer_metrics.abstention_correct,
                    "over_answered": answer_metrics.over_answered,
                    "false_refusal": answer_metrics.false_refusal,
                    "prediction_confidence": prediction_confidence,
                    "retrieved_chunk_ids": json.dumps([row["chunk_id"] for row in retrieved_rows_for_query], ensure_ascii=False),
                    "retrieved_chunks": _retrieved_chunks_payload(retrieved_rows_for_query),
                    "base_retrieved_chunk_ids": json.dumps([row["chunk_id"] for row in retrieved_rows_for_query], ensure_ascii=False),
                    "mrr_at_k": retrieval_metrics["mrr_at_k"],
                    "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
                    "recall_at_k": retrieval_metrics["recall_at_k"],
                    "target_doc_retrieved_at_k": retrieval_metrics["target_doc_retrieved_at_k"],
                    "first_relevant_rank": retrieval_metrics["first_relevant_rank"],
                    "n_relevant_chunks": retrieval_metrics["n_relevant_chunks"],
                    "n_retrieved_relevant_chunks": retrieval_metrics["n_retrieved_relevant_chunks"],
                    "first_target_doc_rank": retrieval_metrics["first_target_doc_rank"],
                    "n_retrieved_target_doc_chunks": retrieval_metrics["n_retrieved_target_doc_chunks"],
                    "gold_answer_overlap": answer_metrics.gold_answer_overlap,
                    "answer_gold_support": answer_metrics.answer_gold_support,
                    "proxy_faithfulness": answer_metrics.proxy_faithfulness,
                    "proxy_context_relevance": answer_metrics.proxy_context_relevance,
                    "answerability_confidence": answer_metrics.answerability_confidence,
                    "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
                    "runtime_retrieval_action": answer_metrics.runtime_retrieval_action,
                    "runtime_retrieval_reason": answer_metrics.runtime_retrieval_reason,
                    "factual_correctness_precision": answer_metrics.factual_correctness_precision,
                    "factual_correctness_recall": answer_metrics.factual_correctness_recall,
                    "factual_correctness_f1": answer_metrics.factual_correctness_f1,
                    "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
                    "noise_sensitivity_irrelevant": answer_metrics.noise_sensitivity_irrelevant,
                    "context_entities_recall": answer_metrics.context_entities_recall,
                    "answer_entity_precision": answer_metrics.answer_entity_precision,
                    "context_claim_recall": answer_metrics.context_claim_recall,
                    "answer_claim_precision": answer_metrics.answer_claim_precision,
                    "answer_claim_recall": answer_metrics.answer_claim_recall,
                    "answer_claim_f1": answer_metrics.answer_claim_f1,
                    "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
                    "evidence_attribution_f1": answer_metrics.evidence_attribution_f1,
                    "evidence_attribution_precision": answer_metrics.evidence_attribution_precision,
                    "evidence_attribution_recall": answer_metrics.evidence_attribution_recall,
                    "evidence_coverage": answer_metrics.evidence_coverage,
                    "attributed_answer_claim_count": answer_metrics.attributed_answer_claim_count,
                    "attributed_gold_claim_count": answer_metrics.attributed_gold_claim_count,
                    "invalid_attribution_count": answer_metrics.invalid_attribution_count,
                    "unsupported_claim_count": answer_metrics.unsupported_claim_count,
                    "missing_gold_claim_count": answer_metrics.missing_gold_claim_count,
                    "contradicted_claim_count": answer_metrics.contradicted_claim_count,
                    "claim_diagnostic": answer_metrics.claim_diagnostic,
                    "llm_status": "provided" if group.get("llm_answer") else "disabled",
                    "llm_error": None,
                    "answer": answer,
                    "answer_mode": answer_mode,
                    "auto_flag": classifier_auto_flag,
                    "primary_error_reason": diagnostics.primary_error_reason,
                    "secondary_error_reason": diagnostics.secondary_error_reason,
                    "diagnostic_explanation": diagnostics.explanation,
                    "manual_flag": "",
                    "manual_comment": "",
                    "gold_cpv_label": gold_label,
                    "gold_cpv_description": gold_description,
                    "prepared_candidate_count": len(normalized_candidates),
                    "prepared_retrieved_candidate_count": len(retrieved_rows_for_query),
                }
            )
        )

    trailing_columns = [
        "question_id",
        "question",
        "program_id",
        "program_name",
        "doc_id",
        "answer_scope",
        "evaluation_scope",
        "gold_answer",
        "expected_keywords",
        "expected_cpv_codes",
        "top1_predicted_cpv",
        "gold_cpv_label",
        "gold_cpv_description",
        "retrieved_chunk_ids",
        "base_retrieved_chunk_ids",
        "answer",
        "answer_mode",
        "llm_status",
        "llm_error",
        "runtime_retrieval_reason",
        "primary_error_reason",
        "secondary_error_reason",
        "diagnostic_explanation",
        "manual_flag",
        "manual_comment",
        "recommendations",
        "recommendation_count",
        "recommended_component",
        "recommendation_priority",
        "recommended_action",
        "recommendation_reason",
        "recommended_experiment",
        "needs_manual_review",
    ]
    results_df = _drop_irrelevant_classifier_columns(
        _standardize_rag_results_columns(
            _reorder_metric_columns(pd.DataFrame(result_rows), trailing_columns=trailing_columns)
        ),
        classifier_type="prepared_rag_results",
    )
    retrieval_metrics_df = _drop_irrelevant_classifier_columns(_drop_duplicate_metric_columns(_reorder_metric_columns(
        pd.DataFrame(retrieval_metric_rows),
        trailing_columns=[
            "question_id",
            "question",
            "program_id",
            "program_name",
            "doc_id",
            "answer_scope",
            "evaluation_scope",
            "expected_cpv_codes",
            "top1_predicted_cpv",
        ],
    )), classifier_type="prepared_rag_results")
    rag_results_csv = os.path.join(run_dir, "rag_results.csv")
    retrieved_csv = os.path.join(run_dir, "retrieved_chunks.csv")
    retrieval_metrics_csv = os.path.join(run_dir, "retrieval_metrics.csv")
    diagnostics_csv = os.path.join(run_dir, "diagnostics.csv")
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "prepared_source_rows.csv"), index=False)
    results_df.to_csv(rag_results_csv, index=False)
    _drop_irrelevant_classifier_columns(
        pd.DataFrame(ranking_rows),
        classifier_type="prepared_rag_results",
    ).to_csv(retrieved_csv, index=False)
    retrieval_metrics_df.to_csv(retrieval_metrics_csv, index=False)
    _drop_irrelevant_classifier_columns(
        pd.DataFrame(diagnostic_rows),
        classifier_type="prepared_rag_results",
    ).to_csv(diagnostics_csv, index=False)

    metric_rows = _clean_cpv_rows(result_rows)
    advisor_rows = _clean_cpv_rows(result_rows, keep_recommendations=True)
    aggregate_answer_metrics = _cpv_outcome_summary(metric_rows)
    aggregate_retrieval_metrics = _cpv_retrieval_summary(retrieval_metric_rows)
    aggregate_diagnostics = summarize_diagnostics(diagnostic_rows)
    aggregate_cpv_diagnostics = _summarize_cpv_diagnostics(metric_rows, top_k=effective_predicted_top_k)
    classifier_calibration = summarize_confidence_calibration(
        result_rows,
        confidence_key="prediction_confidence",
        correct_fn=lambda row: row.get("auto_flag") == "correct",
    )
    ranking_metrics = evaluate_ranked_predictions(
        evaluation_items,
        prediction_records,
        top_k=effective_predicted_top_k,
        distance_fn=cpv_structural_distance,
        precision_denominator="returned_k",
    )

    retrieval_metrics_json = os.path.join(run_dir, "retrieval_metrics_summary.json")
    diagnostics_json = os.path.join(run_dir, "diagnostics_summary.json")
    with open(retrieval_metrics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_retrieval_metrics, f, ensure_ascii=False, indent=2)
    with open(diagnostics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_diagnostics, f, ensure_ascii=False, indent=2)

    summary = {
        "experiment": classifier_label,
        "chunking_strategy": "prepared_results",
        "retriever": "prepared_rag_results",
        "chunk_size": 0,
        "chunk_overlap": 0,
        "hybrid_alpha": None,
        "top_k": effective_predicted_top_k,
        "reranker": {"enabled": False, "type": None, "top_n": 0, "weight": 0.0},
        "n_chunks": len(catalog_chunks),
        "n_questions": len(grouped),
        "n_correct": int((results_df["auto_flag"] == "correct").sum()),
        "n_incorrect": int((results_df["auto_flag"] == "incorrect").sum()),
        "answer_metrics": aggregate_answer_metrics,
        "retrieval_metrics": aggregate_retrieval_metrics,
        "diagnostics": aggregate_diagnostics,
        "llm": {
            "enabled": any(bool(group.get("llm_answer")) for group in grouped.values()),
            "model": "provided",
            "answer_generation": False,
        },
        "classifier": {
            "type": "prepared_rag_results",
            "label": classifier_label,
            "source_path": prepared_results_path,
            "retrieval_top_k": effective_retrieved_top_k,
            "ranking_metrics": ranking_metrics,
            "calibration": classifier_calibration,
            "cpv_diagnostics": aggregate_cpv_diagnostics,
            "input_contract": {
                **_input_contract_fields(),
                "current_columns": list(rows[0].keys()) if rows else [],
                "current_top_k_columns": [
                    "Predicted answer #1",
                    "Score #1",
                    "Predicted answer #2",
                    "Score #2",
                    "Predicted answer #3",
                    "Score #3",
                ],
                "top_k_layout": "Multiple rows with the same ID are treated as ranked candidates for one query; wide ranked columns with #1/#2/#3 are also supported.",
            },
            "recommended_next_steps": _classifier_next_steps("prepared_rag_results"),
        },
        "visualization": {
            "enabled": create_visualization or create_showcase,
            "strategy_score_profile_svg": None,
            "strategy_chunk_alignment_svg": None,
        },
        "showcase": {
            "enabled": False,
            "score_profile_svg": None,
            "chunk_alignment_svg": None,
            "metric_overview_svg": None,
            "diagnostics_svg": None,
            "showcase_md": None,
            "improvement_summary": None,
        },
        "outputs": {
            "rag_results_csv": rag_results_csv,
            "retrieved_chunks_csv": retrieved_csv,
            "retrieval_metrics_csv": retrieval_metrics_csv,
            "retrieval_metrics_summary_json": retrieval_metrics_json,
            "diagnostics_csv": diagnostics_csv,
            "diagnostics_summary_json": diagnostics_json,
            "strategy_score_profile_svg": None,
            "strategy_chunk_alignment_svg": None,
            "strategy_metric_overview_svg": None,
            "strategy_diagnostics_svg": None,
            "strategy_showcase_md": None,
            "prepared_source_rows_csv": os.path.join(run_dir, "prepared_source_rows.csv"),
        },
    }
    quality_advisor = build_run_advisor(summary, advisor_rows)
    quality_advisor_json = os.path.join(run_dir, "quality_advisor.json")
    with open(quality_advisor_json, "w", encoding="utf-8") as f:
        json.dump(quality_advisor, f, ensure_ascii=False, indent=2)
    quality_report_md = write_quality_report(
        os.path.join(run_dir, "quality_report.md"),
        quality_advisor,
        summary,
    )
    summary["advisor"] = quality_advisor
    summary["outputs"]["quality_advisor_json"] = quality_advisor_json
    summary["outputs"]["quality_report_md"] = quality_report_md

    if create_visualization or create_showcase:
        showcase_bundle = write_classifier_showcase_bundle(
            summary=summary,
            ranking_rows=ranking_rows,
            experiment_dir=run_dir,
        )
        _apply_showcase_bundle(summary, showcase_bundle)

    summary_json = os.path.join(run_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["outputs"]["summary_json"] = summary_json
    return summary


def evaluate_api_ted_cpv_classifier(
    *,
    api_url: str,
    auth_token_env: str,
    extra_headers: Dict[str, object],
    timeout_seconds: float,
    ted_corpus_export_path: str,
    queries_path: str,
    top_k: int,
    classifier_label: str,
    run_dir: str,
    create_visualization: bool,
    create_showcase: bool,
) -> Dict[str, object]:
    import pandas as pd

    catalog = load_cpv_catalog_from_ted_corpus_export(ted_corpus_export_path)
    queries = load_queries(queries_path)
    catalog_chunks = build_cpv_chunks(catalog, use_examples=True)
    parent_lookup = build_parent_lookup(catalog)
    catalog_by_code = {record.code: record for record in catalog}
    label_by_code = {record.code: record.label for record in catalog}
    description_by_code = {record.code: record.description for record in catalog}

    prediction_records: List[PredictionRecord] = []
    ranking_rows: List[Dict[str, object]] = []
    result_rows: List[Dict[str, object]] = []
    answer_metric_rows: List[Dict[str, object]] = []
    retrieval_metric_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    explanation_count = 0
    request_latencies_ms: List[float] = []

    for query in queries:
        request_payload = {
            "id": query.id,
            "query": query.query,
            "top_k": top_k,
        }
        started = time.perf_counter()
        response_payload = _post_classifier_request(
            api_url=api_url,
            payload=request_payload,
            auth_token_env=auth_token_env,
            extra_headers=extra_headers,
            timeout_seconds=timeout_seconds,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        request_latencies_ms.append(latency_ms)

        response_id = str(response_payload.get("id", query.id))
        response_query = str(
            response_payload.get("query", response_payload.get("user_query", query.query))
        )
        predictions_raw = response_payload.get("predictions", response_payload.get("top_k_answers", []))
        if not isinstance(predictions_raw, list):
            raise RuntimeError(f"Classifier API response for {query.id} must contain a list in `predictions`.")

        normalized_candidates: List[RankedCandidate] = []
        retrieved_rows_for_query: List[Dict[str, object]] = []
        for rank, candidate in enumerate(predictions_raw[:top_k], start=1):
            if not isinstance(candidate, dict):
                continue
            label = _normalize_prediction_label(candidate)
            if not label:
                continue
            score = _normalize_prediction_score(candidate, fallback=max(0.0, 1.0 - 0.05 * (rank - 1)))
            normalized_candidates.append(
                RankedCandidate(
                    label=label,
                    score=score,
                    metadata={
                        k: v
                        for k, v in candidate.items()
                        if k not in _classifier_payload_excluded_keys()
                        and _normalize_header(k) not in _classifier_payload_excluded_keys()
                    },
                )
            )
            retrieved_rows_for_query.append(
                _catalog_row_from_prediction(
                    predicted_label=label,
                    score=score,
                    rank=rank,
                    query_text=response_query,
                    catalog_by_code=catalog_by_code,
                )
            )

        explanation = str(response_payload.get("explanation", "") or "").strip()
        if explanation:
            explanation_count += 1

        prediction_records.append(
            PredictionRecord(
                id=response_id,
                candidates=normalized_candidates,
                metadata={
                    "query": response_query,
                    "explanation": explanation,
                    "latency_ms": latency_ms,
                },
            )
        )

        gold_label = label_by_code.get(query.gold_cpv_code, query.gold_cpv_code)
        gold_description = description_by_code.get(query.gold_cpv_code, "")
        item = {
            "id": query.id,
            "question": query.query,
            "gold_answer": f"{query.gold_cpv_code} {gold_label}".strip(),
            "expected_keywords": [query.gold_cpv_code, gold_label],
            "doc_id": query.gold_cpv_code,
        }

        answer = ""
        if normalized_candidates:
            top1 = normalized_candidates[0]
            top1_label = label_by_code.get(top1.label, top1.label)
            answer = f"{top1.label} {top1_label}".strip()

        runtime_retrieval_result = runtime_retrieval_evaluation(
            question=query.query,
            retrieved=retrieved_rows_for_query,
        )
        answer_metrics = evaluate_answer_metrics(
            item,
            answer,
            retrieved_rows_for_query,
            runtime_retrieval_result=runtime_retrieval_result,
        )
        retrieval_metrics = evaluate_retrieval_metrics(
            item=item,
            retrieved=retrieved_rows_for_query,
            candidate_chunks=catalog_chunks,
            k=min(top_k, len(catalog)),
        )
        diagnostics = diagnose_failure(
            answer_metrics=answer_metrics,
            retrieval_metrics=retrieval_metrics,
            llm_status="disabled",
            answer_mode="api_classifier_label",
        )
        classification_metrics = _cpv_classification_metrics(
            expected_codes=[query.gold_cpv_code],
            ranked_labels=[candidate.label for candidate in normalized_candidates[:top_k]],
        )
        classifier_auto_flag = _cpv_auto_flag(classification_metrics)
        diagnostics = _cpv_diagnostics(classification_metrics, retrieval_metrics)
        prediction_confidence = (
            _normalized_prediction_confidence(normalized_candidates[0].score)
            if normalized_candidates and normalized_candidates[0].score is not None
            else None
        )
        cpv_rank_diagnostics = _cpv_rank_diagnostics(
            expected_codes=[query.gold_cpv_code],
            ranked_labels=[candidate.label for candidate in normalized_candidates[:top_k]],
            scores=[
                float(candidate.score)
                for candidate in normalized_candidates[:top_k]
                if candidate.score is not None
            ],
            query_text=query.query,
            top_k=top_k,
            prediction_confidence=prediction_confidence,
        )

        for rank, row in enumerate(retrieved_rows_for_query, start=1):
            relevance_grade = retrieval_relevance_grade(item, row)
            ranking_rows.append(
                {
                    "question_id": query.id,
                    "question": query.query,
                    "rank": rank,
                    "auto_flag": classifier_auto_flag,
                    "retriever": "api_classifier",
                    "chunk_id": row["chunk_id"],
                    "score": row["score"],
                    "doc_id": row["doc_id"],
                    "section_id": row["section_id"],
                    "title": row["title"],
                    "chunking_strategy": row["chunking_strategy"],
                    "source_type": row["source_type"],
                    "text": row["text"],
                    "relevance_grade": relevance_grade,
                    "is_relevant": is_relevant_grade(relevance_grade),
                }
            )

        answer_metric_rows.append(
            {
                "question_id": query.id,
                "question": query.query,
                **classification_metrics,
                "answer_accuracy_label": classifier_auto_flag,
                "expected_answerable": answer_metrics.expected_answerable,
                "abstained": answer_metrics.abstained,
                "abstention_correct": answer_metrics.abstention_correct,
                "over_answered": answer_metrics.over_answered,
                "false_refusal": answer_metrics.false_refusal,
                "llm_used": False,
                "llm_status": "disabled",
                "llm_error": None,
                "gold_answer_overlap": answer_metrics.gold_answer_overlap,
                "answer_gold_support": answer_metrics.answer_gold_support,
                "proxy_faithfulness": answer_metrics.proxy_faithfulness,
                "proxy_context_relevance": answer_metrics.proxy_context_relevance,
                "answer_has_gold_substring": answer_metrics.answer_has_gold_substring,
                "answerability_confidence": answer_metrics.answerability_confidence,
                "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
                "factual_correctness_precision": answer_metrics.factual_correctness_precision,
                "factual_correctness_recall": answer_metrics.factual_correctness_recall,
                "factual_correctness_f1": answer_metrics.factual_correctness_f1,
                "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
                "noise_sensitivity_irrelevant": answer_metrics.noise_sensitivity_irrelevant,
                "context_entities_recall": answer_metrics.context_entities_recall,
                "answer_entity_precision": answer_metrics.answer_entity_precision,
                "context_claim_recall": answer_metrics.context_claim_recall,
                "answer_claim_precision": answer_metrics.answer_claim_precision,
                "answer_claim_recall": answer_metrics.answer_claim_recall,
                "answer_claim_f1": answer_metrics.answer_claim_f1,
                "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
                "evidence_attribution_f1": answer_metrics.evidence_attribution_f1,
                "evidence_attribution_precision": answer_metrics.evidence_attribution_precision,
                "evidence_attribution_recall": answer_metrics.evidence_attribution_recall,
                "evidence_coverage": answer_metrics.evidence_coverage,
                "attributed_answer_claim_count": answer_metrics.attributed_answer_claim_count,
                "attributed_gold_claim_count": answer_metrics.attributed_gold_claim_count,
                "invalid_attribution_count": answer_metrics.invalid_attribution_count,
                "unsupported_claim_count": answer_metrics.unsupported_claim_count,
                "missing_gold_claim_count": answer_metrics.missing_gold_claim_count,
                "contradicted_claim_count": answer_metrics.contradicted_claim_count,
                "claim_diagnostic": answer_metrics.claim_diagnostic,
            }
        )
        retrieval_metric_rows.append(
            {
                "question_id": query.id,
                "question": query.query,
                **classification_metrics,
                "program_id": "cpv",
                "program_name": "CPV",
                "doc_id": query.gold_cpv_code,
                "answer_scope": json.dumps({}, ensure_ascii=False),
                "evaluation_scope": json.dumps({"doc_id": [query.gold_cpv_code]}, ensure_ascii=False),
                "mrr_at_k": retrieval_metrics["mrr_at_k"],
                "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
                "recall_at_k": retrieval_metrics["recall_at_k"],
                "first_relevant_rank": retrieval_metrics["first_relevant_rank"],
                "n_relevant_chunks": retrieval_metrics["n_relevant_chunks"],
                "n_retrieved_relevant_chunks": retrieval_metrics["n_retrieved_relevant_chunks"],
                "target_doc_retrieved_at_k": retrieval_metrics["target_doc_retrieved_at_k"],
                "first_target_doc_rank": retrieval_metrics["first_target_doc_rank"],
                "n_retrieved_target_doc_chunks": retrieval_metrics["n_retrieved_target_doc_chunks"],
            }
        )
        diagnostic_rows.append(
            {
                "question_id": query.id,
                "question": query.query,
                "program_id": "cpv",
                "program_name": "CPV",
                "doc_id": query.gold_cpv_code,
                "answer_scope": json.dumps({}, ensure_ascii=False),
                "evaluation_scope": json.dumps({"doc_id": [query.gold_cpv_code]}, ensure_ascii=False),
                "primary_error_reason": diagnostics.primary_error_reason,
                "secondary_error_reason": diagnostics.secondary_error_reason,
                **cpv_rank_diagnostics,
                "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
                "context_claim_recall": answer_metrics.context_claim_recall,
                "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
                "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
                "context_entities_recall": answer_metrics.context_entities_recall,
                "explanation": diagnostics.explanation,
            }
        )

        result_rows.append(
            apply_question_recommendations(
                {
                    "question_id": query.id,
                    "question": query.query,
                    "retrieval_query": retrieval_query,
                    "query_augmentation_mode": resolved_query_augmentation,
                    "query_augmentation_status": query_augmentation_result.status,
                    "program_id": "cpv",
                    "program_name": "CPV",
                    "doc_id": query.gold_cpv_code,
                    "answer_scope": json.dumps({}, ensure_ascii=False),
                    "evaluation_scope": json.dumps({"doc_id": [query.gold_cpv_code]}, ensure_ascii=False),
                    "gold_answer": item["gold_answer"],
                    "expected_keywords": json.dumps(item["expected_keywords"], ensure_ascii=False),
                    **classification_metrics,
                    **cpv_rank_diagnostics,
                    **_all_retrieval_metric_fields(retrieval_metrics),
                    **_all_answer_metric_fields(answer_metrics),
                    "expected_answerable": answer_metrics.expected_answerable,
                    "abstained": answer_metrics.abstained,
                    "abstention_correct": answer_metrics.abstention_correct,
                    "over_answered": answer_metrics.over_answered,
                    "false_refusal": answer_metrics.false_refusal,
                    "prediction_confidence": prediction_confidence,
                    "retrieved_chunk_ids": json.dumps([row["chunk_id"] for row in retrieved_rows_for_query], ensure_ascii=False),
                    "retrieved_chunks": _retrieved_chunks_payload(retrieved_rows_for_query),
                    "base_retrieved_chunk_ids": json.dumps([row["chunk_id"] for row in retrieved_rows_for_query], ensure_ascii=False),
                    "mrr_at_k": retrieval_metrics["mrr_at_k"],
                    "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
                    "recall_at_k": retrieval_metrics["recall_at_k"],
                    "target_doc_retrieved_at_k": retrieval_metrics["target_doc_retrieved_at_k"],
                    "first_target_doc_rank": retrieval_metrics["first_target_doc_rank"],
                    "n_retrieved_target_doc_chunks": retrieval_metrics["n_retrieved_target_doc_chunks"],
                    "gold_answer_overlap": answer_metrics.gold_answer_overlap,
                    "answer_gold_support": answer_metrics.answer_gold_support,
                    "proxy_faithfulness": answer_metrics.proxy_faithfulness,
                    "proxy_context_relevance": answer_metrics.proxy_context_relevance,
                    "answerability_confidence": answer_metrics.answerability_confidence,
                    "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
                    "runtime_retrieval_action": answer_metrics.runtime_retrieval_action,
                    "runtime_retrieval_reason": answer_metrics.runtime_retrieval_reason,
                    "factual_correctness_precision": answer_metrics.factual_correctness_precision,
                    "factual_correctness_recall": answer_metrics.factual_correctness_recall,
                    "factual_correctness_f1": answer_metrics.factual_correctness_f1,
                    "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
                    "noise_sensitivity_irrelevant": answer_metrics.noise_sensitivity_irrelevant,
                    "context_entities_recall": answer_metrics.context_entities_recall,
                    "answer_entity_precision": answer_metrics.answer_entity_precision,
                    "context_claim_recall": answer_metrics.context_claim_recall,
                    "answer_claim_precision": answer_metrics.answer_claim_precision,
                    "answer_claim_recall": answer_metrics.answer_claim_recall,
                    "answer_claim_f1": answer_metrics.answer_claim_f1,
                    "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
                    "evidence_attribution_f1": answer_metrics.evidence_attribution_f1,
                    "claim_diagnostic": answer_metrics.claim_diagnostic,
                    "llm_status": "disabled",
                    "llm_error": None,
                    "answer": answer,
                    "answer_mode": "api_classifier_label",
                    "classifier_explanation": explanation,
                    "auto_flag": classifier_auto_flag,
                    "primary_error_reason": diagnostics.primary_error_reason,
                    "secondary_error_reason": diagnostics.secondary_error_reason,
                    "diagnostic_explanation": diagnostics.explanation,
                    "manual_flag": "",
                    "manual_comment": "",
                    "gold_cpv_label": gold_label,
                    "gold_cpv_description": gold_description,
                    "api_explanation": explanation,
                    "api_latency_ms": latency_ms,
                }
            )
        )

    results_df = _drop_irrelevant_classifier_columns(
        _standardize_rag_results_columns(pd.DataFrame(result_rows)),
        classifier_type="api_classifier",
    )

    rag_results_csv = os.path.join(run_dir, "rag_results.csv")
    retrieved_csv = os.path.join(run_dir, "retrieved_chunks.csv")
    retrieval_metrics_csv = os.path.join(run_dir, "retrieval_metrics.csv")
    diagnostics_csv = os.path.join(run_dir, "diagnostics.csv")
    results_df.to_csv(rag_results_csv, index=False)
    _drop_irrelevant_classifier_columns(
        pd.DataFrame(ranking_rows),
        classifier_type="api_classifier",
    ).to_csv(retrieved_csv, index=False)
    _drop_irrelevant_classifier_columns(
        _drop_duplicate_metric_columns(pd.DataFrame(retrieval_metric_rows)),
        classifier_type="api_classifier",
    ).to_csv(retrieval_metrics_csv, index=False)
    _drop_irrelevant_classifier_columns(
        pd.DataFrame(diagnostic_rows),
        classifier_type="api_classifier",
    ).to_csv(diagnostics_csv, index=False)

    metric_rows = _clean_cpv_rows(result_rows)
    advisor_rows = _clean_cpv_rows(result_rows, keep_recommendations=True)
    aggregate_answer_metrics = _cpv_outcome_summary(metric_rows)
    aggregate_retrieval_metrics = _cpv_retrieval_summary(retrieval_metric_rows)
    aggregate_diagnostics = summarize_diagnostics(diagnostic_rows)
    aggregate_cpv_diagnostics = _summarize_cpv_diagnostics(metric_rows, top_k=top_k)
    classifier_calibration = summarize_confidence_calibration(
        result_rows,
        confidence_key="prediction_confidence",
        correct_fn=lambda row: row.get("auto_flag") == "correct",
    )

    retrieval_metrics_json = os.path.join(run_dir, "retrieval_metrics_summary.json")
    diagnostics_json = os.path.join(run_dir, "diagnostics_summary.json")
    with open(retrieval_metrics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_retrieval_metrics, f, ensure_ascii=False, indent=2)
    with open(diagnostics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_diagnostics, f, ensure_ascii=False, indent=2)

    ranking_metrics = evaluate_ranked_predictions(
        build_items_from_cpv_queries(queries_path),
        prediction_records,
        top_k=top_k,
        distance_fn=cpv_structural_distance,
    )
    top1_top2_margins = []
    for record in prediction_records:
        if len(record.candidates) < 2:
            continue
        first = record.candidates[0].score
        second = record.candidates[1].score
        if first is None or second is None:
            continue
        top1_top2_margins.append(float(first) - float(second))

    summary = {
        "experiment": classifier_label,
        "chunking_strategy": "api_classifier",
        "retriever": "api_classifier",
        "chunk_size": 0,
        "chunk_overlap": 0,
        "hybrid_alpha": None,
        "top_k": top_k,
        "reranker": {
            "enabled": False,
            "type": None,
            "top_n": 0,
            "weight": 0.0,
        },
        "n_chunks": len(catalog),
        "n_questions": len(queries),
        "n_correct": int((results_df["auto_flag"] == "correct").sum()),
        "n_incorrect": int((results_df["auto_flag"] == "incorrect").sum()),
        "answer_metrics": aggregate_answer_metrics,
        "retrieval_metrics": aggregate_retrieval_metrics,
        "diagnostics": aggregate_diagnostics,
        "llm": {
            "enabled": False,
            "model": None,
            "answer_generation": False,
        },
        "classifier": {
            "type": "api_classifier",
            "label": classifier_label,
            "api_url": api_url,
            "avg_top1_top2_margin": (
                sum(top1_top2_margins) / len(top1_top2_margins) if top1_top2_margins else None
            ),
            "explanation_coverage": explanation_count / max(len(queries), 1),
            "mean_latency_ms": (
                sum(request_latencies_ms) / len(request_latencies_ms) if request_latencies_ms else None
            ),
            "ranking_metrics": ranking_metrics,
            "calibration": classifier_calibration,
            "cpv_diagnostics": aggregate_cpv_diagnostics,
            "request_contract": {
                "required": ["id", "query", "top_k"],
                "description": "The evaluator sends one query per request and expects a ranked candidate list back.",
            },
            "response_contract": {
                **_input_contract_fields(),
                "required": ["id", "query", "predictions"],
                "prediction_list_aliases": ["predictions", "top_k_answers"],
                "optional": ["answer", "explanation", "metadata", "latency_ms"],
            },
            "recommended_next_steps": _classifier_next_steps("api_classifier"),
        },
        "visualization": {
            "enabled": create_visualization or create_showcase,
            "strategy_score_profile_svg": None,
            "strategy_chunk_alignment_svg": None,
        },
        "showcase": {
            "enabled": False,
            "score_profile_svg": None,
            "chunk_alignment_svg": None,
            "metric_overview_svg": None,
            "diagnostics_svg": None,
            "showcase_md": None,
            "improvement_summary": None,
        },
        "outputs": {
            "rag_results_csv": rag_results_csv,
            "retrieved_chunks_csv": retrieved_csv,
            "retrieval_metrics_csv": retrieval_metrics_csv,
            "retrieval_metrics_summary_json": retrieval_metrics_json,
            "diagnostics_csv": diagnostics_csv,
            "diagnostics_summary_json": diagnostics_json,
            "strategy_score_profile_svg": None,
            "strategy_chunk_alignment_svg": None,
            "strategy_metric_overview_svg": None,
            "strategy_diagnostics_svg": None,
            "strategy_showcase_md": None,
        },
    }
    quality_advisor = build_run_advisor(summary, advisor_rows)
    quality_advisor_json = os.path.join(run_dir, "quality_advisor.json")
    with open(quality_advisor_json, "w", encoding="utf-8") as f:
        json.dump(quality_advisor, f, ensure_ascii=False, indent=2)
    quality_report_md = write_quality_report(
        os.path.join(run_dir, "quality_report.md"),
        quality_advisor,
        summary,
    )
    summary["advisor"] = quality_advisor
    summary["outputs"]["quality_advisor_json"] = quality_advisor_json
    summary["outputs"]["quality_report_md"] = quality_report_md
    if create_visualization or create_showcase:
        showcase_bundle = write_classifier_showcase_bundle(
            summary=summary,
            ranking_rows=ranking_rows,
            experiment_dir=run_dir,
        )
        _apply_showcase_bundle(summary, showcase_bundle)

    summary_json = os.path.join(run_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["outputs"]["summary_json"] = summary_json
    return summary
