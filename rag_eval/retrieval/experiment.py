from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rag_eval.evaluation.advisor import apply_question_recommendations, build_run_advisor, write_quality_report
from rag_eval.evaluation.evidence_graph import build_document_evidence_graph_summary
from rag_eval.retrieval.chunking import build_chunks
from rag_eval.retrieval.kg import (
    ablate_kg_graph_edges,
    build_kg_supervision_terms,
    build_knowledge_graph,
    evaluate_answer_kg_path_grounding,
    evaluate_kg_for_question,
    evaluate_kg_retrieval_diagnostics,
    graph_quality_diagnostics,
    graph_augmented_retrieval,
    summarize_kg_metrics,
)
from rag_eval.evaluation.judge import judge_claims_with_llm
from rag_eval.evaluation.llm import (
    LLM_QUERY_AUGMENTATION_MODES,
    augment_query_with_llm,
    critique_and_revise_answer_with_llm,
    generate_answer_with_llm,
    rewrite_query_for_retry,
)
from rag_eval.evaluation.metrics import (
    diagnose_failure,
    evaluate_answer_metrics,
    evaluate_retrieval_metrics,
    is_relevant_grade,
    keyword_extractive_answer,
    retrieval_relevance_grade,
    runtime_retrieval_evaluation,
    summarize_answer_metrics,
    summarize_confidence_calibration,
    summarize_diagnostics,
    summarize_retrieval_metrics,
)
from rag_eval.core.models import LLMCallResult, LLMConfig, Paragraph, Section
from rag_eval.retrieval.engines import build_retriever_with_backend, rerank_with_lexical_signal, retrieve_top_k
from rag_eval.core.text_utils import metadata_value_matches
from rag_eval.reporting.visualization import (
    write_chunk_relevance_comparison_svg,
    write_strategy_score_profile_svg,
    write_strategy_showcase_bundle,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_csv_list(raw_value: str) -> List[int]:
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_float_csv_list(raw_value: str | Sequence[float] | None) -> List[float]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        values = [item.strip() for item in raw_value.split(",") if item.strip()]
    else:
        values = [str(item) for item in raw_value]
    return [min(max(float(value), 0.0), 1.0) for value in values]


def infer_question_type(item: Dict) -> str:
    explicit = str(
        item.get("question_type")
        or item.get("task_type")
        or item.get("query_type")
        or ""
    ).strip()
    if explicit:
        return explicit
    question = str(item.get("question", "")).casefold()
    required_triples = list(item.get("must_have_triples", []))
    if item.get("cpv_code") or item.get("expected_cpv") or item.get("classification_type"):
        return "taxonomy"
    if any(term in question for term in ["summarize", "summary", "überblick", "zusammenfassung", "compare all", "across"]):
        return "summary_global"
    relation_terms = ["requires", "requirement", "depends", "deadline", "relationship", "relation", "voraussetz", "frist", "abhängig"]
    if any(term in question for term in relation_terms):
        return "relation"
    if len(required_triples) >= 2 or any(term in question for term in [" and ", " und ", "both", "mehrere", "compare"]):
        return "multi_hop"
    return "single_hop"


def summarize_rows_by_question_type(rows: Sequence[Dict], metric_keys: Sequence[str]) -> Dict[str, Dict[str, object]]:
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        question_type = str(row.get("question_type") or "unknown")
        groups.setdefault(question_type, []).append(row)
    summary: Dict[str, Dict[str, object]] = {}
    for question_type, group_rows in sorted(groups.items()):
        type_summary: Dict[str, object] = {"n": len(group_rows)}
        for key in metric_keys:
            values = []
            for row in group_rows:
                value = row.get(key)
                if value is None or value == "":
                    continue
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
            type_summary[f"mean_{key}"] = (sum(values) / len(values)) if values else None
        summary[question_type] = type_summary
    return summary


def list_field(value: object) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def infer_program_names_from_doc_paths(paths: Sequence[str]) -> List[str]:
    program_names: List[str] = []
    for path in paths:
        normalized = str(path or "").replace("\\", "/").strip("/")
        if not normalized:
            continue
        program_name = normalized.split("/", 1)[0].strip()
        if program_name and program_name not in program_names:
            program_names.append(program_name)
    return program_names


def _normalized_path(value: object) -> str:
    return str(value or "").replace("\\", "/").strip()


def _doc_type(doc_path: str) -> str:
    filename = Path(_normalized_path(doc_path)).name.casefold()
    if "berichtigung" in filename:
        return "correction"
    if "auslauf" in filename:
        return "phase_out"
    if any(token in filename for token in ["aenderung", "änderung", "nachtrag", "supplement"]):
        return "amendment"
    return "base"


def _question_prefers_special_regulation(question: str) -> str | None:
    normalized = str(question or "").casefold()
    if any(token in normalized for token in ["berichtigung", "corrected", "correction", "erratum"]):
        return "correction"
    if any(token in normalized for token in ["auslauf", "übergang", "uebergang", "transition", "phasing out"]):
        return "phase_out"
    if any(token in normalized for token in ["änderung", "aenderung", "amendment", "change order", "supplement"]):
        return "amendment"
    return None


def _doc_type_priority(doc_path: str, question: str) -> int:
    preferred_type = _question_prefers_special_regulation(question)
    doc_type = _doc_type(doc_path)
    if preferred_type is not None:
        return 0 if doc_type == preferred_type else 1
    priority = {"base": 0, "amendment": 1, "correction": 2, "phase_out": 3}
    return priority.get(doc_type, 4)


def _preferred_doc_paths(doc_paths: Sequence[str], question: str) -> List[str]:
    normalized_paths = sorted({_normalized_path(path) for path in doc_paths if _normalized_path(path)})
    if not normalized_paths:
        return []
    best_priority = min(_doc_type_priority(path, question) for path in normalized_paths)
    return [path for path in normalized_paths if _doc_type_priority(path, question) == best_priority]


def _question_year(item: Dict) -> int | None:
    raw_year = item.get("year")
    if raw_year is None or raw_year == "":
        return None
    try:
        year = int(raw_year)
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None


def _doc_year(doc_path: str) -> int | None:
    filename = Path(str(doc_path or "")).name
    matches = re.findall(r"20\d{2}", filename)
    if not matches:
        return None
    try:
        return int(matches[0])
    except ValueError:
        return None


def _valid_year(value: str) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None


def _doc_text_year(text: str) -> int | None:
    normalized = re.sub(r"\s+", " ", str(text or ""))
    if not normalized:
        return None

    primary_patterns = [
        r"\b(?:vom|from|dated|as of)\s+\d{1,2}\.?\s+[A-Za-zÄÖÜäöüß]+\s+(20\d{2})\b",
        r"\b(?:herausgegeben|published|issued)\s+(?:am|on)\s+\d{1,2}\.?\s+[A-Za-zÄÖÜäöüß]+\s+(20\d{2})\b",
        r"\b(?:amtliche\s+mitteilung|official\s+notice)\s+(?:nr\.?|no\.?)\s+\d+\s*/\s*(20\d{2})\b",
        r"\b(?:ordnung|satzung|regulation|statute)\b.{0,120}\b(?:vom|from|dated)\s+\d{1,2}\.?\s+[A-Za-zÄÖÜäöüß]+\s+(20\d{2})\b",
    ]
    for pattern in primary_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            year = _valid_year(match.group(1))
            if year is not None:
                return year

    first_page_like_text = normalized[:2500]
    first_years = [
        year
        for year in (_valid_year(match.group(0)) for match in re.finditer(r"\b20\d{2}\b", first_page_like_text))
        if year is not None
    ]
    if first_years:
        return first_years[0]

    return None


def _chunk_sort_key(chunk: Dict) -> tuple[int, int, str]:
    section_id = str(chunk.get("section_id", ""))
    source_id = str(chunk.get("source_id", ""))
    chunk_id = str(chunk.get("chunk_id", ""))
    preamble_rank = 0 if section_id == "PREAMBLE" or "PREAMBLE" in chunk_id else 1
    section_match = re.search(r"\|s(\d+)\|", source_id) or re.search(r"\|s(\d+)\|", chunk_id)
    section_pos = int(section_match.group(1)) if section_match else 9999
    return (preamble_rank, section_pos, chunk_id)


def _doc_year_from_chunks(doc_path: str, chunks: Sequence[Dict]) -> int | None:
    doc_chunks = [
        chunk
        for chunk in chunks
        if str(chunk.get("doc_path", "")).strip() == doc_path
    ]
    if not doc_chunks:
        return _doc_year(doc_path)

    ordered_chunks = sorted(doc_chunks, key=_chunk_sort_key)
    head_text = "\n\n".join(
        "\n".join(
            part
            for part in [
                str(chunk.get("section_id", "")),
                str(chunk.get("title", "")),
                str(chunk.get("text", "")),
            ]
            if part.strip()
        )
        for chunk in ordered_chunks[:3]
    )
    return _doc_text_year(head_text) or _doc_year(doc_path)


def _select_doc_paths_for_year(
    *,
    item: Dict,
    chunks: Sequence[Dict],
) -> List[str]:
    explicit_doc_paths = list_field(item.get("doc_path")) + list_field(item.get("doc_paths"))
    question = str(item.get("question", ""))
    year = _question_year(item)

    if year is None:
        return _preferred_doc_paths(explicit_doc_paths, question)

    program_names = list_field(item.get("program_name")) + list_field(item.get("program_names"))
    if not program_names:
        program_names = infer_program_names_from_doc_paths(explicit_doc_paths)

    program_chunks = [
        chunk
        for chunk in chunks
        if not program_names
        or any(metadata_value_matches(chunk.get("program_name", ""), program_name) for program_name in program_names)
    ]
    doc_years: Dict[str, int | None] = {}
    for chunk in program_chunks:
        doc_path = str(chunk.get("doc_path", "")).strip()
        if not doc_path or doc_path in doc_years:
            continue
        doc_years[doc_path] = _doc_year_from_chunks(doc_path, program_chunks)

    if not doc_years:
        return _preferred_doc_paths(explicit_doc_paths, question)

    # Students stay on the base MPO/BPO version that was in force at the
    # time of enrollment. Earlier amendments that were already in force by
    # that enrollment year remain relevant to the cohort, while later ones
    # should not override it unless the question explicitly asks about a
    # special regulation.
    eligible_base_docs = {
        doc_path: doc_year
        for doc_path, doc_year in doc_years.items()
        if _doc_type(doc_path) == "base" and doc_year is not None and doc_year <= year
    }
    if eligible_base_docs:
        target_base_year = max(eligible_base_docs.values())
        selected_base_docs = {
            doc_path
            for doc_path, doc_year in eligible_base_docs.items()
            if doc_year == target_base_year
        }
        if selected_base_docs:
            selected_doc_paths = set(selected_base_docs)
            for doc_path, doc_year in doc_years.items():
                doc_type = _doc_type(doc_path)
                if doc_type not in {"amendment", "correction"}:
                    continue
                if doc_year is None or doc_year > year:
                    continue
                if doc_year < target_base_year:
                    continue
                selected_doc_paths.add(doc_path)
            return sorted(selected_doc_paths)

    eligible = {
        doc_path: doc_year
        for doc_path, doc_year in doc_years.items()
        if doc_year is not None and doc_year <= year
    }
    selected_pool = eligible if eligible else {doc_path: doc_year for doc_path, doc_year in doc_years.items() if doc_year is not None}
    if not selected_pool:
        return _preferred_doc_paths(explicit_doc_paths, question)

    if eligible:
        target_year = max(selected_pool.values())
    else:
        target_year = min(selected_pool.values())

    selected_doc_paths = [doc_path for doc_path, doc_year in selected_pool.items() if doc_year == target_year]
    if selected_doc_paths:
        preferred_doc_paths = _preferred_doc_paths(selected_doc_paths, question)
        return preferred_doc_paths or selected_doc_paths

    return _preferred_doc_paths(explicit_doc_paths, question)


def question_metadata_filter(
    item: Dict,
    *,
    chunks: Sequence[Dict],
    include_document: bool,
    domain_specific_logic: bool = True,
) -> Dict[str, object]:
    metadata_filter: Dict[str, object] = {}
    doc_paths = list_field(item.get("doc_path")) + list_field(item.get("doc_paths"))
    program_names = list_field(item.get("program_name")) + list_field(item.get("program_names"))
    if domain_specific_logic and not program_names:
        # Fall back to the document folder so retrieval stays inside the
        # program directory even when a question row omitted program_name.
        program_names = infer_program_names_from_doc_paths(doc_paths)

    fields = [
        ("program_id", "program_ids"),
    ]
    if program_names:
        metadata_filter["program_name"] = program_names
    if domain_specific_logic:
        selected_doc_paths = _select_doc_paths_for_year(item=item, chunks=chunks)
    else:
        selected_doc_paths = doc_paths
    if not selected_doc_paths:
        selected_doc_paths = doc_paths
    if selected_doc_paths:
        metadata_filter["doc_path"] = selected_doc_paths
    for singular, plural in fields:
        values = list_field(item.get(singular)) + list_field(item.get(plural))
        if values:
            metadata_filter[singular] = values
    return metadata_filter


def filter_chunks_by_metadata(chunks: Sequence[Dict], metadata_filter: Dict[str, object]) -> List[Dict]:
    if not metadata_filter:
        return list(chunks)
    filtered: List[Dict] = []
    for chunk in chunks:
        matches = True
        for key, expected in metadata_filter.items():
            values = expected if isinstance(expected, (list, tuple, set)) else [expected]
            if not any(metadata_value_matches(chunk.get(key, ""), value) for value in values):
                matches = False
                break
        if matches:
            filtered.append(chunk)
    return filtered


def assemble_context_rows(
    rows: Sequence[Dict],
    *,
    mode: str,
    max_chunks: int | None,
    max_chars: int | None,
    kg_retrieval: Dict[str, object] | None = None,
) -> List[Dict]:
    selected = list(rows)
    if mode == "kg_first":
        selected.sort(
            key=lambda row: (
                0 if row.get("retrieval_source") == "graph" else 1,
                -float(row.get("kg_graph_score", 0.0)),
                -float(row.get("score", 0.0)),
            )
        )
    elif mode == "group_by_doc":
        selected.sort(key=lambda row: (str(row.get("doc_id", "")), str(row.get("section_id", "")), -float(row.get("score", 0.0))))
    elif mode == "dedupe_section":
        seen = set()
        deduped = []
        for row in selected:
            key = (row.get("doc_id"), row.get("section_id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        selected = deduped
    elif mode == "kg_organized":
        selected = organize_context_with_kg_paths(selected, kg_retrieval or {})

    if max_chunks is not None and max_chunks > 0:
        selected = selected[:max_chunks]
    if max_chars is not None and max_chars > 0:
        out = []
        total = 0
        for row in selected:
            row_chars = len(str(row.get("text", "")))
            if out and total + row_chars > max_chars:
                break
            out.append(row)
            total += row_chars
        selected = out
    return selected


def organize_context_with_kg_paths(rows: Sequence[Dict], kg_retrieval: Dict[str, object]) -> List[Dict]:
    supporting = [
        relation
        for relation in kg_retrieval.get("supporting_relations", [])
        if isinstance(relation, dict)
    ]
    relation_by_chunk: Dict[str, Dict[str, object]] = {}
    for relation in supporting:
        chunk_id = str(relation.get("chunk_id", ""))
        if not chunk_id:
            continue
        current = relation_by_chunk.get(chunk_id)
        current_depth = float(current.get("activation_depth", 99)) if current else 99
        relation_depth = float(relation.get("activation_depth", 1) or 1)
        relation_intent = float(relation.get("intent_score", 0.0) or 0.0)
        current_intent = float(current.get("intent_score", 0.0)) if current else -1.0
        if current is None or (relation_depth, -relation_intent) < (current_depth, -current_intent):
            relation_by_chunk[chunk_id] = relation

    out: List[Dict] = []
    for row in rows:
        organized = dict(row)
        relation = relation_by_chunk.get(str(row.get("chunk_id", "")))
        if relation:
            header = " | ".join(
                part
                for part in [
                    f"KG path: {relation.get('seed', '')}",
                    f"{relation.get('subject', '')} --{relation.get('predicate', '')}--> {relation.get('object', '')}",
                    f"depth {relation.get('activation_depth', 1)}",
                ]
                if str(part).strip()
            )
            organized["kg_context_header"] = header
            organized["kg_context_depth"] = int(relation.get("activation_depth", 1) or 1)
            organized["kg_context_intent_score"] = float(relation.get("intent_score", 0.0) or 0.0)
        else:
            organized["kg_context_header"] = "Semantic seed context" if row.get("retrieval_source") != "graph" else "KG-related context"
            organized["kg_context_depth"] = 0 if row.get("retrieval_source") != "graph" else 99
            organized["kg_context_intent_score"] = float(row.get("kg_intent_score", 0.0) or 0.0)
        out.append(organized)

    out.sort(
        key=lambda row: (
            0 if row.get("retrieval_source") in {"vector", "vector+graph"} else 1,
            int(row.get("kg_context_depth", 99) or 99),
            -float(row.get("kg_context_intent_score", 0.0) or 0.0),
            -float(row.get("kg_graph_score", 0.0) or 0.0),
            str(row.get("doc_id", "")),
            str(row.get("section_id", "")),
        )
    )
    return out


def decision_policy_result(
    *,
    prediction_confidence: float | None,
    runtime_retrieval_status: str,
    context_claim_recall: float | None,
    grounded_claim_ratio: float | None,
    min_confidence: float,
    min_context_claim_recall: float,
    min_grounded_claim_ratio: float,
) -> Dict[str, object]:
    reasons = []
    if prediction_confidence is not None and prediction_confidence < min_confidence:
        reasons.append("low_confidence")
    if context_claim_recall is not None and context_claim_recall < min_context_claim_recall:
        reasons.append("low_context_claim_recall")
    if grounded_claim_ratio is not None and grounded_claim_ratio < min_grounded_claim_ratio:
        reasons.append("low_grounded_claim_ratio")
    if runtime_retrieval_status in {"missing_evidence", "weak_evidence"}:
        reasons.append(runtime_retrieval_status)
    if "missing_evidence" in reasons:
        action = "abstain"
    elif reasons:
        action = "manual_review"
    else:
        action = "auto_accept"
    return {"decision_action": action, "decision_reasons": ",".join(reasons)}


def normalized_prediction_confidence(score: object) -> float | None:
    try:
        if score is None or score == "":
            return None
        value = float(score)
    except (TypeError, ValueError):
        return None
    if 0.0 <= value <= 1.0:
        return value
    return None


def build_run_dir(base_output_dir: str, run_name: Optional[str]) -> str:
    run_id = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, run_id)
    ensure_dir(run_dir)
    return run_dir


def _truncate_text(value: object, max_chars: int) -> str:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def prepare_experiment_resources(
    *,
    sections: Sequence[Section],
    paragraphs: Sequence[Paragraph],
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    retriever_type: str,
    embedding_model: str,
    kg_enabled: bool,
    questions: Sequence[Dict],
    search_backend_config: Dict[str, object] | None = None,
    search_index_name: str | None = None,
) -> Dict[str, object]:
    chunks = build_chunks(sections, paragraphs, strategy, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError(f"No chunks generated for strategy '{strategy}'.")

    kg_supervision_terms = build_kg_supervision_terms(questions) if kg_enabled else {}
    kg_graph = (
        build_knowledge_graph(chunks, extra_entity_terms=kg_supervision_terms)
        if kg_enabled
        else {"entities": [], "relations": []}
    )
    kg_graph_quality = graph_quality_diagnostics(kg_graph, chunks) if kg_enabled else {}
    retriever_state = build_retriever_with_backend(
        chunks,
        retriever_type,
        embedding_model,
        search_backend_config=search_backend_config,
        index_name=search_index_name,
    )
    return {
        "chunks": chunks,
        "kg_supervision_terms": kg_supervision_terms,
        "kg_graph": kg_graph,
        "kg_graph_quality": kg_graph_quality,
        "retriever_state": retriever_state,
    }


def run_single_experiment(
    *,
    sections: Sequence[Section],
    paragraphs: Sequence[Paragraph],
    questions: Sequence[Dict],
    run_dir: str,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    retriever_type: str,
    embedding_model: str,
    hybrid_alpha: float,
    llm_config: LLMConfig,
    judge_config: LLMConfig | None,
    create_strategy_visualization: bool,
    create_strategy_showcase: bool,
    kg_enabled: bool,
    kg_graph_weight: float,
    kg_profile: str = "balanced",
    kg_algorithm: str | None = None,
    kg_max_added_chunks: int | None = None,
    kg_ppr_iterations: int | None = None,
    kg_ppr_damping: float | None = None,
    kg_quality_threshold: float | None = None,
    kg_intent_weight: float | None = None,
    kg_ablation_edge_dropouts: str | Sequence[float] | None = None,
    answer_mode: str = "grounded_llm",
    context_mode: str = "ranked",
    max_context_chunks: int | None = None,
    max_context_chars: int | None = None,
    query_augmentation: str = "none",
    query_augmentation_max_terms: int = 8,
    decision_min_confidence: float = 0.0,
    decision_min_context_claim_recall: float = 0.0,
    decision_min_grounded_claim_ratio: float = 1.0,
    runtime_retrieval_evaluator_enabled: bool = True,
    abstain_on_weak_evidence: bool = False,
    self_rag_retry_on_weak_evidence: bool = False,
    self_rag_retry_max_attempts: int = 1,
    self_rag_critique: bool = False,
    rerank_top_n: int = 0,
    rerank_weight: float = 0.25,
    search_backend_config: Dict[str, object] | None = None,
    search_index_name: str | None = None,
    prepared_resources: Dict[str, object] | None = None,
    domain_specific_logic: bool = True,
) -> Dict:
    import pandas as pd

    judge_config = judge_config or LLMConfig(False, "", llm_config.api_key_env, 0.0)
    resolved_answer_mode = answer_mode or "grounded_llm"
    resolved_context_mode = context_mode or "ranked"
    requested_answer_mode = answer_mode
    requested_context_mode = context_mode
    resolved_max_context_chunks = max_context_chunks
    resolved_max_context_chars = max_context_chars
    resolved_query_augmentation = query_augmentation or "none"
    resolved_decision_min_confidence = float(decision_min_confidence or 0.0)
    resolved_decision_min_context_claim_recall = float(decision_min_context_claim_recall or 0.0)
    resolved_kg_ablation_edge_dropouts = parse_float_csv_list(kg_ablation_edge_dropouts)

    experiment_slug = f"{strategy}_{retriever_type}_size{chunk_size}_overlap{chunk_overlap}"
    experiment_slug = f"{experiment_slug}_ans_{resolved_answer_mode}_ctx_{resolved_context_mode}"
    if abstain_on_weak_evidence:
        experiment_slug = f"{experiment_slug}_abstain"
    if self_rag_retry_on_weak_evidence:
        experiment_slug = f"{experiment_slug}_retry{self_rag_retry_max_attempts}"
    if self_rag_critique:
        experiment_slug = f"{experiment_slug}_critique"
    if judge_config.enabled:
        experiment_slug = f"{experiment_slug}_judge"
    if kg_enabled:
        kg_slug = str(kg_profile or "kg").replace("/", "_").replace(" ", "_")
        algorithm_slug = str(kg_algorithm or "").replace("/", "_").replace(" ", "_")
        experiment_slug = f"{experiment_slug}_kg_{kg_slug}"
        if algorithm_slug:
            experiment_slug = f"{experiment_slug}_{algorithm_slug}"
    experiment_dir = os.path.join(run_dir, experiment_slug)
    ensure_dir(experiment_dir)

    resource_state = prepared_resources or prepare_experiment_resources(
        sections=sections,
        paragraphs=paragraphs,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retriever_type=retriever_type,
        embedding_model=embedding_model,
        kg_enabled=kg_enabled,
        questions=questions,
        search_backend_config=search_backend_config,
        search_index_name=search_index_name,
    )
    chunks = list(resource_state["chunks"])
    if not domain_specific_logic:
        chunks = [dict(chunk, _disable_domain_specific_logic=True) for chunk in chunks]

    chunks_df = pd.DataFrame(chunks)
    chunks_csv = os.path.join(experiment_dir, "chunks.csv")
    chunks_df.to_csv(chunks_csv, index=False)

    kg_supervision_terms = dict(resource_state.get("kg_supervision_terms") or {})
    kg_graph = dict(resource_state.get("kg_graph") or {"entities": [], "relations": []})
    kg_graph_quality = dict(resource_state.get("kg_graph_quality") or {})
    kg_entities_csv: Optional[str] = None
    kg_relations_csv: Optional[str] = None
    kg_metrics_csv: Optional[str] = None
    kg_summary_json: Optional[str] = None
    kg_ablation_csv: Optional[str] = None
    kg_ablation_summary_json: Optional[str] = None
    if kg_enabled:
        kg_entities_csv = os.path.join(experiment_dir, "kg_entities.csv")
        kg_relations_csv = os.path.join(experiment_dir, "kg_relations.csv")
        pd.DataFrame(kg_graph["entities"]).to_csv(kg_entities_csv, index=False)
        pd.DataFrame(kg_graph["relations"]).to_csv(kg_relations_csv, index=False)

    retriever_state = resource_state["retriever_state"]
    faiss_path: Optional[str] = None
    if retriever_state.get("backend") == "dense":
        import faiss

        faiss_path = os.path.join(experiment_dir, "index.faiss")
        faiss.write_index(retriever_state["index"], faiss_path)
    elif retriever_state.get("backend") == "hybrid":
        import faiss

        faiss_path = os.path.join(experiment_dir, "dense_index.faiss")
        faiss.write_index(retriever_state["dense"]["index"], faiss_path)

    results: List[Dict] = []
    retrieved_rows: List[Dict] = []
    answer_metric_rows: List[Dict] = []
    retrieval_metric_rows: List[Dict] = []
    kg_metric_rows: List[Dict] = []
    kg_ablation_rows: List[Dict] = []
    diagnostic_rows: List[Dict] = []
    judge_rows: List[Dict] = []
    claim_evidence_rows: List[Dict] = []
    for item in questions:
        question_type = infer_question_type(item)
        query_augmentation_config = (
            LLMConfig(True, llm_config.model, llm_config.api_key_env, 0.0)
            if resolved_query_augmentation in LLM_QUERY_AUGMENTATION_MODES
            else llm_config
        )
        query_augmentation_result = augment_query_with_llm(
            item["question"],
            query_augmentation_config,
            mode=resolved_query_augmentation,
            max_terms=query_augmentation_max_terms,
        )
        retrieval_query = query_augmentation_result.answer or item["question"]
        answer_metadata_filter = question_metadata_filter(
            item,
            chunks=chunks,
            include_document=False,
            domain_specific_logic=domain_specific_logic,
        )
        evaluation_metadata_filter = question_metadata_filter(
            item,
            chunks=chunks,
            include_document=True,
            domain_specific_logic=domain_specific_logic,
        )
        candidate_chunks = filter_chunks_by_metadata(chunks, evaluation_metadata_filter)
        if evaluation_metadata_filter and not candidate_chunks:
            raise ValueError(
                f"Question {item['id']} has evaluation metadata filter {evaluation_metadata_filter}, "
                "but no chunks match it."
            )
        answer_scope_chunks = filter_chunks_by_metadata(chunks, answer_metadata_filter)
        base_retrieved = retrieve_top_k(
            query=retrieval_query,
            retriever_state=retriever_state,
            chunks=chunks,
            k=min(max(top_k, rerank_top_n or top_k), len(answer_scope_chunks or chunks)),
            hybrid_alpha=hybrid_alpha,
            metadata_filter=answer_metadata_filter,
        )
        base_retrieved = rerank_with_lexical_signal(
            query=retrieval_query,
            rows=base_retrieved,
            top_k=min(top_k, len(base_retrieved)),
            rerank_top_n=rerank_top_n,
            rerank_weight=rerank_weight,
        )
        kg_retrieval = {
            "enabled": False,
            "seed_entities": [],
            "added_chunk_ids": [],
            "replaced_chunk_ids": [],
            "supporting_relations": [],
            "base_chunk_ids": [row["chunk_id"] for row in base_retrieved],
            "fused_chunk_ids": [row["chunk_id"] for row in base_retrieved],
        }
        if kg_enabled:
            retrieved, kg_retrieval = graph_augmented_retrieval(
                query=retrieval_query,
                retrieved=base_retrieved,
                graph=kg_graph,
                chunks=answer_scope_chunks or chunks,
                k=min(top_k, len(answer_scope_chunks or chunks)),
                graph_weight=kg_graph_weight,
                kg_profile=kg_profile,
                graph_algorithm=kg_algorithm,
                max_added_chunks=kg_max_added_chunks,
                ppr_iterations=kg_ppr_iterations,
                ppr_damping=kg_ppr_damping,
                quality_threshold=kg_quality_threshold,
                intent_weight=kg_intent_weight,
            )
        else:
            retrieved = base_retrieved
        retrieved = assemble_context_rows(
            retrieved,
            mode=resolved_context_mode,
            max_chunks=resolved_max_context_chunks,
            max_chars=resolved_max_context_chars,
            kg_retrieval=kg_retrieval,
        )
        runtime_retrieval_result = runtime_retrieval_evaluation(
            question=item["question"],
            retrieved=retrieved,
        )
        self_rag_retry_status = "disabled"
        self_rag_retry_query = ""
        self_rag_retry_attempts = 0
        if (
            runtime_retrieval_evaluator_enabled
            and self_rag_retry_on_weak_evidence
            and runtime_retrieval_result["status"] in {"missing_evidence", "weak_evidence"}
            and self_rag_retry_max_attempts > 0
        ):
            retry_config = LLMConfig(True, llm_config.model, llm_config.api_key_env, 0.0)
            for _attempt in range(max(1, self_rag_retry_max_attempts)):
                self_rag_retry_attempts += 1
                retry_result = rewrite_query_for_retry(
                    item["question"],
                    retrieved,
                    retry_config,
                    reason=str(runtime_retrieval_result.get("reason", "")),
                )
                self_rag_retry_status = retry_result.status
                self_rag_retry_query = retry_result.answer or ""
                if retry_result.status != "success" or not retry_result.answer:
                    break
                retry_base_retrieved = retrieve_top_k(
                    query=retry_result.answer,
                    retriever_state=retriever_state,
                    chunks=chunks,
                    k=min(max(top_k, rerank_top_n or top_k), len(answer_scope_chunks or chunks)),
                    hybrid_alpha=hybrid_alpha,
                    metadata_filter=answer_metadata_filter,
                )
                retry_base_retrieved = rerank_with_lexical_signal(
                    query=retry_result.answer,
                    rows=retry_base_retrieved,
                    top_k=min(top_k, len(retry_base_retrieved)),
                    rerank_top_n=rerank_top_n,
                    rerank_weight=rerank_weight,
                )
                if kg_enabled:
                    retry_retrieved, retry_kg_retrieval = graph_augmented_retrieval(
                        query=retry_result.answer,
                        retrieved=retry_base_retrieved,
                        graph=kg_graph,
                        chunks=answer_scope_chunks or chunks,
                        k=min(top_k, len(answer_scope_chunks or chunks)),
                        graph_weight=kg_graph_weight,
                        kg_profile=kg_profile,
                        graph_algorithm=kg_algorithm,
                        max_added_chunks=kg_max_added_chunks,
                        ppr_iterations=kg_ppr_iterations,
                        ppr_damping=kg_ppr_damping,
                        quality_threshold=kg_quality_threshold,
                        intent_weight=kg_intent_weight,
                    )
                else:
                    retry_retrieved = retry_base_retrieved
                    retry_kg_retrieval = kg_retrieval
                retry_retrieved = assemble_context_rows(
                    retry_retrieved,
                    mode=resolved_context_mode,
                    max_chunks=resolved_max_context_chunks,
                    max_chars=resolved_max_context_chars,
                    kg_retrieval=retry_kg_retrieval,
                )
                retry_runtime = runtime_retrieval_evaluation(
                    question=item["question"],
                    retrieved=retry_retrieved,
                )
                if retry_runtime["score"] > runtime_retrieval_result["score"]:
                    retrieval_query = retry_result.answer
                    base_retrieved = retry_base_retrieved
                    retrieved = retry_retrieved
                    kg_retrieval = retry_kg_retrieval
                    runtime_retrieval_result = retry_runtime
                if retry_runtime["status"] == "good_evidence":
                    self_rag_retry_status = "success_improved"
                    break
        should_abstain = (
            runtime_retrieval_evaluator_enabled
            and abstain_on_weak_evidence
            and runtime_retrieval_result["status"] in {"missing_evidence", "weak_evidence"}
        )
        if should_abstain:
            llm_result = LLMCallResult(
                answer=None,
                used=False,
                status="runtime_retrieval_abstained",
                error=str(runtime_retrieval_result["reason"]),
            )
            answer = "Not enough evidence in the retrieved context to answer reliably."
            answer_mode = "runtime_abstention"
        else:
            if resolved_answer_mode == "extractive":
                answer = keyword_extractive_answer(item["question"], retrieved)
                llm_result = LLMCallResult(answer=None, used=False, status="disabled", error=None)
                answer_mode = "extractive"
            else:
                llm_result = generate_answer_with_llm(
                    item["question"],
                    retrieved,
                    llm_config,
                    answer_mode=resolved_answer_mode,
                    context_style="cite_first" if resolved_answer_mode in {"cite_first", "claim_checklist"} else "standard",
                )
                answer = llm_result.answer
                answer_mode = resolved_answer_mode
                if answer is None:
                    answer = keyword_extractive_answer(item["question"], retrieved)
                    answer_mode = f"extractive_fallback_after_{resolved_answer_mode}"
        self_rag_critique_result = LLMCallResult(answer=answer, used=False, status="disabled", error=None)
        if self_rag_critique and not should_abstain and answer:
            critique_config = LLMConfig(True, llm_config.model, llm_config.api_key_env, 0.0)
            self_rag_critique_result = critique_and_revise_answer_with_llm(
                question=item["question"],
                answer=answer,
                retrieved=retrieved,
                llm_config=critique_config,
            )
            if self_rag_critique_result.answer:
                answer = self_rag_critique_result.answer
        claim_judge_result = judge_claims_with_llm(
            item=item,
            answer=answer,
            retrieved=retrieved,
            config=judge_config,
        )
        answer_metrics = evaluate_answer_metrics(
            item,
            answer,
            retrieved,
            claim_judge_result=claim_judge_result,
            runtime_retrieval_result=runtime_retrieval_result,
        )
        retrieval_metrics = evaluate_retrieval_metrics(
            item=item,
            retrieved=retrieved,
            candidate_chunks=candidate_chunks,
            k=min(top_k, len(candidate_chunks)),
        )
        kg_metrics = None
        if kg_enabled:
            base_kg_metrics = evaluate_kg_for_question(item, base_retrieved, kg_graph)
            kg_metrics = evaluate_kg_for_question(item, retrieved, kg_graph)
            kg_retrieval_diagnostics = evaluate_kg_retrieval_diagnostics(
                item=item,
                base_retrieved=base_retrieved,
                retrieved=retrieved,
                kg_retrieval=kg_retrieval,
                graph=kg_graph,
            )
            kg_path_grounding_metrics = evaluate_answer_kg_path_grounding(
                answer_metrics.claim_evidence_map,
                kg_retrieval,
            )
            for dropout_rate in resolved_kg_ablation_edge_dropouts:
                ablated_graph = ablate_kg_graph_edges(
                    kg_graph,
                    dropout_rate,
                    salt=f"{item['id']}|{kg_profile}",
                )
                ablated_retrieved, ablated_kg_retrieval = graph_augmented_retrieval(
                    query=retrieval_query,
                    retrieved=base_retrieved,
                    graph=ablated_graph,
                    chunks=answer_scope_chunks or chunks,
                    k=min(top_k, len(answer_scope_chunks or chunks)),
                    graph_weight=kg_graph_weight,
                    kg_profile=kg_profile,
                    graph_algorithm=kg_algorithm,
                    max_added_chunks=kg_max_added_chunks,
                    ppr_iterations=kg_ppr_iterations,
                    ppr_damping=kg_ppr_damping,
                    quality_threshold=kg_quality_threshold,
                    intent_weight=kg_intent_weight,
                )
                ablated_retrieved = assemble_context_rows(
                    ablated_retrieved,
                    mode=resolved_context_mode,
                    max_chunks=resolved_max_context_chunks,
                    max_chars=resolved_max_context_chars,
                    kg_retrieval=ablated_kg_retrieval,
                )
                ablated_metrics = evaluate_kg_for_question(item, ablated_retrieved, ablated_graph)
                ablated_recall = ablated_metrics.get("gold_kg_relation_evidence_recall")
                full_recall = kg_metrics.get("gold_kg_relation_evidence_recall")
                kg_ablation_rows.append(
                    {
                        "question_id": item["id"],
                        "question": item["question"],
                        "question_type": question_type,
                        "kg_profile": kg_profile,
                        "edge_dropout_rate": dropout_rate,
                        "full_gold_kg_relation_evidence_recall": full_recall,
                        "ablated_gold_kg_relation_evidence_recall": ablated_recall,
                        "kg_robustness_score": (
                            ablated_recall / full_recall
                            if ablated_recall is not None and full_recall not in {None, 0}
                            else None
                        ),
                        "kg_incompleteness_sensitivity": (
                            full_recall - ablated_recall
                            if ablated_recall is not None and full_recall is not None
                            else None
                        ),
                        "ablated_relation_count": len(ablated_graph.get("relations", [])),
                        "ablated_added_chunk_count": len(ablated_kg_retrieval.get("added_chunk_ids", [])),
                    }
                )
            # Compare relation-evidence coverage before and after graph expansion.
            # A positive delta means KG added context that contains more gold facts.
            base_relation_recall = base_kg_metrics["gold_kg_relation_evidence_recall"]
            final_relation_recall = kg_metrics["gold_kg_relation_evidence_recall"]
            relation_recall_delta = (
                final_relation_recall - base_relation_recall
                if base_relation_recall is not None and final_relation_recall is not None
                else None
            )
            kg_metrics.update(
                {
                    "base_gold_kg_doc_recall": base_kg_metrics["gold_kg_doc_recall"],  # doc coverage before KG expansion
                    "base_gold_kg_section_recall": base_kg_metrics["gold_kg_section_recall"],  # section coverage before KG expansion
                    "base_gold_kg_entity_pair_recall": base_kg_metrics[
                        "gold_kg_entity_pair_recall"
                    ],  # subject+object coverage before KG expansion
                    "base_gold_kg_relation_evidence_recall": base_relation_recall,  # relation evidence before KG expansion
                    "kg_relation_evidence_recall_delta": relation_recall_delta,  # final relation evidence minus base relation evidence
                    "kg_retrieval_added_chunk_count": len(kg_retrieval["added_chunk_ids"]),  # graph-only chunks added to context
                    "kg_retrieval_seed_entities": json.dumps(
                        kg_retrieval["seed_entities"], ensure_ascii=False
                    ),  # entities used to start KG traversal
                    "kg_retrieval_added_chunk_ids": json.dumps(
                        kg_retrieval["added_chunk_ids"], ensure_ascii=False
                    ),  # chunk ids pulled in by KG expansion
                    "kg_retrieval_replaced_chunk_ids": json.dumps(
                        kg_retrieval.get("replaced_chunk_ids", []), ensure_ascii=False
                    ),  # base top-k chunks displaced by graph-intent reranking
                    "kg_retrieval_supporting_relations": json.dumps(
                        kg_retrieval["supporting_relations"], ensure_ascii=False
                    ),  # graph edges that justified added chunks
                    "kg_profile": kg_retrieval.get("kg_profile", kg_profile),
                    "kg_graph_algorithm": kg_retrieval.get("graph_algorithm", kg_algorithm or ""),
                    "kg_settings": json.dumps(
                        kg_retrieval.get("kg_settings", {}), ensure_ascii=False
                    ),
                    "question_type": question_type,
                    **kg_retrieval_diagnostics,
                    **kg_path_grounding_metrics,
                }
            )
        auto_flag = answer_metrics.answer_accuracy_label
        diagnostics = diagnose_failure(
            answer_metrics=answer_metrics,
            retrieval_metrics=retrieval_metrics,
            llm_status=llm_result.status,
            answer_mode=answer_mode,
        )
        prediction_confidence = normalized_prediction_confidence(retrieved[0]["score"]) if retrieved else None
        decision_result = decision_policy_result(
            prediction_confidence=prediction_confidence,
            runtime_retrieval_status=answer_metrics.runtime_retrieval_status,
            context_claim_recall=answer_metrics.context_claim_recall,
            grounded_claim_ratio=answer_metrics.grounded_claim_ratio,
            min_confidence=resolved_decision_min_confidence,
            min_context_claim_recall=resolved_decision_min_context_claim_recall,
            min_grounded_claim_ratio=decision_min_grounded_claim_ratio,
        )

        for rank, row in enumerate(retrieved, start=1):
            relevance_grade = retrieval_relevance_grade(item, row)
            retrieved_rows.append(
                {
                    "question_id": item["id"],
                    "question": item["question"],
                    "question_type": question_type,
                    "retrieval_query": retrieval_query,
                    "query_augmentation_mode": resolved_query_augmentation,
                    "query_augmentation_status": query_augmentation_result.status,
                    "self_rag_retry_status": self_rag_retry_status,
                    "self_rag_retry_attempts": self_rag_retry_attempts,
                    "self_rag_retry_query": self_rag_retry_query,
                    "question_program_id": item.get("program_id", ""),
                    "question_program_name": item.get("program_name", ""),
                    "question_doc_id": item.get("doc_id", ""),
                    "answer_scope": json.dumps(answer_metadata_filter, ensure_ascii=False),
                    "evaluation_scope": json.dumps(evaluation_metadata_filter, ensure_ascii=False),
                    "rank": rank,
                    "auto_flag": auto_flag,
                    "retriever": retriever_type,
                    "chunk_id": row["chunk_id"],
                    "score": row["score"],
                    "base_retrieval_score": row.get("base_retrieval_score", row["score"]),  # original text-retriever score
                    "kg_graph_score": row.get("kg_graph_score", 0.0),  # graph relevance score used in fusion
                    "kg_intent_score": row.get("kg_intent_score", 0.0),  # predicate/query intent match used in fusion
                    "kg_chunk_quality_factor": row.get("kg_chunk_quality_factor", 1.0),  # version/noise filter multiplier
                    "kg_context_header": row.get("kg_context_header", ""),
                    "kg_context_depth": row.get("kg_context_depth", ""),
                    "kg_context_intent_score": row.get("kg_context_intent_score", ""),
                    "retrieval_source": row.get("retrieval_source", "vector"),  # vector, graph, or vector+graph
                    "doc_id": row["doc_id"],
                    "doc_path": row.get("doc_path", row["doc_id"]),
                    "program_id": row.get("program_id", ""),
                    "program_name": row.get("program_name", ""),
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
                "question_id": item["id"],
                "question": item["question"],
                "question_type": question_type,
                "retrieval_query": retrieval_query,
                "query_augmentation_status": query_augmentation_result.status,
                "self_rag_retry_status": self_rag_retry_status,
                "self_rag_retry_attempts": self_rag_retry_attempts,
                "self_rag_retry_query": self_rag_retry_query,
                "self_rag_critique_status": self_rag_critique_result.status,
                "self_rag_critique_error": self_rag_critique_result.error,
                "answer_mode": answer_mode,
                "context_mode": resolved_context_mode,
                **decision_result,
                "program_id": item.get("program_id", ""),
                "program_name": item.get("program_name", ""),
                "doc_id": item.get("doc_id", ""),
                "answer_scope": json.dumps(answer_metadata_filter, ensure_ascii=False),
                "evaluation_scope": json.dumps(evaluation_metadata_filter, ensure_ascii=False),
                "answer_accuracy_label": answer_metrics.answer_accuracy_label,
                "expected_answerable": answer_metrics.expected_answerable,
                "abstained": answer_metrics.abstained,
                "abstention_correct": answer_metrics.abstention_correct,
                "over_answered": answer_metrics.over_answered,
                "false_refusal": answer_metrics.false_refusal,
                "answerability_confidence": answer_metrics.answerability_confidence,
                "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
                "runtime_retrieval_action": answer_metrics.runtime_retrieval_action,
                "runtime_retrieval_score": answer_metrics.runtime_retrieval_score,
                "runtime_retrieval_reason": answer_metrics.runtime_retrieval_reason,
                "llm_used": llm_result.used,
                "llm_status": llm_result.status,
                "llm_error": llm_result.error,
                "gold_answer_overlap": answer_metrics.gold_answer_overlap,
                "answer_gold_support": answer_metrics.answer_gold_support,
                "proxy_faithfulness": answer_metrics.proxy_faithfulness,
                "proxy_context_relevance": answer_metrics.proxy_context_relevance,
                "answer_has_gold_substring": answer_metrics.answer_has_gold_substring,
                "claim_judge_used": answer_metrics.claim_judge_used,
                "claim_judge_status": answer_metrics.claim_judge_status,
                "claim_judge_error": answer_metrics.claim_judge_error,
                "claim_judge_model": answer_metrics.claim_judge_model,
                "gold_claim_count": answer_metrics.gold_claim_count,
                "answer_claim_count": answer_metrics.answer_claim_count,
                "context_claim_recall": answer_metrics.context_claim_recall,
                "answer_claim_recall": answer_metrics.answer_claim_recall,
                "answer_claim_precision": answer_metrics.answer_claim_precision,
                "answer_claim_f1": answer_metrics.answer_claim_f1,
                "factual_correctness_precision": answer_metrics.factual_correctness_precision,
                "factual_correctness_recall": answer_metrics.factual_correctness_recall,
                "factual_correctness_f1": answer_metrics.factual_correctness_f1,
                "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
                "hallucinated_claim_ratio": answer_metrics.hallucinated_claim_ratio,
                "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
                "noise_sensitivity_irrelevant": answer_metrics.noise_sensitivity_irrelevant,
                "context_utilization": answer_metrics.context_utilization,
                "context_entities_recall": answer_metrics.context_entities_recall,
                "answer_entity_precision": answer_metrics.answer_entity_precision,
                "evidence_attribution_precision": answer_metrics.evidence_attribution_precision,
                "evidence_attribution_recall": answer_metrics.evidence_attribution_recall,
                "evidence_attribution_f1": answer_metrics.evidence_attribution_f1,
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
                "question_id": item["id"],
                "question": item["question"],
                "question_type": question_type,
                "retrieval_query": retrieval_query,
                "context_mode": resolved_context_mode,
                **decision_result,
                "program_id": item.get("program_id", ""),
                "program_name": item.get("program_name", ""),
                "doc_id": item.get("doc_id", ""),
                "answer_scope": json.dumps(answer_metadata_filter, ensure_ascii=False),
                "evaluation_scope": json.dumps(evaluation_metadata_filter, ensure_ascii=False),
                "mrr_at_k": retrieval_metrics["mrr_at_k"],  # reciprocal rank of first relevant chunk in top-k
                "ndcg_at_k": retrieval_metrics["ndcg_at_k"],  # how well relevant chunks are ordered near the top
                "recall_at_k": retrieval_metrics["recall_at_k"],  # share of all relevant chunks captured in top-k
                "ragas_recall_at_k": retrieval_metrics["ragas_recall_at_k"],  # share of reference facts covered by top-k context
                "first_relevant_rank": retrieval_metrics["first_relevant_rank"],  # position of first relevant chunk
                "n_relevant_chunks": retrieval_metrics["n_relevant_chunks"],  # total relevant chunks in the candidate pool
                "n_retrieved_relevant_chunks": retrieval_metrics["n_retrieved_relevant_chunks"],  # relevant chunks found in top-k
                "target_doc_retrieved_at_k": retrieval_metrics["target_doc_retrieved_at_k"],  # whether top-k contains the hidden target doc_id
                "first_target_doc_rank": retrieval_metrics["first_target_doc_rank"],  # first rank from the hidden target doc_id
                "n_retrieved_target_doc_chunks": retrieval_metrics["n_retrieved_target_doc_chunks"],  # top-k chunks from target doc_id
            }
        )
        if kg_metrics is not None:
            kg_metric_rows.append(kg_metrics)
        diagnostic_rows.append(
            {
                "question_id": item["id"],
                "question": item["question"],
                "question_type": question_type,
                "program_id": item.get("program_id", ""),
                "program_name": item.get("program_name", ""),
                "doc_id": item.get("doc_id", ""),
                "answer_scope": json.dumps(answer_metadata_filter, ensure_ascii=False),
                "evaluation_scope": json.dumps(evaluation_metadata_filter, ensure_ascii=False),
                "primary_error_reason": diagnostics.primary_error_reason,
                "secondary_error_reason": diagnostics.secondary_error_reason,
                "context_mode": resolved_context_mode,
                **decision_result,
                "expected_answerable": answer_metrics.expected_answerable,
                "abstained": answer_metrics.abstained,
                "over_answered": answer_metrics.over_answered,
                "false_refusal": answer_metrics.false_refusal,
                "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
                "runtime_retrieval_action": answer_metrics.runtime_retrieval_action,
                "runtime_retrieval_score": answer_metrics.runtime_retrieval_score,
                "self_rag_retry_status": self_rag_retry_status,
                "self_rag_retry_attempts": self_rag_retry_attempts,
                "self_rag_retry_query": self_rag_retry_query,
                "self_rag_critique_status": self_rag_critique_result.status,
                "claim_judge_used": answer_metrics.claim_judge_used,
                "claim_judge_status": answer_metrics.claim_judge_status,
                "claim_diagnostic": answer_metrics.claim_diagnostic,
                "context_claim_recall": answer_metrics.context_claim_recall,
                "answer_claim_recall": answer_metrics.answer_claim_recall,
                "answer_claim_precision": answer_metrics.answer_claim_precision,
                "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
                "hallucinated_claim_ratio": answer_metrics.hallucinated_claim_ratio,
                "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
                "noise_sensitivity_irrelevant": answer_metrics.noise_sensitivity_irrelevant,
                "context_utilization": answer_metrics.context_utilization,
                "context_entities_recall": answer_metrics.context_entities_recall,
                "answer_entity_precision": answer_metrics.answer_entity_precision,
                "evidence_attribution_precision": answer_metrics.evidence_attribution_precision,
                "evidence_attribution_recall": answer_metrics.evidence_attribution_recall,
                "evidence_attribution_f1": answer_metrics.evidence_attribution_f1,
                "evidence_coverage": answer_metrics.evidence_coverage,
                "invalid_attribution_count": answer_metrics.invalid_attribution_count,
                "explanation": diagnostics.explanation,
            }
        )

        result_row = {
                "question_id": item["id"],
                "question": item["question"],
                "question_type": question_type,
                "retrieval_query": retrieval_query,
                "query_augmentation_mode": resolved_query_augmentation,
                "query_augmentation_status": query_augmentation_result.status,
                "answer_mode": answer_mode,
                "context_mode": resolved_context_mode,
                "self_rag_retry_status": self_rag_retry_status,
                "self_rag_retry_attempts": self_rag_retry_attempts,
                "self_rag_retry_query": self_rag_retry_query,
                "self_rag_critique_status": self_rag_critique_result.status,
                "self_rag_critique_error": self_rag_critique_result.error,
                **decision_result,
                "program_id": item.get("program_id", ""),
                "gold_answer": item.get("gold_answer", ""),
                "expected_keywords": json.dumps(item.get("expected_keywords", []), ensure_ascii=False),
                "retrieved_chunk_ids": json.dumps(
                    [row["chunk_id"] for row in retrieved], ensure_ascii=False
                ),
                "prediction_confidence": prediction_confidence,
                "prediction_score_raw": float(retrieved[0]["score"]) if retrieved else None,
                "retrieved_chunks": json.dumps(
                    [
                        {
                            "rank": rank,
                            "chunk_id": row["chunk_id"],
                            "score": row.get("score"),
                            "section_id": row.get("section_id", ""),
                            "title": row.get("title", ""),
                            "retrieval_source": row.get("retrieval_source", "vector"),
                            "text_preview": _truncate_text(row.get("text", ""), 800),
                        }
                        for rank, row in enumerate(retrieved, start=1)
                    ],
                    ensure_ascii=False,
                ),
                "kg_retrieval_added_chunk_ids": (
                    json.dumps(kg_retrieval["added_chunk_ids"], ensure_ascii=False)
                    if kg_enabled
                    else None
                ),  # extra chunks added from KG traversal
                "kg_retrieval_seed_entities": (
                    json.dumps(kg_retrieval["seed_entities"], ensure_ascii=False)
                    if kg_enabled
                    else None
                ),  # query/context entities used as graph seeds
                "kg_retrieval_replaced_chunk_ids": (
                    json.dumps(kg_retrieval.get("replaced_chunk_ids", []), ensure_ascii=False)
                    if kg_enabled
                    else None
                ),  # base top-k chunks displaced by graph-intent reranking
                "mrr_at_k": retrieval_metrics["mrr_at_k"],
                "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
                "recall_at_k": retrieval_metrics["recall_at_k"],
                "ragas_recall_at_k": retrieval_metrics["ragas_recall_at_k"],
                "expected_answerable": answer_metrics.expected_answerable,
                "abstained": answer_metrics.abstained,
                "abstention_correct": answer_metrics.abstention_correct,
                "over_answered": answer_metrics.over_answered,
                "false_refusal": answer_metrics.false_refusal,
                "answerability_confidence": answer_metrics.answerability_confidence,
                "runtime_retrieval_status": answer_metrics.runtime_retrieval_status,
                "runtime_retrieval_action": answer_metrics.runtime_retrieval_action,
                "runtime_retrieval_score": answer_metrics.runtime_retrieval_score,
                "runtime_retrieval_reason": answer_metrics.runtime_retrieval_reason,
                "first_target_doc_rank": retrieval_metrics["first_target_doc_rank"],
                "n_retrieved_target_doc_chunks": retrieval_metrics[
                    "n_retrieved_target_doc_chunks"
                ],
                "kg_entity_recall": kg_metrics["entity_recall"] if kg_metrics else None,
                "kg_relation_recall": kg_metrics["relation_recall"] if kg_metrics else None,
                "kg_relation_gap_count": kg_metrics["relation_gap_count"] if kg_metrics else None,
                "kg_has_relation_gap": kg_metrics["has_relation_gap"] if kg_metrics else None,
                "gold_kg_doc_recall": kg_metrics["gold_kg_doc_recall"] if kg_metrics else None,  # final context contains gold evidence doc
                "gold_kg_section_recall": kg_metrics["gold_kg_section_recall"] if kg_metrics else None,  # final context contains gold section
                "gold_kg_entity_pair_recall": kg_metrics["gold_kg_entity_pair_recall"] if kg_metrics else None,  # final context contains subject+object
                "gold_kg_relation_evidence_recall": (
                    kg_metrics["gold_kg_relation_evidence_recall"] if kg_metrics else None
                ),  # final context contains subject+object+relation cue
                "base_gold_kg_relation_evidence_recall": (
                    kg_metrics["base_gold_kg_relation_evidence_recall"] if kg_metrics else None
                ),  # same relation-evidence metric before KG expansion
                "kg_relation_evidence_recall_delta": (
                    kg_metrics["kg_relation_evidence_recall_delta"] if kg_metrics else None
                ),  # improvement from KG expansion
                "kg_graph_gain_at_k": kg_metrics["kg_graph_gain_at_k"] if kg_metrics else None,
                "kg_graph_noise_at_k": kg_metrics["kg_graph_noise_at_k"] if kg_metrics else None,
                "kg_added_evidence_precision": kg_metrics["kg_added_evidence_precision"] if kg_metrics else None,
                "kg_added_evidence_recall": kg_metrics["kg_added_evidence_recall"] if kg_metrics else None,
                "kg_path_availability": kg_metrics["kg_path_availability"] if kg_metrics else None,
                "kg_path_correctness": kg_metrics["kg_path_correctness"] if kg_metrics else None,
                "answer_claim_kg_path_support_rate": (
                    kg_metrics["answer_claim_kg_path_support_rate"] if kg_metrics else None
                ),
                "kg_path_grounded_claim_ratio": (
                    kg_metrics["kg_path_grounded_claim_ratio"] if kg_metrics else None
                ),
                "unsupported_claim_missing_kg_path_rate": (
                    kg_metrics["unsupported_claim_missing_kg_path_rate"] if kg_metrics else None
                ),
                "kg_error_type": kg_metrics["kg_error_type"] if kg_metrics else None,  # first missing KG evidence layer
                "gold_answer_overlap": answer_metrics.gold_answer_overlap,
                "answer_gold_support": answer_metrics.answer_gold_support,
                "proxy_faithfulness": answer_metrics.proxy_faithfulness,
                "proxy_context_relevance": answer_metrics.proxy_context_relevance,
                "gold_claim_count": answer_metrics.gold_claim_count,
                "answer_claim_count": answer_metrics.answer_claim_count,
                "context_claim_recall": answer_metrics.context_claim_recall,
                "answer_claim_recall": answer_metrics.answer_claim_recall,
                "answer_claim_precision": answer_metrics.answer_claim_precision,
                "answer_claim_f1": answer_metrics.answer_claim_f1,
                "factual_correctness_precision": answer_metrics.factual_correctness_precision,
                "factual_correctness_recall": answer_metrics.factual_correctness_recall,
                "factual_correctness_f1": answer_metrics.factual_correctness_f1,
                "grounded_claim_ratio": answer_metrics.grounded_claim_ratio,
                "hallucinated_claim_ratio": answer_metrics.hallucinated_claim_ratio,
                "noise_sensitivity_relevant": answer_metrics.noise_sensitivity_relevant,
                "noise_sensitivity_irrelevant": answer_metrics.noise_sensitivity_irrelevant,
                "context_utilization": answer_metrics.context_utilization,
                "context_entities_recall": answer_metrics.context_entities_recall,
                "answer_entity_precision": answer_metrics.answer_entity_precision,
                "evidence_attribution_precision": answer_metrics.evidence_attribution_precision,
                "evidence_attribution_recall": answer_metrics.evidence_attribution_recall,
                "evidence_attribution_f1": answer_metrics.evidence_attribution_f1,
                "evidence_coverage": answer_metrics.evidence_coverage,
                "attributed_answer_claim_count": answer_metrics.attributed_answer_claim_count,
                "attributed_gold_claim_count": answer_metrics.attributed_gold_claim_count,
                "invalid_attribution_count": answer_metrics.invalid_attribution_count,
                "claim_evidence_map": json.dumps(
                    answer_metrics.claim_evidence_map,
                    ensure_ascii=False,
                ),
                "unsupported_claim_count": answer_metrics.unsupported_claim_count,
                "missing_gold_claim_count": answer_metrics.missing_gold_claim_count,
                "contradicted_claim_count": answer_metrics.contradicted_claim_count,
                "claim_diagnostic": answer_metrics.claim_diagnostic,
                "answer": answer,
                "auto_flag": auto_flag,
                "primary_error_reason": diagnostics.primary_error_reason,
                "secondary_error_reason": diagnostics.secondary_error_reason,
                "diagnostic_explanation": diagnostics.explanation,
            }
        results.append(apply_question_recommendations(result_row))
        if claim_judge_result.used or claim_judge_result.status not in {"disabled", "no_claims"}:
            judge_rows.append(
                {
                    "question_id": item["id"],
                    "status": claim_judge_result.status,
                    "error": claim_judge_result.error,
                    "model": claim_judge_result.model,
                    "metrics": claim_judge_result.metrics,
                    "raw_response": _truncate_text(claim_judge_result.raw_response, 12000),
                }
            )
        for claim_index, claim_row in enumerate(answer_metrics.claim_evidence_map, start=1):
            claim_evidence_rows.append(
                {
                    "question_id": item["id"],
                    "question": item["question"],
                    "claim_index": claim_index,
                    "claim_type": claim_row.get("claim_type"),
                    "claim": claim_row.get("claim"),
                    "context_nli": claim_row.get("context_nli"),
                    "answer_nli": claim_row.get("answer_nli"),
                    "gold_nli": claim_row.get("gold_nli"),
                    "supporting_chunk_ids": json.dumps(
                        claim_row.get("supporting_chunk_ids", []),
                        ensure_ascii=False,
                    ),
                    "raw_supporting_chunk_ids": json.dumps(
                        claim_row.get("raw_supporting_chunk_ids", claim_row.get("supporting_chunk_ids", [])),
                        ensure_ascii=False,
                    ),
                    "invalid_attribution_count": claim_row.get("invalid_attribution_count", 0),
                    "supporting_chunks": json.dumps(
                        claim_row.get("supporting_chunks", []),
                        ensure_ascii=False,
                    ),
                }
            )

    results_df = pd.DataFrame(results)

    results_csv = os.path.join(experiment_dir, "rag_results.csv")
    results_df.to_csv(results_csv, index=False)

    retrieved_df = pd.DataFrame(retrieved_rows)
    retrieved_csv = os.path.join(experiment_dir, "retrieved_chunks.csv")
    retrieved_df.to_csv(retrieved_csv, index=False)

    answer_metrics_df = pd.DataFrame(answer_metric_rows)
    answer_metrics_csv = os.path.join(experiment_dir, "answer_metrics.csv")
    answer_metrics_df.to_csv(answer_metrics_csv, index=False)
    aggregate_answer_metrics = summarize_answer_metrics(answer_metric_rows)
    answer_metrics_by_question_type = summarize_rows_by_question_type(
        answer_metric_rows,
        [
            "context_claim_recall",
            "grounded_claim_ratio",
            "hallucinated_claim_ratio",
            "factual_correctness_recall",
            "evidence_attribution_recall",
        ],
    )
    answer_metrics_json = os.path.join(experiment_dir, "answer_metrics_summary.json")
    with open(answer_metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                **aggregate_answer_metrics,
                "by_question_type": answer_metrics_by_question_type,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    claim_evidence_csv: Optional[str] = None
    claim_evidence_jsonl: Optional[str] = None
    if claim_evidence_rows:
        claim_evidence_csv = os.path.join(experiment_dir, "claim_evidence_map.csv")
        pd.DataFrame(claim_evidence_rows).to_csv(claim_evidence_csv, index=False)
        claim_evidence_jsonl = os.path.join(experiment_dir, "claim_evidence_map.jsonl")
        with open(claim_evidence_jsonl, "w", encoding="utf-8") as f:
            for row in claim_evidence_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    retrieval_metrics_df = pd.DataFrame(retrieval_metric_rows)
    retrieval_metrics_csv = os.path.join(experiment_dir, "retrieval_metrics.csv")
    retrieval_metrics_df.to_csv(retrieval_metrics_csv, index=False)
    aggregate_retrieval_metrics = summarize_retrieval_metrics(retrieval_metric_rows)
    retrieval_metrics_by_question_type = summarize_rows_by_question_type(
        retrieval_metric_rows,
        ["mrr_at_k", "ndcg_at_k", "recall_at_k", "ragas_recall_at_k"],
    )
    retrieval_metrics_json = os.path.join(experiment_dir, "retrieval_metrics_summary.json")
    with open(retrieval_metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                **aggregate_retrieval_metrics,
                "by_question_type": retrieval_metrics_by_question_type,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    aggregate_kg_metrics = summarize_kg_metrics(kg_metric_rows) if kg_enabled else {}
    kg_metrics_by_question_type = (
        summarize_rows_by_question_type(
            kg_metric_rows,
            [
                "gold_kg_relation_evidence_recall",
                "kg_relation_evidence_recall_delta",
                "kg_graph_gain_at_k",
                "kg_graph_noise_at_k",
                "kg_added_evidence_precision",
                "kg_added_evidence_recall",
                "kg_path_availability",
                "kg_path_correctness",
                "kg_graph_faithfulness",
                "answer_claim_kg_path_support_rate",
                "kg_path_grounded_claim_ratio",
            ],
        )
        if kg_enabled
        else {}
    )
    kg_ablation_summary = (
        {
            "edge_dropout_rates": resolved_kg_ablation_edge_dropouts,
            "overall": summarize_rows_by_question_type(
                [{**row, "question_type": "all"} for row in kg_ablation_rows],
                ["kg_robustness_score", "kg_incompleteness_sensitivity", "ablated_gold_kg_relation_evidence_recall"],
            ).get("all", {}),
            "by_question_type": summarize_rows_by_question_type(
                kg_ablation_rows,
                ["kg_robustness_score", "kg_incompleteness_sensitivity", "ablated_gold_kg_relation_evidence_recall"],
            ),
        }
        if kg_enabled and kg_ablation_rows
        else {}
    )
    if kg_enabled:
        kg_metrics_df = pd.DataFrame(kg_metric_rows)
        kg_metrics_csv = os.path.join(experiment_dir, "kg_metrics.csv")
        kg_metrics_df.to_csv(kg_metrics_csv, index=False)
        if kg_ablation_rows:
            kg_ablation_csv = os.path.join(experiment_dir, "kg_incompleteness_ablation.csv")
            pd.DataFrame(kg_ablation_rows).to_csv(kg_ablation_csv, index=False)
            kg_ablation_summary_json = os.path.join(experiment_dir, "kg_incompleteness_ablation_summary.json")
            with open(kg_ablation_summary_json, "w", encoding="utf-8") as f:
                json.dump(kg_ablation_summary, f, ensure_ascii=False, indent=2)
        kg_summary_json = os.path.join(experiment_dir, "kg_summary.json")
        with open(kg_summary_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_entities": len(kg_graph["entities"]),
                    "n_relations": len(kg_graph["relations"]),
                    "graph_quality": kg_graph_quality,
                    "metrics": aggregate_kg_metrics,
                    "metrics_by_question_type": kg_metrics_by_question_type,
                    "incompleteness_ablation": kg_ablation_summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    diagnostics_df = pd.DataFrame(diagnostic_rows)
    diagnostics_csv = os.path.join(experiment_dir, "diagnostics.csv")
    diagnostics_df.to_csv(diagnostics_csv, index=False)
    aggregate_diagnostics = summarize_diagnostics(diagnostic_rows)
    prediction_calibration = summarize_confidence_calibration(
        results,
        confidence_key="prediction_confidence",
        correct_fn=lambda row: row.get("auto_flag") == "correct",
    )
    decision_counts = {
        action: sum(1 for row in results if row.get("decision_action") == action)
        for action in sorted({str(row.get("decision_action", "")) for row in results if row.get("decision_action")})
    }
    diagnostics_json = os.path.join(experiment_dir, "diagnostics_summary.json")
    with open(diagnostics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_diagnostics, f, ensure_ascii=False, indent=2)
    evidence_graph_summary = build_document_evidence_graph_summary(
        result_rows=results,
        ranking_rows=retrieved_rows,
    )
    evidence_graph_json = os.path.join(experiment_dir, "evidence_graph_summary.json")
    with open(evidence_graph_json, "w", encoding="utf-8") as f:
        json.dump(evidence_graph_summary, f, ensure_ascii=False, indent=2)

    claim_judge_jsonl: Optional[str] = None
    if judge_rows:
        claim_judge_jsonl = os.path.join(experiment_dir, "claim_judge_results.jsonl")
        with open(claim_judge_jsonl, "w", encoding="utf-8") as f:
            for row in judge_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    error_report_md = os.path.join(experiment_dir, "error_report.md")
    with open(error_report_md, "w", encoding="utf-8") as f:
        f.write(f"# Error Report: {experiment_slug}\n\n")
        for row in diagnostic_rows:
            f.write(f"## {row['question_id']}\n")
            f.write(f"- Question: {row['question']}\n")
            f.write(f"- Primary reason: {row['primary_error_reason']}\n")
            f.write(f"- Secondary reason: {row['secondary_error_reason']}\n")
            f.write(f"- Explanation: {row['explanation']}\n\n")

    summary = {
        "experiment": experiment_slug,
        "chunking_strategy": strategy,
        "retriever": retriever_type,
        "search_backend": retriever_state.get("search_backend", {"backend": "local", "index_name": None}),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "hybrid_alpha": hybrid_alpha if retriever_type == "hybrid" else None,
        "top_k": top_k,
        "reranker": {
            "enabled": rerank_top_n > 1 and rerank_weight > 0.0,
            "type": "lexical_overlap",
            "top_n": rerank_top_n,
            "weight": rerank_weight,
        },
        "evaluation_settings": {
            "requested_answer_mode": requested_answer_mode,
            "requested_context_mode": requested_context_mode,
            "answer_mode": resolved_answer_mode,
            "context_mode": resolved_context_mode,
            "max_context_chunks": resolved_max_context_chunks,
            "max_context_chars": resolved_max_context_chars,
            "query_augmentation": resolved_query_augmentation,
            "query_augmentation_max_terms": query_augmentation_max_terms,
            "self_rag_retry_on_weak_evidence": self_rag_retry_on_weak_evidence,
            "self_rag_retry_max_attempts": self_rag_retry_max_attempts,
            "self_rag_critique": self_rag_critique,
            "decision_min_confidence": resolved_decision_min_confidence,
            "decision_min_context_claim_recall": resolved_decision_min_context_claim_recall,
            "decision_min_grounded_claim_ratio": decision_min_grounded_claim_ratio,
            "decision_counts": decision_counts,
        },
        "n_chunks": len(chunks),
        "n_questions": len(questions),
        "n_correct": int((results_df["auto_flag"] == "correct").sum()),
        "n_incorrect": int((results_df["auto_flag"] == "incorrect").sum()),
        "answer_metrics": aggregate_answer_metrics,
        "answer_metrics_by_question_type": answer_metrics_by_question_type,
        "retrieval_metrics": aggregate_retrieval_metrics,
        "retrieval_metrics_by_question_type": retrieval_metrics_by_question_type,
        "kg": {
            "enabled": kg_enabled,
            "graph_weight": kg_graph_weight if kg_enabled else None,
            "profile": kg_profile if kg_enabled else None,
            "algorithm": kg_algorithm if kg_enabled else None,
            "ablation_edge_dropouts": resolved_kg_ablation_edge_dropouts if kg_enabled else [],
            "weak_supervision_entity_terms": sum(len(rows) for rows in kg_supervision_terms.values()),
            "n_entities": len(kg_graph["entities"]),
            "n_relations": len(kg_graph["relations"]),
            "graph_quality": kg_graph_quality if kg_enabled else {},
            "metrics": aggregate_kg_metrics,
            "metrics_by_question_type": kg_metrics_by_question_type,
            "incompleteness_ablation": kg_ablation_summary,
        },
        "evidence_graph": evidence_graph_summary,
        "diagnostics": aggregate_diagnostics,
        "llm": {
            "enabled": (
                llm_config.enabled
                or resolved_query_augmentation in LLM_QUERY_AUGMENTATION_MODES
                or self_rag_retry_on_weak_evidence
                or self_rag_critique
            ),
            "model": (
                llm_config.model
                if (
                    llm_config.enabled
                    or resolved_query_augmentation in LLM_QUERY_AUGMENTATION_MODES
                    or self_rag_retry_on_weak_evidence
                    or self_rag_critique
                )
                else None
            ),
            "answer_generation": llm_config.enabled,
            "query_augmentation": resolved_query_augmentation,
            "self_rag_retry_on_weak_evidence": self_rag_retry_on_weak_evidence,
            "self_rag_critique": self_rag_critique,
        },
        "runtime_retrieval_evaluator": {
            "enabled": runtime_retrieval_evaluator_enabled,
            "abstain_on_weak_evidence": abstain_on_weak_evidence,
            "self_rag_retry_on_weak_evidence": self_rag_retry_on_weak_evidence,
            "self_rag_retry_max_attempts": self_rag_retry_max_attempts,
            "self_rag_critique": self_rag_critique,
        },
        "judge": {
            "enabled": bool(judge_config and judge_config.enabled),
            "model": judge_config.model if judge_config and judge_config.enabled else None,
            "status_counts": aggregate_diagnostics.get("counts_by_claim_judge_status", {}),
            "claim_judge_results_jsonl": claim_judge_jsonl,
            "claim_evidence_map_csv": claim_evidence_csv,
            "claim_evidence_map_jsonl": claim_evidence_jsonl,
        },
        "calibration": {
            "prediction_confidence": prediction_calibration,
        },
        "visualization": {
            "enabled": create_strategy_visualization,
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
            "chunks_csv": chunks_csv,
            "faiss_index": faiss_path,
            "rag_results_csv": results_csv,
            "retrieved_chunks_csv": retrieved_csv,
            "answer_metrics_csv": answer_metrics_csv,
            "answer_metrics_summary_json": answer_metrics_json,
            "claim_evidence_map_csv": claim_evidence_csv,
            "claim_evidence_map_jsonl": claim_evidence_jsonl,
            "retrieval_metrics_csv": retrieval_metrics_csv,
            "retrieval_metrics_summary_json": retrieval_metrics_json,
            "kg_entities_csv": kg_entities_csv,
            "kg_relations_csv": kg_relations_csv,
            "kg_metrics_csv": kg_metrics_csv,
            "kg_summary_json": kg_summary_json,
            "kg_incompleteness_ablation_csv": kg_ablation_csv,
            "kg_incompleteness_ablation_summary_json": kg_ablation_summary_json,
            "diagnostics_csv": diagnostics_csv,
            "diagnostics_summary_json": diagnostics_json,
            "evidence_graph_summary_json": evidence_graph_json,
            "claim_judge_results_jsonl": claim_judge_jsonl,
            "error_report_md": error_report_md,
            "strategy_score_profile_svg": None,
            "strategy_chunk_alignment_svg": None,
            "strategy_metric_overview_svg": None,
            "strategy_diagnostics_svg": None,
            "strategy_showcase_md": None,
        },
    }
    quality_advisor = build_run_advisor(summary, results)
    quality_advisor_json = os.path.join(experiment_dir, "quality_advisor.json")
    with open(quality_advisor_json, "w", encoding="utf-8") as f:
        json.dump(quality_advisor, f, ensure_ascii=False, indent=2)
    quality_report_md = write_quality_report(
        os.path.join(experiment_dir, "quality_report.md"),
        quality_advisor,
        summary,
    )
    summary["advisor"] = quality_advisor
    summary["outputs"]["quality_advisor_json"] = quality_advisor_json
    summary["outputs"]["quality_report_md"] = quality_report_md

    if create_strategy_visualization:
        score_profile_svg = write_strategy_score_profile_svg(
            retrieved_rows,
            os.path.join(experiment_dir, "strategy_score_profile.svg"),
            experiment_slug=experiment_slug,
            top_k=top_k,
        )
        chunk_alignment_svg = write_chunk_relevance_comparison_svg(
            retrieved_rows,
            os.path.join(experiment_dir, "strategy_chunk_alignment.svg"),
            top_k=top_k,
            chart_label=experiment_slug,
        )
        unique_chunk_alignment_svg = write_chunk_relevance_comparison_svg(
            retrieved_rows,
            os.path.join(experiment_dir, "strategy_unique_chunk_alignment.svg"),
            top_k=top_k,
            chart_label=experiment_slug,
            unique_relevance=True,
        )
        summary["visualization"]["strategy_score_profile_svg"] = score_profile_svg
        summary["visualization"]["strategy_chunk_alignment_svg"] = chunk_alignment_svg
        summary["visualization"]["strategy_unique_chunk_alignment_svg"] = unique_chunk_alignment_svg
        summary["outputs"]["strategy_score_profile_svg"] = score_profile_svg
        summary["outputs"]["strategy_chunk_alignment_svg"] = chunk_alignment_svg
        summary["outputs"]["strategy_unique_chunk_alignment_svg"] = unique_chunk_alignment_svg

    if create_strategy_showcase:
        showcase_bundle = write_strategy_showcase_bundle(
            summary=summary,
            retrieved_rows=retrieved_rows,
            experiment_dir=experiment_dir,
        )
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

    with open(os.path.join(experiment_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
