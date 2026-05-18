from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List

from rag_eval.evaluation.advisor import apply_question_recommendations, build_run_advisor, write_quality_report
from rag_eval.classifiers.cpv_baseline import build_cpv_chunks, build_parent_lookup, load_cpv_catalog, load_queries
from rag_eval.evaluation.adapters import build_items_from_cpv_queries
from rag_eval.evaluation.core import (
    EvaluationItem,
    PredictionRecord,
    RankedCandidate,
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
    summarize_answer_metrics,
    summarize_diagnostics,
    summarize_retrieval_metrics,
    summarize_confidence_calibration,
)
from rag_eval.retrieval.engines import build_retriever, rerank_with_lexical_signal, retrieve_top_k
from rag_eval.reporting.visualization import write_classifier_showcase_bundle


DUPLICATE_METRIC_COLUMNS = [
    "gold_hit_at_k",
    "gold_first_rank",
    "reciprocal_rank",
    "hierarchical_distance_top1",
    "normalized_hierarchical_distance_top1",
    "cpv_hierarchy_distance_top1",
    "cpv_common_prefix_length_top1",
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
]


def _apply_showcase_bundle(summary: Dict[str, object], showcase_bundle: Dict[str, object]) -> None:
    summary["showcase"] = showcase_bundle
    summary["visualization"]["enabled"] = True
    summary["visualization"]["strategy_score_profile_svg"] = showcase_bundle["score_profile_svg"]
    summary["visualization"]["strategy_chunk_alignment_svg"] = showcase_bundle["chunk_alignment_svg"]
    summary["outputs"]["strategy_score_profile_svg"] = showcase_bundle["score_profile_svg"]
    summary["outputs"]["strategy_chunk_alignment_svg"] = showcase_bundle["chunk_alignment_svg"]
    summary["outputs"]["strategy_metric_overview_svg"] = showcase_bundle["metric_overview_svg"]
    summary["outputs"]["strategy_diagnostics_svg"] = showcase_bundle["diagnostics_svg"]
    summary["outputs"]["strategy_showcase_md"] = showcase_bundle["showcase_md"]


def _normalize_prediction_label(candidate: Dict[str, object]) -> str:
    for key in ["label", "cpv_code", "answer", "id", "code"]:
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalize_prediction_score(candidate: Dict[str, object], *, fallback: float) -> float:
    for key in ["score", "confidence", "probability"]:
        value = candidate.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return fallback


def _normalize_header(value: object) -> str:
    return "".join(char for char in str(value or "").casefold() if char.isalnum())


def _first_present(row: Dict[str, object], keys: List[str]) -> object:
    for key in keys:
        if key in row and row[key] not in {None, ""}:
            return row[key]
    return ""


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


def _prepared_candidates_from_row(row: Dict[str, object], *, row_index: int) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    numbered_ranks = sorted(
        {
            int(match.group(1))
            for key in row
            for match in [re.fullmatch(r"predictedcpv(\d+)", str(key))]
            if match
        }
    )
    for rank in numbered_ranks:
        predicted_codes = _extract_cpv_codes(row.get(f"predictedcpv{rank}"))
        if not predicted_codes:
            continue
        score_raw = _first_present(
            row,
            [
                f"vectorscore{rank}",
                f"rrfscore{rank}",
                f"score{rank}",
                f"confidence{rank}",
                f"probability{rank}",
                f"retrievalscore{rank}",
            ],
        )
        base_score = _parse_float(score_raw, fallback=max(0.0, 1.0 - rank * 0.001))
        for offset, predicted_code in enumerate(predicted_codes):
            candidates.append(
                {
                    "label": predicted_code,
                    "score": float(base_score or 0.0) - (offset * 0.000001),
                    "rank": rank,
                    "source_row": row,
                    "source_row_index": row_index,
                }
            )
    if candidates:
        return candidates

    predicted_codes = _extract_cpv_codes(
        _first_present(row, ["predictedcpv", "prediction", "predicted", "cpv", "answercpv"])
    )
    if not predicted_codes:
        predicted_codes = [""]
    score_raw = _first_present(row, ["rrfscore", "vectorscore", "score", "confidence", "probability", "retrievalscore"])
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
            "source_row": row,
            "source_row_index": row_index,
        }
        for offset, predicted_code in enumerate(predicted_codes)
    ]


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
    candidate_label: str,
    score: float,
    rank: int,
    query_text: str,
    catalog_by_code: Dict[str, object],
    source_row: Dict[str, object],
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
    chunk_id = _first_present(source_row, ["chunkid", "chunkids", "retrievedchunkids", "retrievedchunkid"])
    chunk_text = _first_present(source_row, ["chunk", "chunks", "chunktext", "retrievedchunks", "retrievedchunktext", "context", "contexts"])
    if chunk_id:
        row["chunk_id"] = str(chunk_id)
    if chunk_text:
        row["text"] = str(chunk_text)
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
]


def _drop_irrelevant_classifier_columns(df, *, classifier_type: str):
    if df.empty:
        return df
    if classifier_type not in {"ted_cpv", "api_classifier", "prepared_rag_results"}:
        return df
    return df.drop(columns=[column for column in CPV_IRRELEVANT_COLUMNS if column in df.columns])


def _standardize_rag_results_columns(df):
    if df.empty:
        return df
    df = _drop_duplicate_metric_columns(df)
    leading_columns = [
        "question_id",
        "question",
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
    same_group_top1 = bool(common_prefix_length is not None and common_prefix_length >= 3)
    partial_correct = bool(same_group_top1 and not exact_top1_match)
    return {
        "exact_top1_match": exact_top1_match,
        "cpv_hierarchy_similarity_top1": hierarchy_similarity,
        "partial_correct": partial_correct,
        "top1_predicted_cpv": top1_label,
        "expected_cpv_codes": json.dumps(list(gold_set), ensure_ascii=False),
    }


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
        "ragas_recall_at_k": retrieval_metrics["ragas_recall_at_k"],
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
    cpv_catalog_path: str,
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
) -> Dict[str, object]:
    import pandas as pd

    catalog = load_cpv_catalog(cpv_catalog_path)
    queries = load_queries(queries_path)
    chunks = build_cpv_chunks(catalog, use_examples=use_examples)
    retriever_state = build_retriever(chunks, retriever, embedding_model)
    # mean_cpv_hierarchy_similarity_top1 shows how close top-1 is to the gold CPV code.
    parent_lookup = build_parent_lookup(catalog)
    label_by_code = {record.code: record.label for record in catalog}
    description_by_code = {record.code: record.description for record in catalog}

    prediction_records: List[PredictionRecord] = []
    ranking_rows: List[Dict[str, object]] = []
    result_rows: List[Dict[str, object]] = []
    answer_metric_rows: List[Dict[str, object]] = []
    retrieval_metric_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []

    for query in queries:
        retrieved = retrieve_top_k(
            query=query.query,
            retriever_state=retriever_state,
            chunks=chunks,
            k=min(max(top_k, rerank_top_n or top_k), len(chunks)),
        )
        retrieved = rerank_with_lexical_signal(
            query=query.query,
            rows=retrieved,
            top_k=min(top_k, len(retrieved)),
            rerank_top_n=rerank_top_n,
            rerank_weight=rerank_weight,
        )
        prediction_records.append(
            PredictionRecord(
                id=query.id,
                candidates=[
                    RankedCandidate(label=str(row["cpv_code"]), score=float(row["score"]))
                    for row in retrieved
                ],
                metadata={"query": query.query},
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

        for rank, row in enumerate(retrieved, start=1):
            relevance_grade = retrieval_relevance_grade(item, row)
            ranking_rows.append(
                {
                    "question_id": query.id,
                    "question": query.query,
                    "rank": rank,
                    "auto_flag": answer_metrics.answer_accuracy_label,
                    "retriever": retriever,
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
                "answer_accuracy_label": answer_metrics.answer_accuracy_label,
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
                "ragas_recall_at_k": retrieval_metrics["ragas_recall_at_k"],
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
                "program_id": "cpv",
                "program_name": "CPV",
                "doc_id": query.gold_cpv_code,
                "answer_scope": json.dumps({}, ensure_ascii=False),
                "evaluation_scope": json.dumps({"doc_id": [query.gold_cpv_code]}, ensure_ascii=False),
                "gold_answer": item["gold_answer"],
                "expected_keywords": json.dumps(item["expected_keywords"], ensure_ascii=False),
                **classification_metrics,
                **_all_retrieval_metric_fields(retrieval_metrics),
                **_all_answer_metric_fields(answer_metrics),
                "expected_answerable": answer_metrics.expected_answerable,
                "abstained": answer_metrics.abstained,
                "abstention_correct": answer_metrics.abstention_correct,
                "over_answered": answer_metrics.over_answered,
                "false_refusal": answer_metrics.false_refusal,
                "prediction_confidence": float(retrieved[0]["score"]) if retrieved else None,
                "retrieved_chunk_ids": json.dumps([row["chunk_id"] for row in retrieved], ensure_ascii=False),
                "retrieved_chunks": _retrieved_chunks_payload(retrieved),
                "base_retrieved_chunk_ids": json.dumps([row["chunk_id"] for row in retrieved], ensure_ascii=False),
                "mrr_at_k": retrieval_metrics["mrr_at_k"],
                "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
                "recall_at_k": retrieval_metrics["recall_at_k"],
                "ragas_recall_at_k": retrieval_metrics["ragas_recall_at_k"],
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
                "auto_flag": answer_metrics.answer_accuracy_label,
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
    answer_metrics_csv = os.path.join(run_dir, "answer_metrics.csv")
    retrieval_metrics_csv = os.path.join(run_dir, "retrieval_metrics.csv")
    diagnostics_csv = os.path.join(run_dir, "diagnostics.csv")
    results_df.to_csv(rag_results_csv, index=False)
    _drop_irrelevant_classifier_columns(
        pd.DataFrame(ranking_rows),
        classifier_type="ted_cpv",
    ).to_csv(retrieved_csv, index=False)
    _drop_irrelevant_classifier_columns(
        _drop_duplicate_metric_columns(pd.DataFrame(answer_metric_rows)),
        classifier_type="ted_cpv",
    ).to_csv(answer_metrics_csv, index=False)
    _drop_irrelevant_classifier_columns(
        _drop_duplicate_metric_columns(pd.DataFrame(retrieval_metric_rows)),
        classifier_type="ted_cpv",
    ).to_csv(retrieval_metrics_csv, index=False)
    pd.DataFrame(diagnostic_rows).to_csv(diagnostics_csv, index=False)

    aggregate_answer_metrics = summarize_answer_metrics(answer_metric_rows)
    aggregate_retrieval_metrics = summarize_retrieval_metrics(retrieval_metric_rows)
    aggregate_diagnostics = summarize_diagnostics(diagnostic_rows)
    classifier_calibration = summarize_confidence_calibration(
        result_rows,
        confidence_key="prediction_confidence",
        correct_fn=lambda row: row.get("auto_flag") == "correct",
    )

    answer_metrics_json = os.path.join(run_dir, "answer_metrics_summary.json")
    retrieval_metrics_json = os.path.join(run_dir, "retrieval_metrics_summary.json")
    diagnostics_json = os.path.join(run_dir, "diagnostics_summary.json")
    with open(answer_metrics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_answer_metrics, f, ensure_ascii=False, indent=2)
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
        "chunking_strategy": "cpv_entry",
        "retriever": retriever,
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
        "n_chunks": len(chunks),
        "n_questions": len(queries),
        "n_correct": int((results_df["auto_flag"] == "correct").sum()),
        "n_partial_correct": int((results_df.get("partial_correct") == True).sum()),
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
            "type": "ted_cpv",
            "label": classifier_label,
            "use_examples": use_examples,
            "avg_top1_top2_margin": (
                sum(top1_top2_margins) / len(top1_top2_margins) if top1_top2_margins else None
            ),
            "explanation_coverage": 0.0,
            "ranking_metrics": ranking_metrics,
            "calibration": classifier_calibration,
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
            "answer_metrics_csv": answer_metrics_csv,
            "answer_metrics_summary_json": answer_metrics_json,
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
    quality_advisor = build_run_advisor(summary, result_rows)
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
    cpv_catalog_path: str,
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

    catalog = load_cpv_catalog(cpv_catalog_path)
    catalog_chunks = build_cpv_chunks(catalog, use_examples=True)
    parent_lookup = build_parent_lookup(catalog)
    catalog_by_code = {record.code: record for record in catalog}
    label_by_code = {record.code: record.label for record in catalog}
    description_by_code = {record.code: record.description for record in catalog}

    grouped: Dict[str, Dict[str, object]] = {}
    for row_index, row in enumerate(rows, start=1):
        query_id = str(_first_present(row, ["id", "queryid", "questionid", "banfid"]) or row_index).strip()
        query_text = str(_first_present(row, ["querybanf", "query", "question", "banf"]) or "").strip()
        expected_codes = _extract_cpv_codes(
            _first_present(row, ["expectedcpv", "goldcpv", "referencecpv", "expected", "gold"])
        )
        llm_answer = str(
            _first_present(row, ["llmanswer", "answer", "generatedanswer", "finalanswer", "raganswer"]) or ""
        ).strip()

        group = grouped.setdefault(
            query_id,
            {
                "id": query_id,
                "query": query_text,
                "expected_codes": expected_codes,
                "llm_answer": llm_answer,
                "rows": [],
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
                candidate["rank"] = len(group["rows"]) + 1
            group["rows"].append(candidate)

    prediction_records: List[PredictionRecord] = []
    evaluation_items = []
    ranking_rows: List[Dict[str, object]] = []
    result_rows: List[Dict[str, object]] = []
    answer_metric_rows: List[Dict[str, object]] = []
    retrieval_metric_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []

    for group in grouped.values():
        query_id = str(group["id"])
        query_text = str(group["query"])
        expected_codes = [str(code) for code in group["expected_codes"] if str(code).strip()]
        candidates_raw = _dedupe_ranked_candidates(group["rows"])[:top_k]
        retrieved_rows_for_query = [
            _prepared_candidate_row(
                candidate_label=str(candidate["label"]),
                score=float(candidate["score"]),
                rank=rank,
                query_text=query_text,
                catalog_by_code=catalog_by_code,
                source_row=candidate["source_row"],
            )
            for rank, candidate in enumerate(candidates_raw, start=1)
            if str(candidate.get("label", "")).strip()
        ]
        normalized_candidates = [
            RankedCandidate(
                label=str(row["cpv_code"]),
                score=float(row["score"]),
                metadata={"source": "prepared_rag_results"},
            )
            for row in retrieved_rows_for_query
        ]
        ranked_labels = [candidate.label for candidate in normalized_candidates[:top_k]]
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
            k=min(top_k, len(catalog_chunks)),
        )
        diagnostics = diagnose_failure(
            answer_metrics=answer_metrics,
            retrieval_metrics=retrieval_metrics,
            llm_status="provided" if group.get("llm_answer") else "disabled",
            answer_mode=answer_mode,
        )

        for rank, row in enumerate(retrieved_rows_for_query, start=1):
            relevance_grade = retrieval_relevance_grade(item, row)
            ranking_rows.append(
                {
                    "question_id": query_id,
                    "question": query_text,
                    "rank": rank,
                    "auto_flag": answer_metrics.answer_accuracy_label,
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
                "answer_accuracy_label": answer_metrics.answer_accuracy_label,
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
                "ragas_recall_at_k": retrieval_metrics["ragas_recall_at_k"],
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
                    **_all_retrieval_metric_fields(retrieval_metrics),
                    **_all_answer_metric_fields(answer_metrics),
                    "expected_answerable": answer_metrics.expected_answerable,
                    "abstained": answer_metrics.abstained,
                    "abstention_correct": answer_metrics.abstention_correct,
                    "over_answered": answer_metrics.over_answered,
                    "false_refusal": answer_metrics.false_refusal,
                    "prediction_confidence": (
                        float(normalized_candidates[0].score) if normalized_candidates and normalized_candidates[0].score is not None else None
                    ),
                    "retrieved_chunk_ids": json.dumps([row["chunk_id"] for row in retrieved_rows_for_query], ensure_ascii=False),
                    "retrieved_chunks": _retrieved_chunks_payload(retrieved_rows_for_query),
                    "base_retrieved_chunk_ids": json.dumps([row["chunk_id"] for row in retrieved_rows_for_query], ensure_ascii=False),
                    "mrr_at_k": retrieval_metrics["mrr_at_k"],
                    "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
                    "recall_at_k": retrieval_metrics["recall_at_k"],
                    "ragas_recall_at_k": retrieval_metrics["ragas_recall_at_k"],
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
                    "auto_flag": answer_metrics.answer_accuracy_label,
                    "primary_error_reason": diagnostics.primary_error_reason,
                    "secondary_error_reason": diagnostics.secondary_error_reason,
                    "diagnostic_explanation": diagnostics.explanation,
                    "manual_flag": "",
                    "manual_comment": "",
                    "gold_cpv_label": gold_label,
                    "gold_cpv_description": gold_description,
                    "prepared_candidate_count": len(normalized_candidates),
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
    answer_metrics_df = _drop_irrelevant_classifier_columns(_drop_duplicate_metric_columns(_reorder_metric_columns(
        pd.DataFrame(answer_metric_rows),
        trailing_columns=[
            "question_id",
            "question",
            "expected_cpv_codes",
            "top1_predicted_cpv",
            "answer",
            "answer_mode",
            "llm_status",
            "llm_error",
            "runtime_retrieval_reason",
            "claim_diagnostic",
        ],
    )), classifier_type="prepared_rag_results")
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
    answer_metrics_csv = os.path.join(run_dir, "answer_metrics.csv")
    retrieval_metrics_csv = os.path.join(run_dir, "retrieval_metrics.csv")
    diagnostics_csv = os.path.join(run_dir, "diagnostics.csv")
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "prepared_source_rows.csv"), index=False)
    results_df.to_csv(rag_results_csv, index=False)
    _drop_irrelevant_classifier_columns(
        pd.DataFrame(ranking_rows),
        classifier_type="prepared_rag_results",
    ).to_csv(retrieved_csv, index=False)
    answer_metrics_df.to_csv(answer_metrics_csv, index=False)
    retrieval_metrics_df.to_csv(retrieval_metrics_csv, index=False)
    pd.DataFrame(diagnostic_rows).to_csv(diagnostics_csv, index=False)

    aggregate_answer_metrics = summarize_answer_metrics(answer_metric_rows)
    aggregate_retrieval_metrics = summarize_retrieval_metrics(retrieval_metric_rows)
    aggregate_diagnostics = summarize_diagnostics(diagnostic_rows)
    classifier_calibration = summarize_confidence_calibration(
        result_rows,
        confidence_key="prediction_confidence",
        correct_fn=lambda row: row.get("auto_flag") == "correct",
    )
    ranking_metrics = evaluate_ranked_predictions(
        evaluation_items,
        prediction_records,
        top_k=top_k,
        distance_fn=cpv_structural_distance,
    )

    answer_metrics_json = os.path.join(run_dir, "answer_metrics_summary.json")
    retrieval_metrics_json = os.path.join(run_dir, "retrieval_metrics_summary.json")
    diagnostics_json = os.path.join(run_dir, "diagnostics_summary.json")
    with open(answer_metrics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_answer_metrics, f, ensure_ascii=False, indent=2)
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
        "top_k": top_k,
        "reranker": {"enabled": False, "type": None, "top_n": 0, "weight": 0.0},
        "n_chunks": len(catalog_chunks),
        "n_questions": len(grouped),
        "n_correct": int((results_df["auto_flag"] == "correct").sum()),
        "n_partial_correct": int((results_df.get("partial_correct") == True).sum()),
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
            "ranking_metrics": ranking_metrics,
            "calibration": classifier_calibration,
            "input_contract": {
                "current_columns": ["ID", "Query (BANF)", "Expected CPV", "Predicted CPV", "RRF Score"],
                "current_top_k_columns": [
                    "Predicted CPV #1",
                    "Vector Score #1",
                    "Predicted CPV #2",
                    "Vector Score #2",
                    "Predicted CPV #3",
                    "Vector Score #3",
                ],
                "supported_future_columns": [
                    "Rank",
                    "Score",
                    "Confidence",
                    "LLM Answer",
                    "Answer",
                    "Chunks",
                    "Chunk Text",
                    "Retrieved Chunks",
                    "Chunk ID",
                ],
                "top_k_layout": "Multiple rows with the same ID are treated as ranked candidates for one query.",
            },
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
            "answer_metrics_csv": answer_metrics_csv,
            "answer_metrics_summary_json": answer_metrics_json,
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
    quality_advisor = build_run_advisor(summary, result_rows)
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
    cpv_catalog_path: str,
    queries_path: str,
    top_k: int,
    classifier_label: str,
    run_dir: str,
    create_visualization: bool,
    create_showcase: bool,
) -> Dict[str, object]:
    import pandas as pd

    catalog = load_cpv_catalog(cpv_catalog_path)
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
                    metadata={k: v for k, v in candidate.items() if k not in {"label", "cpv_code", "answer", "id", "code", "score", "confidence", "probability"}},
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

        for rank, row in enumerate(retrieved_rows_for_query, start=1):
            relevance_grade = retrieval_relevance_grade(item, row)
            ranking_rows.append(
                {
                    "question_id": query.id,
                    "question": query.query,
                    "rank": rank,
                    "auto_flag": answer_metrics.answer_accuracy_label,
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
                "answer_accuracy_label": answer_metrics.answer_accuracy_label,
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
                "ragas_recall_at_k": retrieval_metrics["ragas_recall_at_k"],
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
                    "program_id": "cpv",
                    "program_name": "CPV",
                    "doc_id": query.gold_cpv_code,
                    "answer_scope": json.dumps({}, ensure_ascii=False),
                    "evaluation_scope": json.dumps({"doc_id": [query.gold_cpv_code]}, ensure_ascii=False),
                    "gold_answer": item["gold_answer"],
                    "expected_keywords": json.dumps(item["expected_keywords"], ensure_ascii=False),
                    **classification_metrics,
                    **_all_retrieval_metric_fields(retrieval_metrics),
                    **_all_answer_metric_fields(answer_metrics),
                    "expected_answerable": answer_metrics.expected_answerable,
                    "abstained": answer_metrics.abstained,
                    "abstention_correct": answer_metrics.abstention_correct,
                    "over_answered": answer_metrics.over_answered,
                    "false_refusal": answer_metrics.false_refusal,
                    "prediction_confidence": (
                        float(normalized_candidates[0].score) if normalized_candidates and normalized_candidates[0].score is not None else None
                    ),
                    "retrieved_chunk_ids": json.dumps([row["chunk_id"] for row in retrieved_rows_for_query], ensure_ascii=False),
                    "retrieved_chunks": _retrieved_chunks_payload(retrieved_rows_for_query),
                    "base_retrieved_chunk_ids": json.dumps([row["chunk_id"] for row in retrieved_rows_for_query], ensure_ascii=False),
                    "mrr_at_k": retrieval_metrics["mrr_at_k"],
                    "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
                    "recall_at_k": retrieval_metrics["recall_at_k"],
                    "ragas_recall_at_k": retrieval_metrics["ragas_recall_at_k"],
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
                    "auto_flag": answer_metrics.answer_accuracy_label,
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
    answer_metrics_csv = os.path.join(run_dir, "answer_metrics.csv")
    retrieval_metrics_csv = os.path.join(run_dir, "retrieval_metrics.csv")
    diagnostics_csv = os.path.join(run_dir, "diagnostics.csv")
    results_df.to_csv(rag_results_csv, index=False)
    _drop_irrelevant_classifier_columns(
        pd.DataFrame(ranking_rows),
        classifier_type="api_classifier",
    ).to_csv(retrieved_csv, index=False)
    _drop_irrelevant_classifier_columns(
        _drop_duplicate_metric_columns(pd.DataFrame(answer_metric_rows)),
        classifier_type="api_classifier",
    ).to_csv(answer_metrics_csv, index=False)
    _drop_irrelevant_classifier_columns(
        _drop_duplicate_metric_columns(pd.DataFrame(retrieval_metric_rows)),
        classifier_type="api_classifier",
    ).to_csv(retrieval_metrics_csv, index=False)
    pd.DataFrame(diagnostic_rows).to_csv(diagnostics_csv, index=False)

    aggregate_answer_metrics = summarize_answer_metrics(answer_metric_rows)
    aggregate_retrieval_metrics = summarize_retrieval_metrics(retrieval_metric_rows)
    aggregate_diagnostics = summarize_diagnostics(diagnostic_rows)
    classifier_calibration = summarize_confidence_calibration(
        result_rows,
        confidence_key="prediction_confidence",
        correct_fn=lambda row: row.get("auto_flag") == "correct",
    )

    answer_metrics_json = os.path.join(run_dir, "answer_metrics_summary.json")
    retrieval_metrics_json = os.path.join(run_dir, "retrieval_metrics_summary.json")
    diagnostics_json = os.path.join(run_dir, "diagnostics_summary.json")
    with open(answer_metrics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_answer_metrics, f, ensure_ascii=False, indent=2)
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
        "n_partial_correct": int((results_df.get("partial_correct") == True).sum()),
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
            "response_contract": {
                "required": ["id", "query", "predictions[].label|cpv_code|answer", "predictions[].score|confidence"],
                "optional": [
                    "answer",
                    "explanation",
                    "metadata",
                    "confidence",
                    "retrieved_contexts",
                    "latency_ms",
                ],
            },
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
            "answer_metrics_csv": answer_metrics_csv,
            "answer_metrics_summary_json": answer_metrics_json,
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
    quality_advisor = build_run_advisor(summary, result_rows)
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
