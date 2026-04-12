from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from rag_eval.chunking import build_chunks
from rag_eval.llm import generate_answer_with_llm
from rag_eval.metrics import (
    diagnose_failure,
    evaluate_answer_metrics,
    evaluate_retrieval_metrics,
    keyword_extractive_answer,
    summarize_answer_metrics,
    summarize_diagnostics,
    summarize_retrieval_metrics,
)
from rag_eval.models import LLMConfig, Paragraph, Section
from rag_eval.retrieval import build_retriever, retrieve_top_k


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_csv_list(raw_value: str) -> List[int]:
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def list_field(value: object) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def question_metadata_filter(item: Dict, *, include_document: bool) -> Dict[str, object]:
    metadata_filter: Dict[str, object] = {}
    fields = [
        ("program_id", "program_ids"),
        ("program_name", "program_names"),
    ]
    if include_document:
        fields.extend(
            [
                ("doc_id", "doc_ids"),
                ("doc_path", "doc_paths"),
            ]
        )
    for singular, plural in fields:
        values = list_field(item.get(singular)) + list_field(item.get(plural))
        if values:
            metadata_filter[singular] = values
    return metadata_filter


def normalize_metadata_value(value: object) -> str:
    normalized = str(value).casefold()
    normalized = normalized.replace("&", "and").replace("/", " ")
    normalized = "".join(char for char in normalized if char.isascii())
    normalized = "".join(char if char.isalnum() else " " for char in normalized)
    aliases = {
        "ba": "bachelor",
        "bpo": "bachelor",
        "ma": "master",
        "mpo": "master",
        "sciences": "science",
    }
    return " ".join(aliases.get(token, token) for token in normalized.split())


def metadata_value_matches(actual: object, expected: object) -> bool:
    actual_norm = normalize_metadata_value(actual)
    expected_norm = normalize_metadata_value(expected)
    return (
        actual_norm == expected_norm
        or actual_norm in expected_norm
        or expected_norm in actual_norm
    )


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


def build_run_dir(base_output_dir: str, run_name: Optional[str]) -> str:
    run_id = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, run_id)
    ensure_dir(run_dir)
    return run_dir


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
) -> Dict:
    import pandas as pd

    experiment_slug = f"{strategy}_{retriever_type}_size{chunk_size}_overlap{chunk_overlap}"
    experiment_dir = os.path.join(run_dir, experiment_slug)
    ensure_dir(experiment_dir)

    chunks = build_chunks(sections, paragraphs, strategy, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError(f"No chunks generated for strategy '{strategy}'.")

    chunks_df = pd.DataFrame(chunks)
    chunks_csv = os.path.join(experiment_dir, "chunks.csv")
    chunks_df.to_csv(chunks_csv, index=False)

    retriever_state = build_retriever(chunks, retriever_type, embedding_model)
    faiss_path: Optional[str] = None
    if retriever_type == "dense":
        import faiss

        faiss_path = os.path.join(experiment_dir, "index.faiss")
        faiss.write_index(retriever_state["index"], faiss_path)
    elif retriever_type == "hybrid":
        import faiss

        faiss_path = os.path.join(experiment_dir, "dense_index.faiss")
        faiss.write_index(retriever_state["dense"]["index"], faiss_path)

    results: List[Dict] = []
    retrieved_rows: List[Dict] = []
    answer_metric_rows: List[Dict] = []
    retrieval_metric_rows: List[Dict] = []
    diagnostic_rows: List[Dict] = []
    for item in questions:
        answer_metadata_filter = question_metadata_filter(item, include_document=False)
        evaluation_metadata_filter = question_metadata_filter(item, include_document=True)
        candidate_chunks = filter_chunks_by_metadata(chunks, evaluation_metadata_filter)
        if evaluation_metadata_filter and not candidate_chunks:
            raise ValueError(
                f"Question {item['id']} has evaluation metadata filter {evaluation_metadata_filter}, "
                "but no chunks match it."
            )
        answer_scope_chunks = filter_chunks_by_metadata(chunks, answer_metadata_filter)
        retrieved = retrieve_top_k(
            query=item["question"],
            retriever_state=retriever_state,
            chunks=chunks,
            k=min(top_k, len(answer_scope_chunks or chunks)),
            hybrid_alpha=hybrid_alpha,
            metadata_filter=answer_metadata_filter,
        )
        llm_result = generate_answer_with_llm(item["question"], retrieved, llm_config)
        answer = llm_result.answer
        answer_mode = "llm_grounded"
        if answer is None:
            answer = keyword_extractive_answer(item["question"], retrieved)
            answer_mode = "extractive_fallback"
        answer_metrics = evaluate_answer_metrics(item, answer, retrieved)
        retrieval_metrics = evaluate_retrieval_metrics(
            item=item,
            retrieved=retrieved,
            candidate_chunks=candidate_chunks,
            k=min(top_k, len(candidate_chunks)),
        )
        auto_flag = answer_metrics.answer_accuracy_label
        diagnostics = diagnose_failure(
            answer_metrics=answer_metrics,
            retrieval_metrics=retrieval_metrics,
            llm_status=llm_result.status,
            answer_mode=answer_mode,
        )

        for rank, row in enumerate(retrieved, start=1):
            retrieved_rows.append(
                {
                    "question_id": item["id"],
                    "question": item["question"],
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
                    "doc_id": row["doc_id"],
                    "doc_path": row.get("doc_path", row["doc_id"]),
                    "program_id": row.get("program_id", ""),
                    "program_name": row.get("program_name", ""),
                    "section_id": row["section_id"],
                    "title": row["title"],
                    "chunking_strategy": row["chunking_strategy"],
                    "source_type": row["source_type"],
                    "text": row["text"],
                }
            )

        answer_metric_rows.append(
            {
                "question_id": item["id"],
                "question": item["question"],
                "program_id": item.get("program_id", ""),
                "program_name": item.get("program_name", ""),
                "doc_id": item.get("doc_id", ""),
                "answer_scope": json.dumps(answer_metadata_filter, ensure_ascii=False),
                "evaluation_scope": json.dumps(evaluation_metadata_filter, ensure_ascii=False),
                "answer_accuracy_label": answer_metrics.answer_accuracy_label,
                "llm_used": llm_result.used,
                "llm_status": llm_result.status,
                "llm_error": llm_result.error,
                "gold_answer_overlap": answer_metrics.gold_answer_overlap,
                "answer_gold_support": answer_metrics.answer_gold_support,
                "proxy_faithfulness": answer_metrics.proxy_faithfulness,
                "proxy_context_relevance": answer_metrics.proxy_context_relevance,
                "answer_has_gold_substring": answer_metrics.answer_has_gold_substring,
            }
        )
        retrieval_metric_rows.append(
            {
                "question_id": item["id"],
                "question": item["question"],
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
        diagnostic_rows.append(
            {
                "question_id": item["id"],
                "question": item["question"],
                "program_id": item.get("program_id", ""),
                "program_name": item.get("program_name", ""),
                "doc_id": item.get("doc_id", ""),
                "answer_scope": json.dumps(answer_metadata_filter, ensure_ascii=False),
                "evaluation_scope": json.dumps(evaluation_metadata_filter, ensure_ascii=False),
                "primary_error_reason": diagnostics.primary_error_reason,
                "secondary_error_reason": diagnostics.secondary_error_reason,
                "explanation": diagnostics.explanation,
            }
        )

        results.append(
            {
                "question_id": item["id"],
                "question": item["question"],
                "program_id": item.get("program_id", ""),
                "program_name": item.get("program_name", ""),
                "doc_id": item.get("doc_id", ""),
                "answer_scope": json.dumps(answer_metadata_filter, ensure_ascii=False),
                "evaluation_scope": json.dumps(evaluation_metadata_filter, ensure_ascii=False),
                "gold_answer": item.get("gold_answer", ""),
                "expected_keywords": json.dumps(item.get("expected_keywords", []), ensure_ascii=False),
                "retrieved_chunk_ids": json.dumps(
                    [row["chunk_id"] for row in retrieved], ensure_ascii=False
                ),
                "mrr_at_k": retrieval_metrics["mrr_at_k"],
                "ndcg_at_k": retrieval_metrics["ndcg_at_k"],
                "recall_at_k": retrieval_metrics["recall_at_k"],
                "ragas_recall_at_k": retrieval_metrics["ragas_recall_at_k"],
                "target_doc_retrieved_at_k": retrieval_metrics["target_doc_retrieved_at_k"],
                "first_target_doc_rank": retrieval_metrics["first_target_doc_rank"],
                "n_retrieved_target_doc_chunks": retrieval_metrics[
                    "n_retrieved_target_doc_chunks"
                ],
                "gold_answer_overlap": answer_metrics.gold_answer_overlap,
                "answer_gold_support": answer_metrics.answer_gold_support,
                "proxy_faithfulness": answer_metrics.proxy_faithfulness,
                "proxy_context_relevance": answer_metrics.proxy_context_relevance,
                "llm_status": llm_result.status,
                "llm_error": llm_result.error,
                "answer": answer,
                "answer_mode": answer_mode,
                "auto_flag": auto_flag,
                "primary_error_reason": diagnostics.primary_error_reason,
                "secondary_error_reason": diagnostics.secondary_error_reason,
                "diagnostic_explanation": diagnostics.explanation,
                "manual_flag": "",
                "manual_comment": "",
            }
        )

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        max_manual = min(10, len(results_df))
        results_df.loc[: max_manual - 1, "manual_flag"] = "reviewed"
        results_df.loc[: max_manual - 1, "manual_comment"] = "fill_correct_or_incorrect"

    results_csv = os.path.join(experiment_dir, "rag_results.csv")
    results_df.to_csv(results_csv, index=False)

    retrieved_df = pd.DataFrame(retrieved_rows)
    retrieved_csv = os.path.join(experiment_dir, "retrieved_chunks.csv")
    retrieved_df.to_csv(retrieved_csv, index=False)

    answer_metrics_df = pd.DataFrame(answer_metric_rows)
    answer_metrics_csv = os.path.join(experiment_dir, "answer_metrics.csv")
    answer_metrics_df.to_csv(answer_metrics_csv, index=False)
    aggregate_answer_metrics = summarize_answer_metrics(answer_metric_rows)
    answer_metrics_json = os.path.join(experiment_dir, "answer_metrics_summary.json")
    with open(answer_metrics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_answer_metrics, f, ensure_ascii=False, indent=2)

    retrieval_metrics_df = pd.DataFrame(retrieval_metric_rows)
    retrieval_metrics_csv = os.path.join(experiment_dir, "retrieval_metrics.csv")
    retrieval_metrics_df.to_csv(retrieval_metrics_csv, index=False)
    aggregate_retrieval_metrics = summarize_retrieval_metrics(retrieval_metric_rows)
    retrieval_metrics_json = os.path.join(experiment_dir, "retrieval_metrics_summary.json")
    with open(retrieval_metrics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_retrieval_metrics, f, ensure_ascii=False, indent=2)

    diagnostics_df = pd.DataFrame(diagnostic_rows)
    diagnostics_csv = os.path.join(experiment_dir, "diagnostics.csv")
    diagnostics_df.to_csv(diagnostics_csv, index=False)
    aggregate_diagnostics = summarize_diagnostics(diagnostic_rows)
    diagnostics_json = os.path.join(experiment_dir, "diagnostics_summary.json")
    with open(diagnostics_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_diagnostics, f, ensure_ascii=False, indent=2)

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
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "hybrid_alpha": hybrid_alpha if retriever_type == "hybrid" else None,
        "top_k": top_k,
        "n_chunks": len(chunks),
        "n_questions": len(questions),
        "n_correct": int((results_df["auto_flag"] == "correct").sum()),
        "n_incorrect": int((results_df["auto_flag"] == "incorrect").sum()),
        "answer_metrics": aggregate_answer_metrics,
        "retrieval_metrics": aggregate_retrieval_metrics,
        "diagnostics": aggregate_diagnostics,
        "llm": {
            "enabled": llm_config.enabled,
            "model": llm_config.model if llm_config.enabled else None,
            "answer_generation": llm_config.enabled,
        },
        "outputs": {
            "chunks_csv": chunks_csv,
            "faiss_index": faiss_path,
            "rag_results_csv": results_csv,
            "retrieved_chunks_csv": retrieved_csv,
            "answer_metrics_csv": answer_metrics_csv,
            "answer_metrics_summary_json": answer_metrics_json,
            "retrieval_metrics_csv": retrieval_metrics_csv,
            "retrieval_metrics_summary_json": retrieval_metrics_json,
            "diagnostics_csv": diagnostics_csv,
            "diagnostics_summary_json": diagnostics_json,
            "error_report_md": error_report_md,
        },
    }
    with open(os.path.join(experiment_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
