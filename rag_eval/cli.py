from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from typing import Dict, List, Tuple

from rag_eval.experiment import build_run_dir, ensure_dir, parse_csv_list, run_single_experiment
from rag_eval.io import extract_paragraphs, load_questions, parse_pdf_sections, resolve_doc_paths
from rag_eval.metrics import build_recommendation, rank_experiments, score_experiment
from rag_eval.models import LLMConfig, Section
from rag_eval.retrieval import DEFAULT_EMBEDDING_MODEL


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline RAG pipeline with reusable chunking experiments."
    )
    parser.add_argument("--docs", required=True, help="PDF path, glob, or comma-separated list.")
    parser.add_argument(
        "--questions",
        default="outputs/questions_enriched.json",
        help="Path to questions JSON. Defaults to outputs/questions_enriched.json.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Base directory for run artifacts.")
    parser.add_argument("--run-name", default=None, help="Optional stable run directory name.")
    parser.add_argument(
        "--chunking",
        default="fixed_words",
        choices=["fixed_words", "fixed_tokens", "by_section", "by_paragraph", "auto"],
        help="Chunking strategy or 'auto' to evaluate several strategies.",
    )
    parser.add_argument("--chunk-size", type=int, default=450, help="Chunk size for fixed strategies.")
    parser.add_argument(
        "--overlap", type=int, default=60, help="Chunk overlap for fixed strategies."
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many chunks to retrieve.")
    parser.add_argument(
        "--retriever",
        default=None,
        choices=["tfidf", "bm25", "dense", "hybrid", "auto"],
        help="Retriever backend or 'auto' to evaluate several retrievers.",
    )
    parser.add_argument(
        "--embedding-backend",
        default="tfidf",
        choices=["tfidf", "sentence-transformers"],
        help="Deprecated alias for retriever selection: tfidf -> tfidf, sentence-transformers -> dense.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Model used only with the sentence-transformers backend.",
    )
    parser.add_argument(
        "--auto-chunk-sizes",
        default="256,450",
        help="Comma-separated chunk sizes used when --chunking=auto.",
    )
    parser.add_argument(
        "--auto-overlaps",
        default="0,60",
        help="Comma-separated overlaps used when --chunking=auto.",
    )
    parser.add_argument(
        "--auto-retrievers",
        default="tfidf,bm25,dense,hybrid",
        help="Comma-separated retrievers used when --retriever=auto.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Weight for dense scores in hybrid retrieval; BM25 gets (1-alpha).",
    )
    parser.add_argument(
        "--weight-answer",
        type=float,
        default=0.45,
        help="Weight of answer faithfulness proxy in experiment ranking.",
    )
    parser.add_argument(
        "--weight-correctness",
        type=float,
        default=0.3,
        help="Weight of answer correctness ratio in experiment ranking.",
    )
    parser.add_argument(
        "--weight-retrieval",
        type=float,
        default=0.25,
        help="Weight of retrieval quality in experiment ranking.",
    )
    parser.add_argument(
        "--llm-enable",
        action="store_true",
        help="Enable OpenAI-based answer generation from retrieved chunks.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="OpenAI model used to generate answers from retrieved chunks.",
    )
    parser.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name containing the OpenAI API key.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    import pandas as pd

    ensure_dir(args.output_dir)
    run_dir = build_run_dir(args.output_dir, args.run_name)
    doc_paths = resolve_doc_paths(args.docs)
    questions = load_questions(args.questions)
    resolved_retriever = args.retriever
    if resolved_retriever is None:
        resolved_retriever = "dense" if args.embedding_backend == "sentence-transformers" else "tfidf"
    llm_config = LLMConfig(
        enabled=args.llm_enable,
        model=args.llm_model,
        api_key_env=args.openai_api_key_env,
    )

    all_sections: List[Section] = []
    raw_parts: List[str] = []
    for path in doc_paths:
        raw_text, sections = parse_pdf_sections(path)
        raw_parts.append(f"===== {os.path.basename(path)} =====\n{raw_text}")
        all_sections.extend(sections)

    paragraphs = extract_paragraphs(all_sections)

    raw_path = os.path.join(run_dir, "raw_documents.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(raw_parts))

    sections_csv = os.path.join(run_dir, "sections.csv")
    pd.DataFrame([asdict(section) for section in all_sections]).to_csv(sections_csv, index=False)

    paragraphs_csv = os.path.join(run_dir, "paragraphs.csv")
    pd.DataFrame([asdict(paragraph) for paragraph in paragraphs]).to_csv(
        paragraphs_csv, index=False
    )

    if args.chunking == "auto":
        strategies = ["fixed_words", "fixed_tokens", "by_section", "by_paragraph"]
        auto_sizes = parse_csv_list(args.auto_chunk_sizes)
        auto_overlaps = parse_csv_list(args.auto_overlaps)
        experiments: List[Tuple[str, int, int]] = []
        for strategy in strategies:
            if strategy in {"fixed_words", "fixed_tokens"}:
                for chunk_size in auto_sizes:
                    for overlap in auto_overlaps:
                        experiments.append((strategy, chunk_size, overlap))
            else:
                experiments.append((strategy, 0, 0))
    else:
        experiments = [(args.chunking, args.chunk_size, args.overlap)]

    if resolved_retriever == "auto":
        retriever_types = [item.strip() for item in args.auto_retrievers.split(",") if item.strip()]
    else:
        retriever_types = [resolved_retriever]

    experiment_summaries: List[Dict] = []
    for strategy, chunk_size, overlap in experiments:
        for retriever_type in retriever_types:
            experiment_summaries.append(
                run_single_experiment(
                    sections=all_sections,
                    paragraphs=paragraphs,
                    questions=questions,
                    run_dir=run_dir,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    top_k=args.top_k,
                    retriever_type=retriever_type,
                    embedding_model=args.embedding_model,
                    hybrid_alpha=args.hybrid_alpha,
                    llm_config=llm_config,
                )
            )

    ranking_weights = {
        "answer": args.weight_answer,
        "correctness": args.weight_correctness,
        "retrieval": args.weight_retrieval,
    }
    ranked_experiments = rank_experiments(experiment_summaries, ranking_weights)
    best_experiment = ranked_experiments[0] if ranked_experiments else None

    ranking_csv = os.path.join(run_dir, "experiment_ranking.csv")
    pd.DataFrame(ranked_experiments).to_csv(ranking_csv, index=False)

    best_config_json = os.path.join(run_dir, "best_config.json")
    best_config_md = os.path.join(run_dir, "best_config.md")
    run_rag_results_csv = os.path.join(run_dir, "rag_results.csv")
    latest_rag_results_csv = os.path.join(args.output_dir, "rag_results.csv")
    recommendation = None
    if best_experiment is not None:
        best_summary = next(
            summary
            for summary in experiment_summaries
            if summary["experiment"] == best_experiment["experiment"]
        )
        best_score = score_experiment(best_summary, ranking_weights)
        recommendation = build_recommendation(best_summary, best_score)
        best_config_payload = {
            "experiment": best_summary["experiment"],
            "chunking_strategy": best_summary["chunking_strategy"],
            "retriever": best_summary["retriever"],
            "chunk_size": best_summary["chunk_size"],
            "chunk_overlap": best_summary["chunk_overlap"],
            "top_k": best_summary["top_k"],
            "hybrid_alpha": best_summary["hybrid_alpha"],
            "score": best_score,
            "recommendation": recommendation,
            "weights": ranking_weights,
        }
        with open(best_config_json, "w", encoding="utf-8") as f:
            json.dump(best_config_payload, f, ensure_ascii=False, indent=2)
        with open(best_config_md, "w", encoding="utf-8") as f:
            f.write("# Best Configuration\n\n")
            f.write(f"{recommendation}\n")
        best_rag_results_csv = best_summary["outputs"]["rag_results_csv"]
        shutil.copyfile(best_rag_results_csv, run_rag_results_csv)
        shutil.copyfile(best_rag_results_csv, latest_rag_results_csv)
    else:
        with open(best_config_json, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        with open(best_config_md, "w", encoding="utf-8") as f:
            f.write("# Best Configuration\n\nNo experiments were executed.\n")
        best_rag_results_csv = None

    run_summary = {
        "run_dir": run_dir,
        "documents": doc_paths,
        "questions_path": args.questions,
        "n_documents": len(doc_paths),
        "n_sections": len(all_sections),
        "n_paragraphs": len(paragraphs),
        "n_questions": len(questions),
        "chunking_mode": args.chunking,
        "retriever_mode": resolved_retriever,
        "llm": {
            "enabled": llm_config.enabled,
            "model": llm_config.model if llm_config.enabled else None,
            "answer_generation": llm_config.enabled,
        },
        "experiments": experiment_summaries,
        "ranking_weights": ranking_weights,
        "best_experiment": best_experiment,
        "recommendation": recommendation,
        "outputs": {
            "raw_documents_txt": raw_path,
            "sections_csv": sections_csv,
            "paragraphs_csv": paragraphs_csv,
            "experiment_ranking_csv": ranking_csv,
            "best_config_json": best_config_json,
            "best_config_md": best_config_md,
            "questions_json": args.questions,
            "rag_results_csv": run_rag_results_csv if best_experiment is not None else None,
            "latest_rag_results_csv": latest_rag_results_csv if best_experiment is not None else None,
        },
        "question_schema": {
            "required": ["question"],
            "recommended": [
                "id",
                "gold_answer",
                "expected_keywords",
            ],
        },
        "notes": [
            "Pipeline includes CLI, normalized ingestion, multiple chunking strategies, multiple retrievers, grounded answer generation, answer metrics, diagnostics, and experiment ranking.",
            "Fixed_tokens currently uses whitespace tokenization over paragraphs as an offline-friendly approximation.",
            "Retrieval metrics are omitted because the simplified question schema no longer includes gold chunk or section annotations.",
            "Answer metrics remain heuristic; LLM is used only for grounded answer generation after retrieval.",
            "Questions are expected to come from a prebuilt enriched dataset such as outputs/questions_enriched.json.",
        ],
    }

    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
