from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from typing import Dict, List, Tuple

from rag_eval.classifiers.evaluation import (
    evaluate_api_ted_cpv_classifier,
    evaluate_local_ted_cpv_classifier,
    evaluate_prepared_rag_results_classifier,
)
from rag_eval.retrieval.experiment import build_run_dir, ensure_dir, parse_csv_list, run_single_experiment
from rag_eval.data.io import extract_paragraphs, load_questions, parse_pdf_sections, resolve_doc_paths
from rag_eval.evaluation.design_attribution import write_design_attribution_artifacts
from rag_eval.evaluation.metrics import build_recommendation, rank_experiments, score_experiment
from rag_eval.core.models import LLMConfig, Section
from rag_eval.retrieval.engines import DEFAULT_EMBEDDING_MODEL
from rag_eval.reporting.visualization import write_run_showcase_index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline RAG pipeline with reusable chunking experiments."
    )
    parser.add_argument(
        "--mode",
        choices=["classifier", "sweep"],
        default="sweep",
        help="Run one concrete local RAG classifier or sweep multiple retrieval/chunking variants.",
    )
    parser.add_argument(
        "--classifier-type",
        choices=["document_qa", "examination_regulations", "ted_cpv", "api_classifier", "prepared_rag_results"],
        default="document_qa",
        help="Classifier to evaluate when --mode=classifier. document_qa is the file-based RAG QA flow; examination_regulations is kept as a backwards-compatible alias.",
    )
    parser.add_argument(
        "--docs",
        default="data/files/**/*.pdf*",
        help="PDF path, directory, glob, or comma-separated list. Defaults to data/files/**/*.pdf*.",
    )
    parser.add_argument(
        "--docs-root",
        default="data/files",
        help="Root directory used to infer program names and stable relative doc_ids.",
    )
    parser.add_argument(
        "--questions",
        default="data/questions_by_file.json",
        help="Path to questions JSON. Defaults to data/questions_by_file.json.",
    )
    parser.add_argument(
        "--cpv-catalog",
        default="data/cpv_ted_train_catalog.csv",
        help="CPV catalog CSV used by the local TED/CPV classifier.",
    )
    parser.add_argument(
        "--cpv-queries",
        default="data/cpv_ted_test_queries.json",
        help="TED/CPV test questions used when --classifier-type=ted_cpv.",
    )
    parser.add_argument(
        "--prepared-results",
        default="data/eval_dataset.xlsx",
        help="Excel file with already prepared RAG/classifier results used when --classifier-type=prepared_rag_results.",
    )
    parser.add_argument(
        "--cpv-use-examples",
        action="store_true",
        help="Append real TED examples to CPV label entries before ranking.",
    )
    parser.add_argument(
        "--api-eval-case",
        choices=["ted_cpv"],
        default="ted_cpv",
        help="Evaluation dataset/schema used for the API-backed classifier.",
    )
    parser.add_argument(
        "--api-classifier-url",
        default=None,
        help="HTTP endpoint for the external classifier. The evaluator will send query payloads and expect ranked predictions back as JSON.",
    )
    parser.add_argument(
        "--api-auth-token-env",
        default="API_CLASSIFIER_TOKEN",
        help="Environment variable used for a Bearer token when calling the external classifier API.",
    )
    parser.add_argument(
        "--api-timeout-seconds",
        type=float,
        default=30.0,
        help="Timeout for each API classifier request.",
    )
    parser.add_argument(
        "--api-extra-headers-json",
        default=None,
        help="Optional JSON object with additional HTTP headers for the external classifier API.",
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
        "--sweep-configs-json",
        default=None,
        help="Optional JSON list of explicit sweep configs. Each item may contain chunking, chunk_size, overlap, retriever, and hybrid_alpha. When set, configs run in list order.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Weight for dense scores in hybrid retrieval; BM25 gets (1-alpha).",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=0,
        help="Optional second-stage reranking window. If > 1, rerank the first N retrieved candidates with a lexical overlap signal.",
    )
    parser.add_argument(
        "--rerank-weight",
        type=float,
        default=0.25,
        help="How strongly the lightweight reranker influences final ranking inside --rerank-top-n.",
    )
    parser.add_argument(
        "--cross-encoder-rerank",
        action="store_true",
        help="Rerank top CPV candidates with a cross-encoder after retrieval and optional KG.",
    )
    parser.add_argument(
        "--cross-encoder-model",
        default=None,
        help="Cross-encoder model id (default: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1).",
    )
    parser.add_argument(
        "--cross-encoder-top-n",
        type=int,
        default=10,
        help="How many top candidates to pass to the cross-encoder reranker.",
    )
    parser.add_argument(
        "--create-strategy-visualization",
        action="store_true",
        help="Write an SVG score-profile visualization for each experiment/strategy.",
    )
    parser.add_argument(
        "--create-strategy-showcase",
        action="store_true",
        help="Write a showcase report with multiple charts and improvement summary for each experiment/strategy.",
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
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the answer-generation LLM.",
    )
    parser.add_argument(
        "--judge-enable",
        action="store_true",
        help="Enable an LLM/NLI judge for claim-level support, contradiction, and completeness metrics.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="OpenAI model used for claim-level judging. Defaults to --llm-model.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the claim-level judge.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=["extractive", "grounded_llm", "cite_first", "claim_checklist"],
        default="cite_first",
        help="Answer synthesis mode for document QA.",
    )
    parser.add_argument(
        "--context-mode",
        choices=["ranked", "kg_first", "kg_organized", "group_by_doc", "dedupe_section"],
        default="dedupe_section",
        help="Context assembly mode for document QA.",
    )
    parser.add_argument("--max-context-chunks", type=int, default=None, help="Limit chunks passed to answer/evaluation.")
    parser.add_argument("--max-context-chars", type=int, default=None, help="Limit total context characters.")
    parser.add_argument(
        "--query-augmentation",
        choices=["none", "llm", "hyde"],
        default="none",
        help="Use an LLM to expand retrieval queries before retrieval; hyde generates a hypothetical source passage.",
    )
    parser.add_argument(
        "--query-augmentation-max-terms",
        type=int,
        default=8,
        help="Maximum added terms requested from the query-augmentation LLM.",
    )
    parser.add_argument("--decision-min-confidence", type=float, default=0.0)
    parser.add_argument("--decision-min-context-claim-recall", type=float, default=0.0)
    parser.add_argument("--decision-min-grounded-claim-ratio", type=float, default=1.0)
    parser.add_argument(
        "--disable-runtime-retrieval-evaluator",
        action="store_true",
        help="Disable gold-free runtime retrieval quality signals.",
    )
    parser.add_argument(
        "--abstain-on-weak-evidence",
        action="store_true",
        help="Use the runtime retrieval evaluator as a generation gate and abstain when evidence is weak or missing.",
    )
    parser.add_argument(
        "--self-rag-retry-on-weak-evidence",
        action="store_true",
        help="When runtime retrieval is weak, rewrite the query and retrieve again before answer generation.",
    )
    parser.add_argument(
        "--self-rag-retry-max-attempts",
        type=int,
        default=1,
        help="Maximum Self-RAG-style query rewrite/retrieval retries for weak evidence.",
    )
    parser.add_argument(
        "--self-rag-critique",
        action="store_true",
        help="Run an LLM self-critique/revision pass after answer generation and before scoring.",
    )
    parser.add_argument(
        "--kg-enable",
        action="store_true",
        help="Build a provenance-aware knowledge graph, use it for graph-augmented retrieval, and write KG metrics.",
    )
    parser.add_argument(
        "--kg-graph-weight",
        type=float,
        default=0.35,
        help="Weight of graph signals when KG retrieval/reranking is enabled.",
    )
    parser.add_argument(
        "--kg-profile",
        choices=["conservative", "balanced", "exploratory", "ppr_only", "direct_only", "selection"],
        default="exploratory",
        help="KG retrieval preset for research comparisons.",
    )
    parser.add_argument(
        "--kg-profiles",
        default="",
        help="Comma-separated KG profiles to compare in sweep mode when --kg-enable is set.",
    )
    parser.add_argument(
        "--kg-algorithm",
        choices=["direct", "ppr", "ppr_direct"],
        default=None,
        help="Override the KG traversal algorithm from --kg-profile.",
    )
    parser.add_argument(
        "--kg-max-added-chunks",
        type=int,
        default=None,
        help="Override max graph-only chunks added to the retrieved context.",
    )
    parser.add_argument(
        "--kg-ppr-iterations",
        type=int,
        default=None,
        help="Override Personalized PageRank-style propagation iterations.",
    )
    parser.add_argument(
        "--kg-ppr-damping",
        type=float,
        default=None,
        help="Override Personalized PageRank-style damping factor.",
    )
    parser.add_argument(
        "--kg-quality-threshold",
        type=float,
        default=None,
        help="Override minimum graph-only chunk quality factor.",
    )
    parser.add_argument(
        "--kg-intent-weight",
        type=float,
        default=None,
        help="Override intent-score contribution to KG candidate ranking.",
    )
    parser.add_argument(
        "--kg-ablation-edge-dropouts",
        default="",
        help="Comma-separated edge dropout rates for KG incompleteness testing, e.g. 0.1,0.3,0.5.",
    )
    return parser


def prune_disabled_kg_from_results_csv(path: str) -> None:
    import pandas as pd

    if not path or not os.path.exists(path):
        return
    df = pd.read_csv(path)
    kg_columns = [
        column
        for column in df.columns
        if column.startswith("kg_") or column.startswith("gold_kg_") or column.startswith("base_gold_kg_")
    ]
    if not kg_columns:
        return
    df = df.drop(columns=kg_columns)
    df.to_csv(path, index=False)


def write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_kg_profile_comparison_rows(experiment_summaries: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for summary in experiment_summaries:
        kg_summary = summary.get("kg", {}) if isinstance(summary.get("kg"), dict) else {}
        if not kg_summary.get("enabled"):
            continue
        base_row = {
            "experiment": summary.get("experiment"),
            "chunking_strategy": summary.get("chunking_strategy"),
            "retriever": summary.get("retriever"),
            "chunk_size": summary.get("chunk_size"),
            "chunk_overlap": summary.get("chunk_overlap"),
            "kg_profile": kg_summary.get("profile"),
            "kg_algorithm": kg_summary.get("algorithm"),
            "question_type": "all",
            "n_questions": summary.get("n_questions"),
        }
        metrics = kg_summary.get("metrics", {}) if isinstance(kg_summary.get("metrics"), dict) else {}
        answer_metrics = summary.get("answer_metrics", {}) if isinstance(summary.get("answer_metrics"), dict) else {}
        retrieval_metrics = summary.get("retrieval_metrics", {}) if isinstance(summary.get("retrieval_metrics"), dict) else {}
        rows.append(
            {
                **base_row,
                "mean_mrr_at_k": retrieval_metrics.get("mean_mrr_at_k"),
                "mean_ndcg_at_k": retrieval_metrics.get("mean_ndcg_at_k"),
                "mean_recall_at_k": retrieval_metrics.get("mean_recall_at_k"),
                "mean_grounded_claim_ratio": answer_metrics.get("mean_grounded_claim_ratio"),
                "mean_hallucinated_claim_ratio": answer_metrics.get("mean_hallucinated_claim_ratio"),
                "mean_kg_graph_gain_at_k": metrics.get("mean_kg_graph_gain_at_k"),
                "mean_kg_graph_noise_at_k": metrics.get("mean_kg_graph_noise_at_k"),
                "mean_kg_added_evidence_precision": metrics.get("mean_kg_added_evidence_precision"),
                "mean_kg_added_evidence_recall": metrics.get("mean_kg_added_evidence_recall"),
                "mean_kg_path_availability": metrics.get("mean_kg_path_availability"),
                "mean_kg_path_correctness": metrics.get("mean_kg_path_correctness"),
                "mean_kg_graph_faithfulness": metrics.get("mean_kg_graph_faithfulness"),
            }
        )
        kg_by_type = (
            kg_summary.get("metrics_by_question_type", {})
            if isinstance(kg_summary.get("metrics_by_question_type"), dict)
            else {}
        )
        retrieval_by_type = (
            summary.get("retrieval_metrics_by_question_type", {})
            if isinstance(summary.get("retrieval_metrics_by_question_type"), dict)
            else {}
        )
        answer_by_type = (
            summary.get("answer_metrics_by_question_type", {})
            if isinstance(summary.get("answer_metrics_by_question_type"), dict)
            else {}
        )
        for question_type, type_metrics in sorted(kg_by_type.items()):
            retrieval_type_metrics = retrieval_by_type.get(question_type, {})
            answer_type_metrics = answer_by_type.get(question_type, {})
            rows.append(
                {
                    **base_row,
                    "question_type": question_type,
                    "n_questions": type_metrics.get("n"),
                    "mean_mrr_at_k": retrieval_type_metrics.get("mean_mrr_at_k"),
                    "mean_ndcg_at_k": retrieval_type_metrics.get("mean_ndcg_at_k"),
                    "mean_recall_at_k": retrieval_type_metrics.get("mean_recall_at_k"),
                    "mean_grounded_claim_ratio": answer_type_metrics.get("mean_grounded_claim_ratio"),
                    "mean_hallucinated_claim_ratio": answer_type_metrics.get("mean_hallucinated_claim_ratio"),
                    "mean_kg_graph_gain_at_k": type_metrics.get("mean_kg_graph_gain_at_k"),
                    "mean_kg_graph_noise_at_k": type_metrics.get("mean_kg_graph_noise_at_k"),
                    "mean_kg_added_evidence_precision": type_metrics.get("mean_kg_added_evidence_precision"),
                    "mean_kg_added_evidence_recall": type_metrics.get("mean_kg_added_evidence_recall"),
                    "mean_kg_path_availability": type_metrics.get("mean_kg_path_availability"),
                    "mean_kg_path_correctness": type_metrics.get("mean_kg_path_correctness"),
                    "mean_kg_graph_faithfulness": type_metrics.get("mean_kg_graph_faithfulness"),
                }
            )
    return rows


def strip_disabled_kg_from_summary(summary: Dict) -> Dict:
    summary = dict(summary)
    summary.pop("kg", None)
    outputs = dict(summary.get("outputs", {}))
    outputs = {
        key: value
        for key, value in outputs.items()
        if not key.startswith("kg_")
    }
    summary["outputs"] = outputs
    return summary


def resolve_retriever_mode(args) -> str:
    if args.retriever is not None:
        return args.retriever
    return "dense" if args.embedding_backend == "sentence-transformers" else "tfidf"


def build_llm_config_from_args(args) -> LLMConfig:
    return LLMConfig(
        enabled=args.llm_enable,
        model=args.llm_model,
        api_key_env=args.openai_api_key_env,
        temperature=args.llm_temperature,
    )


def build_judge_config_from_args(args) -> LLMConfig:
    return LLMConfig(
        enabled=args.judge_enable,
        model=args.judge_model or args.llm_model,
        api_key_env=args.openai_api_key_env,
        temperature=args.judge_temperature,
    )


def llm_is_used_for_optional_steps(args) -> bool:
    return (
        args.llm_enable
        or args.query_augmentation in {"llm", "hyde"}
        or args.self_rag_retry_on_weak_evidence
        or args.self_rag_critique
    )


def load_document_corpus(args, run_dir: str) -> Dict[str, object]:
    import pandas as pd

    doc_paths = resolve_doc_paths(args.docs)
    questions = load_questions(args.questions)

    all_sections: List[Section] = []
    raw_parts: List[str] = []
    for path in doc_paths:
        raw_text, sections = parse_pdf_sections(path, docs_root=args.docs_root)
        title = sections[0].doc_id if sections else os.path.basename(path)
        raw_parts.append(f"===== {title} =====\n{raw_text}")
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

    return {
        "doc_paths": doc_paths,
        "questions": questions,
        "sections": all_sections,
        "paragraphs": paragraphs,
        "raw_documents_txt": raw_path,
        "sections_csv": sections_csv,
        "paragraphs_csv": paragraphs_csv,
    }


def copy_selected_outputs(summary: Dict, run_dir: str, keys: List[str]) -> Dict:
    outputs = dict(summary.get("outputs", {}))
    for key in keys:
        source = outputs.get(key)
        if not source or not os.path.exists(source):
            continue
        target = os.path.join(run_dir, os.path.basename(source))
        if os.path.abspath(source) != os.path.abspath(target):
            shutil.copyfile(source, target)
            outputs[key] = target
    summary = dict(summary)
    summary["outputs"] = outputs
    return summary


def build_experiments(args) -> List[Tuple[str, int, int]]:
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
        return experiments
    return [(args.chunking, args.chunk_size, args.overlap)]


def build_retriever_types(args, resolved_retriever: str) -> List[str]:
    if resolved_retriever == "auto":
        return [item.strip() for item in args.auto_retrievers.split(",") if item.strip()]
    return [resolved_retriever]


def build_sweep_configs(args, resolved_retriever: str) -> List[Dict]:
    valid_chunking = {"fixed_words", "fixed_tokens", "by_section", "by_paragraph"}
    valid_retrievers = {"tfidf", "bm25", "dense", "hybrid"}
    if args.sweep_configs_json:
        raw_configs = json.loads(args.sweep_configs_json)
        if not isinstance(raw_configs, list) or not raw_configs:
            raise ValueError("--sweep-configs-json must be a non-empty JSON list.")
        configs: List[Dict] = []
        for index, raw_config in enumerate(raw_configs, start=1):
            if not isinstance(raw_config, dict):
                raise ValueError(f"Sweep config #{index} must be a JSON object.")
            strategy = str(raw_config.get("chunking") or raw_config.get("strategy") or "").strip()
            retriever_type = str(raw_config.get("retriever") or "").strip()
            if strategy not in valid_chunking:
                raise ValueError(f"Unsupported chunking in sweep config #{index}: {strategy!r}")
            if retriever_type not in valid_retrievers:
                raise ValueError(f"Unsupported retriever in sweep config #{index}: {retriever_type!r}")
            chunk_size = int(raw_config.get("chunk_size") or 0)
            overlap = int(raw_config.get("overlap") or 0)
            if strategy in {"fixed_words", "fixed_tokens"}:
                chunk_size = chunk_size or args.chunk_size
            else:
                chunk_size = 0
                overlap = 0
            configs.append(
                {
                    "order": index,
                    "name": str(raw_config.get("name") or f"custom_{index}"),
                    "strategy": strategy,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "retriever": retriever_type,
                    "hybrid_alpha": float(raw_config.get("hybrid_alpha", args.hybrid_alpha)),
                    "kg_profile": str(raw_config.get("kg_profile") or args.kg_profile),
                    "kg_algorithm": raw_config.get("kg_algorithm", args.kg_algorithm),
                }
            )
        return configs

    configs = []
    kg_profiles = (
        [item.strip() for item in args.kg_profiles.split(",") if item.strip()]
        if args.kg_enable and args.kg_profiles.strip()
        else [args.kg_profile]
    )
    for index, ((strategy, chunk_size, overlap), retriever_type) in enumerate(
        (
            (experiment, retriever_type)
            for experiment in build_experiments(args)
            for retriever_type in build_retriever_types(args, resolved_retriever)
        ),
        start=1,
    ):
        for kg_profile in kg_profiles:
            configs.append(
                {
                    "order": len(configs) + 1,
                    "name": f"{strategy}_{chunk_size}_{overlap}_{retriever_type}_kg_{kg_profile}",
                    "strategy": strategy,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "retriever": retriever_type,
                    "hybrid_alpha": args.hybrid_alpha,
                    "kg_profile": kg_profile,
                    "kg_algorithm": args.kg_algorithm,
                }
            )
    return configs


def run_classifier_mode(args) -> Dict:
    ensure_dir(args.output_dir)
    run_dir = build_run_dir(args.output_dir, args.run_name)
    resolved_retriever = resolve_retriever_mode(args)

    if args.classifier_type == "ted_cpv":
        summary = evaluate_local_ted_cpv_classifier(
            cpv_catalog_path=args.cpv_catalog,
            queries_path=args.cpv_queries,
            retriever=resolved_retriever if resolved_retriever != "auto" else "tfidf",
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            use_examples=args.cpv_use_examples,
            classifier_label=args.classifier_type,
            run_dir=run_dir,
            create_visualization=args.create_strategy_visualization,
            create_showcase=args.create_strategy_showcase,
            rerank_top_n=args.rerank_top_n,
            rerank_weight=args.rerank_weight,
            cross_encoder_rerank=args.cross_encoder_rerank,
            cross_encoder_model=args.cross_encoder_model,
            cross_encoder_top_n=args.cross_encoder_top_n,
            kg_enabled=args.kg_enable,
            kg_graph_weight=args.kg_graph_weight,
            kg_profile=args.kg_profile,
            kg_algorithm=args.kg_algorithm,
            llm_config=build_llm_config_from_args(args),
            query_augmentation=args.query_augmentation,
            query_augmentation_max_terms=args.query_augmentation_max_terms,
        )
        if not args.kg_enable:
            summary = strip_disabled_kg_from_summary(summary)
        run_summary = {
            "run_dir": run_dir,
            "mode": "classifier",
            "classifier_type": args.classifier_type,
            "top_k": args.top_k,
            "visualization": {"enabled": args.create_strategy_visualization or args.create_strategy_showcase},
            "showcase": {"enabled": args.create_strategy_showcase},
            "classifier_summary": summary,
            "outputs": summary["outputs"],
            "kg": {
                "enabled": args.kg_enable,
            },
            "cross_encoder_reranker": {
                "enabled": args.cross_encoder_rerank,
                "model": args.cross_encoder_model,
                "top_n": args.cross_encoder_top_n,
            },
        }
        run_summary = write_design_attribution_artifacts(run_summary, run_dir)
        write_json(os.path.join(run_dir, "run_summary.json"), run_summary)
        return run_summary

    if args.classifier_type == "api_classifier":
        if args.api_eval_case != "ted_cpv":
            raise ValueError(f"Unsupported --api-eval-case: {args.api_eval_case}")
        if not args.api_classifier_url:
            raise ValueError("--api-classifier-url is required when --classifier-type=api_classifier.")
        extra_headers = {}
        if args.api_extra_headers_json:
            extra_headers = json.loads(args.api_extra_headers_json)
            if not isinstance(extra_headers, dict):
                raise ValueError("--api-extra-headers-json must decode to a JSON object.")
        summary = evaluate_api_ted_cpv_classifier(
            api_url=args.api_classifier_url,
            auth_token_env=args.api_auth_token_env,
            extra_headers=extra_headers,
            timeout_seconds=args.api_timeout_seconds,
            cpv_catalog_path=args.cpv_catalog,
            queries_path=args.cpv_queries,
            top_k=args.top_k,
            classifier_label=args.classifier_type,
            run_dir=run_dir,
            create_visualization=args.create_strategy_visualization,
            create_showcase=args.create_strategy_showcase,
        )
        summary = strip_disabled_kg_from_summary(summary)
        run_summary = {
            "run_dir": run_dir,
            "mode": "classifier",
            "classifier_type": args.classifier_type,
            "api_eval_case": args.api_eval_case,
            "top_k": args.top_k,
            "visualization": {"enabled": args.create_strategy_visualization or args.create_strategy_showcase},
            "showcase": {"enabled": args.create_strategy_showcase},
            "classifier_summary": summary,
            "outputs": summary["outputs"],
        }
        run_summary = write_design_attribution_artifacts(run_summary, run_dir)
        write_json(os.path.join(run_dir, "run_summary.json"), run_summary)
        return run_summary

    if args.classifier_type == "prepared_rag_results":
        summary = evaluate_prepared_rag_results_classifier(
            prepared_results_path=args.prepared_results,
            cpv_catalog_path=args.cpv_catalog,
            top_k=args.top_k,
            classifier_label=args.classifier_type,
            run_dir=run_dir,
            create_visualization=args.create_strategy_visualization,
            create_showcase=args.create_strategy_showcase,
        )
        summary = strip_disabled_kg_from_summary(summary)
        run_summary = {
            "run_dir": run_dir,
            "mode": "classifier",
            "classifier_type": args.classifier_type,
            "prepared_results": args.prepared_results,
            "top_k": args.top_k,
            "visualization": {"enabled": args.create_strategy_visualization or args.create_strategy_showcase},
            "showcase": {"enabled": args.create_strategy_showcase},
            "classifier_summary": summary,
            "outputs": summary["outputs"],
        }
        run_summary = write_design_attribution_artifacts(run_summary, run_dir)
        write_json(os.path.join(run_dir, "run_summary.json"), run_summary)
        return run_summary

    llm_config = build_llm_config_from_args(args)
    judge_config = build_judge_config_from_args(args)
    if args.chunking == "auto" or resolved_retriever == "auto":
        raise ValueError("Classifier mode expects one concrete config, not --chunking=auto or --retriever=auto.")

    corpus = load_document_corpus(args, run_dir)

    summary = run_single_experiment(
        sections=corpus["sections"],
        paragraphs=corpus["paragraphs"],
        questions=corpus["questions"],
        run_dir=run_dir,
        strategy=args.chunking,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        top_k=args.top_k,
        retriever_type=resolved_retriever,
        embedding_model=args.embedding_model,
        hybrid_alpha=args.hybrid_alpha,
        llm_config=llm_config,
        judge_config=judge_config,
        create_strategy_visualization=args.create_strategy_visualization,
        create_strategy_showcase=args.create_strategy_showcase,
        kg_enabled=args.kg_enable,
        kg_graph_weight=args.kg_graph_weight,
        kg_profile=args.kg_profile,
        kg_algorithm=args.kg_algorithm,
        kg_max_added_chunks=args.kg_max_added_chunks,
        kg_ppr_iterations=args.kg_ppr_iterations,
        kg_ppr_damping=args.kg_ppr_damping,
        kg_quality_threshold=args.kg_quality_threshold,
        kg_intent_weight=args.kg_intent_weight,
        kg_ablation_edge_dropouts=args.kg_ablation_edge_dropouts,
        answer_mode=args.answer_mode,
        context_mode=args.context_mode,
        max_context_chunks=args.max_context_chunks,
        max_context_chars=args.max_context_chars,
        query_augmentation=args.query_augmentation,
        query_augmentation_max_terms=args.query_augmentation_max_terms,
        decision_min_confidence=args.decision_min_confidence,
        decision_min_context_claim_recall=args.decision_min_context_claim_recall,
        decision_min_grounded_claim_ratio=args.decision_min_grounded_claim_ratio,
        runtime_retrieval_evaluator_enabled=not args.disable_runtime_retrieval_evaluator,
        abstain_on_weak_evidence=args.abstain_on_weak_evidence,
        self_rag_retry_on_weak_evidence=args.self_rag_retry_on_weak_evidence,
        self_rag_retry_max_attempts=args.self_rag_retry_max_attempts,
        self_rag_critique=args.self_rag_critique,
        rerank_top_n=args.rerank_top_n,
        rerank_weight=args.rerank_weight,
        )

    if not args.kg_enable:
        summary = strip_disabled_kg_from_summary(summary)

    summary = copy_selected_outputs(summary, run_dir, [
        "rag_results_csv",
        "retrieved_chunks_csv",
        "answer_metrics_csv",
        "answer_metrics_summary_json",
        "claim_evidence_map_csv",
        "claim_evidence_map_jsonl",
        "retrieval_metrics_csv",
        "retrieval_metrics_summary_json",
        "diagnostics_csv",
        "diagnostics_summary_json",
        "quality_advisor_json",
        "quality_report_md",
        "claim_judge_results_jsonl",
        "strategy_score_profile_svg",
        "strategy_chunk_alignment_svg",
        "strategy_unique_chunk_alignment_svg",
        "strategy_metric_overview_svg",
        "strategy_diagnostics_svg",
        "strategy_showcase_md",
    ])

    if not args.kg_enable:
        prune_disabled_kg_from_results_csv(summary["outputs"].get("rag_results_csv"))

    run_summary = {
        "run_dir": run_dir,
        "mode": "classifier",
        "classifier_type": args.classifier_type,
        "documents": corpus["doc_paths"],
        "docs_root": args.docs_root,
        "questions_path": args.questions,
        "n_documents": len(corpus["doc_paths"]),
        "n_sections": len(corpus["sections"]),
        "n_paragraphs": len(corpus["paragraphs"]),
        "n_questions": len(corpus["questions"]),
        "chunking_mode": args.chunking,
        "retriever_mode": resolved_retriever,
        "llm": {
            "enabled": llm_is_used_for_optional_steps(args),
            "model": llm_config.model if llm_is_used_for_optional_steps(args) else None,
            "answer_generation": llm_config.enabled,
            "query_augmentation": args.query_augmentation,
            "self_rag_retry_on_weak_evidence": args.self_rag_retry_on_weak_evidence,
            "self_rag_critique": args.self_rag_critique,
        },
        "reranker": {
            "enabled": args.rerank_top_n > 1 and args.rerank_weight > 0.0,
            "type": "lexical_overlap",
            "top_n": args.rerank_top_n,
            "weight": args.rerank_weight,
        },
        "judge": {
            "enabled": judge_config.enabled,
            "model": judge_config.model if judge_config.enabled else None,
        },
        "runtime_retrieval_evaluator": {
            "enabled": not args.disable_runtime_retrieval_evaluator,
            "abstain_on_weak_evidence": args.abstain_on_weak_evidence,
            "self_rag_retry_on_weak_evidence": args.self_rag_retry_on_weak_evidence,
            "self_rag_retry_max_attempts": args.self_rag_retry_max_attempts,
            "self_rag_critique": args.self_rag_critique,
        },
        "visualization": {"enabled": args.create_strategy_visualization or args.create_strategy_showcase},
        "showcase": {"enabled": args.create_strategy_showcase},
        "classifier_summary": summary,
        "outputs": {
            "raw_documents_txt": corpus["raw_documents_txt"],
            "sections_csv": corpus["sections_csv"],
            "paragraphs_csv": corpus["paragraphs_csv"],
            **summary["outputs"],
        },
    }
    if args.kg_enable:
        run_summary["kg"] = {
            "enabled": args.kg_enable,
        }
    run_summary = write_design_attribution_artifacts(run_summary, run_dir)
    write_json(os.path.join(run_dir, "run_summary.json"), run_summary)
    return run_summary


def main() -> None:
    args = build_arg_parser().parse_args()
    import pandas as pd

    if args.mode == "classifier":
        run_summary = run_classifier_mode(args)
        print(json.dumps(run_summary, ensure_ascii=False, indent=2))
        return

    ensure_dir(args.output_dir)
    run_dir = build_run_dir(args.output_dir, args.run_name)
    corpus = load_document_corpus(args, run_dir)
    resolved_retriever = resolve_retriever_mode(args)
    llm_config = build_llm_config_from_args(args)
    judge_config = build_judge_config_from_args(args)
    sweep_configs = build_sweep_configs(args, resolved_retriever)

    experiment_summaries: List[Dict] = []
    for config in sweep_configs:
        experiment_summaries.append(
            run_single_experiment(
                sections=corpus["sections"],
                paragraphs=corpus["paragraphs"],
                questions=corpus["questions"],
                run_dir=run_dir,
                strategy=config["strategy"],
                chunk_size=config["chunk_size"],
                chunk_overlap=config["overlap"],
                top_k=args.top_k,
                retriever_type=config["retriever"],
                embedding_model=args.embedding_model,
                hybrid_alpha=config["hybrid_alpha"],
                llm_config=llm_config,
                judge_config=judge_config,
                create_strategy_visualization=args.create_strategy_visualization,
                create_strategy_showcase=args.create_strategy_showcase,
                kg_enabled=args.kg_enable,
                kg_graph_weight=args.kg_graph_weight,
                kg_profile=config.get("kg_profile", args.kg_profile),
                kg_algorithm=config.get("kg_algorithm", args.kg_algorithm),
                kg_max_added_chunks=args.kg_max_added_chunks,
                kg_ppr_iterations=args.kg_ppr_iterations,
                kg_ppr_damping=args.kg_ppr_damping,
                kg_quality_threshold=args.kg_quality_threshold,
                kg_intent_weight=args.kg_intent_weight,
                kg_ablation_edge_dropouts=args.kg_ablation_edge_dropouts,
                answer_mode=args.answer_mode,
                context_mode=args.context_mode,
                max_context_chunks=args.max_context_chunks,
                max_context_chars=args.max_context_chars,
                query_augmentation=args.query_augmentation,
                query_augmentation_max_terms=args.query_augmentation_max_terms,
                decision_min_confidence=args.decision_min_confidence,
                decision_min_context_claim_recall=args.decision_min_context_claim_recall,
                decision_min_grounded_claim_ratio=args.decision_min_grounded_claim_ratio,
                runtime_retrieval_evaluator_enabled=not args.disable_runtime_retrieval_evaluator,
                abstain_on_weak_evidence=args.abstain_on_weak_evidence,
                self_rag_retry_on_weak_evidence=args.self_rag_retry_on_weak_evidence,
                self_rag_retry_max_attempts=args.self_rag_retry_max_attempts,
                self_rag_critique=args.self_rag_critique,
                rerank_top_n=args.rerank_top_n,
                rerank_weight=args.rerank_weight,
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
    kg_profile_comparison_csv = None
    kg_profile_comparison_rows = build_kg_profile_comparison_rows(experiment_summaries)
    if kg_profile_comparison_rows:
        kg_profile_comparison_csv = os.path.join(run_dir, "kg_profile_comparison.csv")
        pd.DataFrame(kg_profile_comparison_rows).to_csv(kg_profile_comparison_csv, index=False)
    strategy_showcase_index_md = None
    if args.create_strategy_showcase:
        strategy_showcase_index_md = write_run_showcase_index(
            experiment_summaries,
            os.path.join(run_dir, "strategy_showcase_index.md"),
        )

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
        "documents": corpus["doc_paths"],
        "docs_root": args.docs_root,
        "questions_path": args.questions,
        "n_documents": len(corpus["doc_paths"]),
        "n_sections": len(corpus["sections"]),
        "n_paragraphs": len(corpus["paragraphs"]),
        "n_questions": len(corpus["questions"]),
        "chunking_mode": "custom" if args.sweep_configs_json else args.chunking,
        "retriever_mode": "custom" if args.sweep_configs_json else resolved_retriever,
        "sweep_configs": sweep_configs,
        "llm": {
            "enabled": llm_is_used_for_optional_steps(args),
            "model": llm_config.model if llm_is_used_for_optional_steps(args) else None,
            "answer_generation": llm_config.enabled,
            "query_augmentation": args.query_augmentation,
            "self_rag_retry_on_weak_evidence": args.self_rag_retry_on_weak_evidence,
            "self_rag_critique": args.self_rag_critique,
        },
        "reranker": {
            "enabled": args.rerank_top_n > 1 and args.rerank_weight > 0.0,
            "type": "lexical_overlap",
            "top_n": args.rerank_top_n,
            "weight": args.rerank_weight,
        },
        "judge": {
            "enabled": judge_config.enabled,
            "model": judge_config.model if judge_config.enabled else None,
        },
        "runtime_retrieval_evaluator": {
            "enabled": not args.disable_runtime_retrieval_evaluator,
            "abstain_on_weak_evidence": args.abstain_on_weak_evidence,
            "self_rag_retry_on_weak_evidence": args.self_rag_retry_on_weak_evidence,
            "self_rag_retry_max_attempts": args.self_rag_retry_max_attempts,
            "self_rag_critique": args.self_rag_critique,
        },
        "visualization": {
            "enabled": args.create_strategy_visualization or args.create_strategy_showcase,
        },
        "showcase": {
            "enabled": args.create_strategy_showcase,
        },
        "kg": {
            "enabled": args.kg_enable,
            "profile": args.kg_profile if args.kg_enable else None,
            "profiles": [item.strip() for item in args.kg_profiles.split(",") if item.strip()] if args.kg_profiles else [],
            "ablation_edge_dropouts": args.kg_ablation_edge_dropouts,
        },
        "experiments": experiment_summaries,
        "ranking_weights": ranking_weights,
        "best_experiment": best_experiment,
        "recommendation": recommendation,
        "outputs": {
            "raw_documents_txt": corpus["raw_documents_txt"],
            "sections_csv": corpus["sections_csv"],
            "paragraphs_csv": corpus["paragraphs_csv"],
            "experiment_ranking_csv": ranking_csv,
            "kg_profile_comparison_csv": kg_profile_comparison_csv,
            "best_config_json": best_config_json,
            "best_config_md": best_config_md,
            "questions_json": args.questions,
            "rag_results_csv": run_rag_results_csv if best_experiment is not None else None,
            "latest_rag_results_csv": latest_rag_results_csv if best_experiment is not None else None,
            "strategy_showcase_index_md": strategy_showcase_index_md,
        },
        "question_schema": {
            "required": ["question"],
            "recommended": [
                "id",
                "program_id",
                "program_name",
                "doc_id",
                "gold_answer",
                "expected_keywords",
            ],
        }
    }

    run_summary = write_design_attribution_artifacts(run_summary, run_dir)
    write_json(os.path.join(run_dir, "run_summary.json"), run_summary)

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
