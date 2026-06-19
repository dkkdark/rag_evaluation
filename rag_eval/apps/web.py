from __future__ import annotations

import base64
import binascii
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_FILES_DIR = PROJECT_ROOT / "data" / "files"
UPLOAD_DIR = DATA_FILES_DIR / "uploads"
RUN_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
UPLOAD_NAME_RE = re.compile(r"[^A-Za-z0-9_. -]+")


@dataclass
class EvaluationJob:
    run_name: str
    command: list[str]
    run_dir: Path
    log_path: Path
    status: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    error: str | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "log_path": str(self.log_path),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "error": self.error,
            "command": self.command,
        }


JOBS: dict[str, EvaluationJob] = {}
JOBS_LOCK = threading.Lock()


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(
    handler: BaseHTTPRequestHandler,
    body: str,
    status: int = 200,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_json_any(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def safe_run_name(raw_name: str | None) -> str:
    if raw_name:
        cleaned = RUN_NAME_RE.sub("_", raw_name.strip()).strip("._-")
    else:
        cleaned = ""
    if not cleaned:
        cleaned = "web_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    return cleaned[:80]


def safe_upload_name(raw_name: str) -> str:
    name = Path(raw_name or "uploaded.pdf").name
    cleaned = UPLOAD_NAME_RE.sub("_", name).strip(" ._-")
    if not cleaned:
        cleaned = "uploaded.pdf"
    suffix = Path(cleaned).suffix.lower()
    if suffix not in {".pdf", ".pdfa"}:
        raise ValueError(f"Unsupported upload type for {raw_name!r}; only PDF files are accepted.")
    return cleaned[:120]


def default_run_name() -> str:
    return "web_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def default_question_pdf_paths(questions_path: Path | None = None) -> list[str]:
    path = questions_path or (PROJECT_ROOT / "data" / "questions_by_file.json")
    payload = read_json_any(path)
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        items = payload["questions"]
    elif isinstance(payload, list):
        items = payload
    else:
        return []

    program_names: list[str] = []
    explicit_doc_paths: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        raw_programs = item.get("program_names")
        if raw_programs is None:
            raw_programs = item.get("program_name")
        if isinstance(raw_programs, (list, tuple, set)):
            candidate_programs = [str(value).strip() for value in raw_programs]
        else:
            candidate_programs = [str(raw_programs or "").strip()]
        for program_name in candidate_programs:
            if program_name and program_name not in program_names:
                program_names.append(program_name)

        raw_doc_paths = item.get("doc_paths")
        if raw_doc_paths is None:
            raw_doc_paths = item.get("doc_path") or item.get("doc_id")
        if isinstance(raw_doc_paths, (list, tuple, set)):
            candidate_doc_paths = [str(value).strip() for value in raw_doc_paths]
        else:
            candidate_doc_paths = [str(raw_doc_paths or "").strip()]
        for doc_path in candidate_doc_paths:
            if not doc_path:
                continue
            normalized = doc_path.replace("\\", "/").lstrip("/")
            if normalized.startswith("data/files/"):
                normalized = normalized[len("data/files/") :]
            explicit_doc_paths.append(normalized)
            program_name = normalized.split("/", 1)[0].strip()
            if program_name and program_name not in program_names:
                program_names.append(program_name)

    defaults: list[str] = []
    for program_name in program_names:
        program_dir = DATA_FILES_DIR / program_name
        if not program_dir.is_dir():
            continue
        for pdf_path in sorted(program_dir.glob("*.pdf*")):
            if not pdf_path.is_file():
                continue
            normalized = str(pdf_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            if normalized not in defaults:
                defaults.append(normalized)

    for doc_path in explicit_doc_paths:
        normalized = doc_path if doc_path.startswith("data/files/") else f"data/files/{doc_path}"
        candidate = PROJECT_ROOT / normalized
        if candidate.is_file() and normalized not in defaults:
            defaults.append(normalized)
    return defaults


def evaluation_python() -> str:
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def bool_flag(payload: dict[str, Any], key: str) -> bool:
    return bool(payload.get(key))


def _payload_int(payload: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(payload.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def filter_active_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only settings that are explicitly selected so CLI defaults apply."""
    active = dict(payload)

    if not bool_flag(active, "kg_enable"):
        for key in (
            "kg_graph_weight",
            "kg_profile",
            "kg_algorithm",
            "kg_max_added_chunks",
            "kg_ppr_iterations",
            "kg_ppr_damping",
            "kg_quality_threshold",
            "kg_intent_weight",
            "kg_ablation_edge_dropouts",
        ):
            active.pop(key, None)

    if not bool_flag(active, "cross_encoder_rerank"):
        active.pop("cross_encoder_model", None)
        active.pop("cross_encoder_top_n", None)
    if not bool_flag(active, "llm_rerank"):
        active.pop("llm_rerank_top_n", None)
        active.pop("llm_rerank_weight", None)
    chunking = str(active.get("chunking") or "")
    if chunking != "auto":
        active.pop("auto_chunk_sizes", None)
        active.pop("auto_overlaps", None)
    if chunking not in {"fixed_words", "fixed_tokens"}:
        active.pop("chunk_size", None)
        active.pop("overlap", None)

    retriever = str(active.get("retriever") or "")
    if retriever not in {"hybrid", "auto"}:
        active.pop("hybrid_alpha", None)
    if retriever != "auto":
        active.pop("auto_retrievers", None)

    if _payload_int(active, "rerank_top_n") <= 0:
        active.pop("rerank_weight", None)
    if not bool_flag(active, "self_rag_retry_on_weak_evidence"):
        active.pop("self_rag_retry_max_attempts", None)

    query_augmentation = str(active.get("query_augmentation") or "").strip().lower()
    if query_augmentation not in {"llm", "hyde", "translate_en"}:
        active.pop("query_augmentation_max_terms", None)
    if query_augmentation in {"", "none"}:
        active.pop("query_augmentation", None)

    llm_needed = (
        bool_flag(active, "llm_enable")
        or bool_flag(active, "llm_rerank")
        or bool_flag(active, "judge_enable")
        or query_augmentation in {"llm", "hyde", "translate_en"}
        or bool_flag(active, "self_rag_retry_on_weak_evidence")
        or bool_flag(active, "self_rag_critique")
    )
    if not (
        bool_flag(active, "llm_enable")
        or bool_flag(active, "llm_rerank")
        or bool_flag(active, "judge_enable")
        or query_augmentation in {"llm", "hyde", "translate_en"}
        or bool_flag(active, "self_rag_retry_on_weak_evidence")
        or bool_flag(active, "self_rag_critique")
    ):
        active.pop("llm_model", None)
        active.pop("llm_temperature", None)
    if not bool_flag(active, "judge_enable"):
        active.pop("judge_model", None)
        active.pop("judge_temperature", None)
    if not llm_needed:
        active.pop("openai_api_key_env", None)

    return active


def add_value_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    command.extend([flag, str(value)])


def build_command(payload: dict[str, Any], run_name: str, output_dir: Path) -> list[str]:
    payload = filter_active_payload(payload)
    classifier_type = str(payload.get("classifier_type", "document_qa"))
    is_document_classifier = classifier_type in {"document_qa", "examination_regulations"}
    selected_docs = payload.get("selected_docs") or []
    if isinstance(selected_docs, str):
        selected_docs = [selected_docs]
    if is_document_classifier:
        if not selected_docs:
            raise ValueError("Select at least one PDF file for document_qa.")
        docs_value = ",".join(str(path) for path in selected_docs)
    else:
        docs_value = payload.get("docs", "data/files/**/*.pdf*")

    command = [
        evaluation_python(),
        "-W",
        "ignore::UserWarning:multiprocessing.resource_tracker",
        "-W",
        "ignore:resource_tracker:UserWarning",
        "-m",
        "rag_eval.entrypoints.evaluate_rag",
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
    ]

    simple_args = {
        "--mode": payload.get("mode", "sweep"),
        "--classifier-type": classifier_type,
        "--docs": docs_value,
        "--questions": payload.get("questions", "data/questions_by_file.json"),
        "--cpv-catalog": payload.get("cpv_catalog", "data/teddata_corpus_export.csv"),
        "--cpv-queries": payload.get("cpv_queries", "data/cpv_ted_test_queries.json"),
        "--prepared-results": payload.get("prepared_results", "data/eval_dataset.xlsx"),
        "--api-classifier-url": payload.get("api_classifier_url"),
        "--api-auth-token-env": payload.get("api_auth_token_env"),
        "--chunking": payload.get("chunking", "fixed_words"),
        "--chunk-size": payload.get("chunk_size", 450),
        "--overlap": payload.get("overlap", 60),
        "--top-k": payload.get("top_k", 8),
        "--retriever": payload.get("retriever", "tfidf"),
        "--embedding-model": payload.get("embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        "--search-index-dir": payload.get("search_index_dir"),
        "--auto-chunk-sizes": payload.get("auto_chunk_sizes"),
        "--auto-overlaps": payload.get("auto_overlaps"),
        "--auto-retrievers": payload.get("auto_retrievers"),
        "--sweep-configs-json": payload.get("sweep_configs_json"),
        "--hybrid-alpha": payload.get("hybrid_alpha"),
        "--answer-mode": payload.get("answer_mode") if is_document_classifier else None,
        "--context-mode": payload.get("context_mode") if is_document_classifier else None,
        "--max-context-chunks": payload.get("max_context_chunks"),
        "--max-context-chars": payload.get("max_context_chars"),
        "--query-augmentation": payload.get("query_augmentation"),
        "--query-augmentation-max-terms": payload.get("query_augmentation_max_terms"),
        "--decision-min-confidence": payload.get("decision_min_confidence"),
        "--decision-min-context-claim-recall": payload.get("decision_min_context_claim_recall"),
        "--decision-min-grounded-claim-ratio": payload.get("decision_min_grounded_claim_ratio"),
        "--self-rag-retry-max-attempts": payload.get("self_rag_retry_max_attempts"),
        "--kg-graph-weight": payload.get("kg_graph_weight"),
        "--kg-profile": payload.get("kg_profile"),
        "--kg-algorithm": payload.get("kg_algorithm"),
        "--kg-max-added-chunks": payload.get("kg_max_added_chunks"),
        "--kg-ppr-iterations": payload.get("kg_ppr_iterations"),
        "--kg-ppr-damping": payload.get("kg_ppr_damping"),
        "--kg-quality-threshold": payload.get("kg_quality_threshold"),
        "--kg-intent-weight": payload.get("kg_intent_weight"),
        "--kg-ablation-edge-dropouts": payload.get("kg_ablation_edge_dropouts"),
        "--rerank-top-n": payload.get("rerank_top_n"),
        "--rerank-weight": payload.get("rerank_weight"),
        "--cross-encoder-model": payload.get("cross_encoder_model"),
        "--cross-encoder-top-n": payload.get("cross_encoder_top_n"),
        "--llm-rerank-top-n": payload.get("llm_rerank_top_n"),
        "--llm-rerank-weight": payload.get("llm_rerank_weight"),
        "--weight-answer": payload.get("weight_answer"),
        "--weight-correctness": payload.get("weight_correctness"),
        "--weight-retrieval": payload.get("weight_retrieval"),
        "--llm-model": payload.get("llm_model"),
        "--openai-api-key-env": payload.get("openai_api_key_env"),
        "--llm-temperature": payload.get("llm_temperature"),
        "--judge-model": payload.get("judge_model"),
        "--judge-temperature": payload.get("judge_temperature"),
    }
    if not is_document_classifier:
        simple_args["--docs-root"] = payload.get("docs_root", "data/files")
    for flag, value in simple_args.items():
        add_value_arg(command, flag, value)

    command.append("--create-strategy-showcase")
    flags = {
        "--llm-enable": "llm_enable",
        "--judge-enable": "judge_enable",
        "--disable-runtime-retrieval-evaluator": "disable_runtime_retrieval_evaluator",
        "--abstain-on-weak-evidence": "abstain_on_weak_evidence",
        "--self-rag-retry-on-weak-evidence": "self_rag_retry_on_weak_evidence",
        "--self-rag-critique": "self_rag_critique",
        "--kg-enable": "kg_enable",
        "--export-kg-for-neo4j": "export_kg_for_neo4j",
        "--cross-encoder-rerank": "cross_encoder_rerank",
        "--llm-rerank": "llm_rerank",
        "--cpv-notice-examples-channel": "cpv_notice_examples_channel",
        "--disable-self-exclusion": "disable_self_exclusion",
    }
    for flag, key in flags.items():
        if bool_flag(payload, key):
            command.append(flag)
    return command


def run_job(job: EvaluationJob) -> None:
    job.status = "running"
    job.started_at = datetime.now().isoformat(timespec="seconds")
    job.run_dir.mkdir(parents=True, exist_ok=True)
    with JOBS_LOCK:
        JOBS[job.run_name] = job
    try:
        with job.log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(job.command) + "\n\n")
            log.flush()
            env = os.environ.copy()
            warning_filters = [
                "ignore::UserWarning:multiprocessing.resource_tracker",
                "ignore:resource_tracker:UserWarning",
            ]
            existing_warnings = str(env.get("PYTHONWARNINGS") or "").strip()
            merged_warnings = [existing_warnings] if existing_warnings else []
            merged_warnings.extend(warning_filters)
            env["PYTHONWARNINGS"] = ",".join(item for item in merged_warnings if item)
            job.process = subprocess.Popen(
                job.command,
                cwd=str(PROJECT_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
            )
            job.return_code = job.process.wait()
        if job.status == "cancelling":
            job.status = "cancelled"
        else:
            job.status = "completed" if job.return_code == 0 else "failed"
    except Exception as exc:  # pragma: no cover - defensive for long-running jobs
        job.status = "failed"
        job.error = str(exc)
    finally:
        job.finished_at = datetime.now().isoformat(timespec="seconds")
        with JOBS_LOCK:
            JOBS[job.run_name] = job


def stop_job(run_name: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(run_name)
    if not job:
        raise ValueError("No running job found for this run.")
    if job.status in {"completed", "failed", "cancelled"}:
        return job.to_dict()
    job.status = "cancelling"
    process = job.process
    if process and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    with JOBS_LOCK:
        JOBS[run_name] = job
    return job.to_dict()


def list_data_files() -> dict[str, list[str]]:
    default_selected_pdfs = default_question_pdf_paths()
    return {
        "pdfs": sorted(
            str(path.relative_to(PROJECT_ROOT))
            for path in (PROJECT_ROOT / "data" / "files").glob("**/*.pdf*")
            if path.is_file()
        ),
        "default_selected_pdfs": default_selected_pdfs,
        "questions": sorted(
            str(path.relative_to(PROJECT_ROOT))
            for path in (PROJECT_ROOT / "data").glob("**/*questions*.json")
            if path.is_file()
        ),
        "csv": sorted(
            str(path.relative_to(PROJECT_ROOT))
            for path in (PROJECT_ROOT / "data").glob("**/*.csv")
            if path.is_file()
        ),
        "json": sorted(
            str(path.relative_to(PROJECT_ROOT))
            for path in (PROJECT_ROOT / "data").glob("**/*.json")
            if path.is_file()
        ),
        "xlsx": sorted(
            str(path.relative_to(PROJECT_ROOT))
            for path in (PROJECT_ROOT / "data").glob("**/*.xlsx")
            if path.is_file()
        ),
    }


def save_uploaded_files(payload: dict[str, Any]) -> dict[str, Any]:
    files = payload.get("files") or []
    if not isinstance(files, list) or not files:
        raise ValueError("Upload payload must contain a non-empty files list.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Each uploaded file must be an object with name and content_base64.")
        filename = safe_upload_name(str(item.get("name") or "uploaded.pdf"))
        raw_content = str(item.get("content_base64") or "")
        try:
            content = base64.b64decode(raw_content, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Invalid base64 content for {filename}.") from exc
        if not content.startswith(b"%PDF"):
            raise ValueError(f"{filename} does not look like a PDF file.")
        target = UPLOAD_DIR / filename
        if target.exists():
            stem = target.stem
            suffix = target.suffix
            target = UPLOAD_DIR / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"
        target.write_bytes(content)
        saved.append(str(target.relative_to(PROJECT_ROOT)))
    return {"saved": saved, "options": list_data_files()}


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "run_summary.json") or {}
    best = summary.get("best_experiment") or {}
    classifier_summary = summary.get("classifier_summary") or {}
    answer_metrics = classifier_summary.get("answer_metrics") or {}
    retrieval_metrics = classifier_summary.get("retrieval_metrics") or {}
    experiments = summary.get("experiments") or []
    status = "completed" if summary else "unknown"
    with JOBS_LOCK:
        live_job = JOBS.get(run_dir.name)
    if live_job:
        status = live_job.status
        job = live_job.to_dict()
    else:
        job = {}

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "status": status,
        "mtime": run_dir.stat().st_mtime,
        "mode": summary.get("mode", "sweep"),
        "classifier_type": summary.get("classifier_type"),
        "n_documents": summary.get("n_documents"),
        "n_questions": summary.get("n_questions"),
        "chunking_mode": summary.get("chunking_mode"),
        "retriever_mode": summary.get("retriever_mode"),
        "best_experiment": best.get("experiment") if isinstance(best, dict) else None,
        "best_score": best.get("score") if isinstance(best, dict) else None,
        "n_experiments": len(experiments),
        "n_correct": classifier_summary.get("n_correct"),
        "n_incorrect": classifier_summary.get("n_incorrect"),
        "answer_accuracy": answer_metrics.get("accuracy"),
        "retrieval_recall": retrieval_metrics.get("mean_recall_at_k"),
        "job": job,
    }


def list_runs() -> list[dict[str, Any]]:
    if not DEFAULT_OUTPUT_DIR.exists():
        return []
    runs = [
        summarize_run(path)
        for path in DEFAULT_OUTPUT_DIR.iterdir()
        if path.is_dir()
    ]
    return sorted(runs, key=lambda row: row["mtime"], reverse=True)


def resolve_run_path(run_name: str) -> Path:
    cleaned = safe_run_name(unquote(run_name))
    path = (DEFAULT_OUTPUT_DIR / cleaned).resolve()
    if DEFAULT_OUTPUT_DIR.resolve() not in path.parents and path != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError("Invalid run path.")
    return path


def resolve_artifact_path(run_dir: Path, raw_path: str) -> Path:
    candidate = Path(unquote(raw_path))
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    allowed_roots = [run_dir.resolve(), DEFAULT_OUTPUT_DIR.resolve()]
    if not any(candidate == root or root in candidate.parents for root in allowed_roots):
        raise ValueError("Artifact path is outside outputs.")
    return candidate


def preview_csv(path: Path, limit: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader):
            if index >= limit:
                break
            rows.append(row)
        return {"columns": reader.fieldnames or [], "rows": rows}


def read_tail(path: Path, max_chars: int = 24000) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        data = handle.read()
    return data[-max_chars:]


def _metric_payload(title: str, value: Any, help_text: str, kind: str = "number") -> dict[str, Any]:
    return {"title": title, "value": value, "help": help_text, "kind": kind}


def _chat_answer_grounding_score(answer: str, retrieved: list[dict[str, Any]]) -> float | None:
    answer_tokens = {
        token
        for token in re.findall(r"\w+", answer.casefold(), flags=re.UNICODE)
        if len(token) > 2
    }
    if not answer_tokens:
        return None
    context_text = "\n".join(str(row.get("text") or "") for row in retrieved)
    context_tokens = {
        token
        for token in re.findall(r"\w+", context_text.casefold(), flags=re.UNICODE)
        if len(token) > 2
    }
    if not context_tokens:
        return 0.0
    return len(answer_tokens.intersection(context_tokens)) / len(answer_tokens)


def _chat_score_margin(retrieved: list[dict[str, Any]]) -> float | None:
    scores = []
    for row in retrieved:
        try:
            scores.append(float(row.get("score")))
        except (TypeError, ValueError):
            continue
    if len(scores) < 2:
        return None
    scores = sorted(scores, reverse=True)
    return scores[0] - scores[1]


def _min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-12:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _chat_rerank_by_question_evidence(
    *,
    question: str,
    rows: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    from rag_eval.evaluation.metrics import informative_tokens, numeric_tokens

    if top_k <= 0 or not rows:
        return []

    question_terms = set(informative_tokens(question))
    question_numbers = numeric_tokens(question)
    base_scores: list[float] = []
    for row in rows:
        try:
            base_scores.append(float(row.get("score") or 0.0))
        except (TypeError, ValueError):
            base_scores.append(0.0)
    normalized_scores = _min_max(base_scores)

    reranked: list[dict[str, Any]] = []
    for row, base_score in zip(rows, normalized_scores):
        search_text = "\n".join(
            str(row.get(key) or "")
            for key in ("section_id", "title", "text")
            if str(row.get(key) or "").strip()
        )
        text_terms = set(informative_tokens(search_text))
        lexical_coverage = (
            len(question_terms.intersection(text_terms)) / len(question_terms)
            if question_terms
            else 0.0
        )
        text_numbers = numeric_tokens(search_text)
        number_coverage = (
            len(question_numbers.intersection(text_numbers)) / len(question_numbers)
            if question_numbers
            else lexical_coverage
        )
        title_terms = set(informative_tokens(str(row.get("title") or "")))
        title_coverage = (
            len(question_terms.intersection(title_terms)) / len(question_terms)
            if question_terms and title_terms
            else 0.0
        )
        evidence_score = (
            0.50 * base_score
            + 0.30 * lexical_coverage
            + 0.15 * number_coverage
            + 0.05 * title_coverage
        )
        updated = dict(row)
        updated["base_score_before_evidence_rerank"] = row.get("score")
        updated["question_evidence_score"] = float(evidence_score)
        updated["question_lexical_coverage"] = float(lexical_coverage)
        updated["question_number_coverage"] = float(number_coverage)
        updated["score"] = float(evidence_score)
        updated["reranked"] = True
        updated["reranker"] = "question_evidence"
        reranked.append(updated)

    reranked.sort(key=lambda item: float(item.get("question_evidence_score") or 0.0), reverse=True)
    return reranked[:top_k]


def _chat_snippet_for_claim(claim: str, source_text: str, *, max_chars: int = 320) -> str:
    from rag_eval.evaluation.metrics import informative_tokens, numeric_tokens

    text = re.sub(r"\s+", " ", str(source_text or "")).strip()
    if not text:
        return ""

    anchors = list(numeric_tokens(claim))
    anchors.extend(sorted(set(informative_tokens(claim)), key=len, reverse=True))
    match_start = 0
    for anchor in anchors:
        if not anchor:
            continue
        match = re.search(re.escape(anchor), text, flags=re.IGNORECASE)
        if match:
            match_start = match.start()
            break

    half = max_chars // 2
    start = max(0, match_start - half)
    end = min(len(text), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def _chat_evidence_spans(
    *,
    answer: str,
    retrieved: list[dict[str, Any]],
    max_spans: int = 4,
) -> list[dict[str, Any]]:
    from rag_eval.evaluation.metrics import claim_support_score, split_claims

    claims = split_claims(answer)
    if not claims and answer.strip():
        claims = [answer.strip()]

    spans: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for claim in claims:
        best_row: dict[str, Any] | None = None
        best_score = 0.0
        best_rank = None
        for rank, row in enumerate(retrieved, start=1):
            source_text = str(row.get("text") or "")
            score = claim_support_score(claim, source_text)
            if score > best_score:
                best_score = score
                best_row = row
                best_rank = rank
        if not best_row or best_score < 0.35:
            continue

        quote = _chat_snippet_for_claim(claim, str(best_row.get("text") or ""))
        if not quote:
            continue
        key = (str(best_row.get("chunk_id") or best_rank or ""), quote)
        if key in seen:
            continue
        seen.add(key)
        spans.append(
            {
                "claim": claim,
                "quote": quote,
                "full_quote": re.sub(r"\s+", " ", str(best_row.get("text") or "")).strip(),
                "rank": best_rank,
                "chunk_id": best_row.get("chunk_id"),
                "doc_id": best_row.get("doc_id"),
                "title": best_row.get("title"),
                "source_score": best_row.get("score"),
                "support_score": best_score,
            }
        )
        if len(spans) >= max_spans:
            break
    return spans


def _chat_metric_cards(
    *,
    runtime_result: dict[str, Any],
    retrieved: list[dict[str, Any]],
    answer: str,
) -> list[dict[str, Any]]:
    top_score = runtime_result.get("top_score")
    unique_sources = {
        str(row.get("doc_id") or row.get("chunk_id") or "").strip()
        for row in retrieved
        if str(row.get("doc_id") or row.get("chunk_id") or "").strip()
    }
    return [
        _metric_payload(
            "Evidence confidence",
            runtime_result.get("score"),
            "A runtime estimate of whether retrieved context is strong enough to answer from.",
            "percent",
        ),
        _metric_payload(
            "Answer grounded in sources",
            _chat_answer_grounding_score(answer, retrieved),
            "Share of answer terms that also appear in retrieved source text. Higher usually means the answer is better grounded.",
            "percent",
        ),
        _metric_payload(
            "Query-context match",
            runtime_result.get("context_relevance"),
            "Lexical overlap between the user question and retrieved context.",
            "percent",
        ),
        _metric_payload(
            "Top source strength",
            top_score,
            "The strongest raw retrieval/candidate score among returned sources.",
        ),
        _metric_payload(
            "Top-2 score gap",
            _chat_score_margin(retrieved),
            "Difference between the best and second-best source score. Small gaps usually mean the decision is ambiguous.",
        ),
        _metric_payload(
            "Source diversity",
            len(unique_sources),
            "How many distinct documents or candidates contributed evidence.",
        ),
    ]


def _chat_api_turn(payload: dict[str, Any], *, question: str, top_k: int) -> dict[str, Any]:
    import time

    from rag_eval.classifiers.evaluation import (
        _normalize_prediction_label,
        _normalize_prediction_score,
        _post_classifier_request,
    )
    from rag_eval.evaluation.metrics import runtime_retrieval_evaluation

    api_url = str(payload.get("api_classifier_url") or "").strip()
    if not api_url:
        raise ValueError("API classifier URL is required for api_classifier chat mode.")
    started = time.perf_counter()
    response_payload = _post_classifier_request(
        api_url=api_url,
        payload={"id": "chat_turn", "query": question, "top_k": top_k},
        auth_token_env=str(payload.get("api_auth_token_env") or "API_CLASSIFIER_TOKEN"),
        extra_headers={},
        timeout_seconds=float(payload.get("api_timeout_seconds") or 30.0),
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    predictions_raw = response_payload.get("predictions", response_payload.get("top_k_answers", []))
    if not isinstance(predictions_raw, list):
        predictions_raw = []

    retrieved: list[dict[str, Any]] = []
    for rank, candidate in enumerate(predictions_raw[:top_k], start=1):
        if not isinstance(candidate, dict):
            continue
        label = _normalize_prediction_label(candidate)
        if not label:
            continue
        score = _normalize_prediction_score(candidate, fallback=max(0.0, 1.0 - 0.05 * (rank - 1)))
        retrieved.append(
            {
                "chunk_id": str(candidate.get("id") or candidate.get("code") or label),
                "doc_id": str(candidate.get("source") or ""),
                "section_id": "api_candidate",
                "title": label,
                "text": str(candidate.get("description") or candidate.get("text") or candidate.get("explanation") or label),
                "score": score,
                "retriever": "api_classifier",
            }
        )

    answer = str(response_payload.get("answer") or response_payload.get("final_answer") or "").strip()
    if not answer and retrieved:
        answer = retrieved[0]["title"]
    if not answer:
        answer = "No answer returned."
    runtime_result = runtime_retrieval_evaluation(question=question, retrieved=retrieved)
    metrics = _chat_metric_cards(
        runtime_result=runtime_result,
        retrieved=retrieved,
        answer=answer,
    )
    metrics.append(
        _metric_payload(
            "API latency",
            latency_ms,
            "Time spent waiting for the external classifier API.",
        )
    )
    return {
        "mode": "api_classifier",
        "question": question,
        "answer": answer,
        "metrics": metrics,
        "evidence_spans": _chat_evidence_spans(answer=answer, retrieved=retrieved),
        "retrieved": _chat_retrieved_payload(retrieved),
        "diagnostic": {
            "primary_error_reason": runtime_result.get("status"),
            "secondary_error_reason": runtime_result.get("action"),
            "explanation": runtime_result.get("reason"),
        },
    }


def _chat_retrieved_payload(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "chunk_id": row.get("chunk_id"),
            "doc_id": row.get("doc_id"),
            "title": row.get("title"),
            "score": row.get("score"),
            "base_score_before_evidence_rerank": row.get("base_score_before_evidence_rerank"),
            "question_evidence_score": row.get("question_evidence_score"),
            "reranker": row.get("reranker"),
            "text": row.get("text"),
        }
        for index, row in enumerate(retrieved, start=1)
    ]


def evaluate_chat_turn(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question") or "").strip()
    if not question:
        raise ValueError("Chat question is required.")

    mode = str(payload.get("chat_mode") or "document_qa")
    top_k = max(1, int(payload.get("top_k") or 5))
    if mode == "api_classifier":
        return _chat_api_turn(payload, question=question, top_k=top_k)

    if mode != "document_qa":
        raise ValueError(f"Unsupported chat mode: {mode}")

    if mode == "document_qa":
        from rag_eval.core.models import LLMConfig
        from rag_eval.data.io import extract_paragraphs, parse_pdf_sections, resolve_doc_paths
        from rag_eval.evaluation.llm import generate_answer_with_llm
        from rag_eval.evaluation.metrics import (
            keyword_extractive_answer,
            runtime_retrieval_evaluation,
        )
        from rag_eval.retrieval.chunking import build_chunks
        from rag_eval.retrieval.engines import (
            DEFAULT_EMBEDDING_MODEL,
            build_retriever_with_backend,
            retrieve_top_k,
        )

        selected_docs = payload.get("selected_docs") or []
        if isinstance(selected_docs, str):
            selected_docs = [selected_docs]
        if not selected_docs:
            raise ValueError("Select at least one PDF file for chat evaluation.")
        doc_patterns = ",".join(str(path) for path in selected_docs)
        doc_paths = resolve_doc_paths(doc_patterns)
        sections = []
        for doc_path in doc_paths:
            _, parsed_sections = parse_pdf_sections(doc_path, "data/files")
            sections.extend(parsed_sections)
        paragraphs = extract_paragraphs(sections)
        chunks = build_chunks(
            sections,
            paragraphs,
            str(payload.get("chunking") or "fixed_words"),
            int(payload.get("chunk_size") or 450),
            int(payload.get("overlap") or 60),
        )
        if not chunks:
            raise ValueError("No chunks were produced from the selected files.")
        retriever_type = str(payload.get("retriever") or "tfidf")
        embedding_model = str(payload.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
        search_backend_config = {
            "backend": "sqlite",
            "index_prefix": "rag-eval",
            "index_dir": str(payload.get("search_index_dir") or ".rag_eval_indices").strip(),
        }
        retriever_state = build_retriever_with_backend(
            chunks,
            retriever_type,
            embedding_model,
            search_backend_config=search_backend_config,
            index_name=None,
        )
        retrieval_depth = max(1, top_k)
        rerank_top_n = max(0, int(payload.get("rerank_top_n") or 0))
        generation_top_k = min(retrieval_depth, rerank_top_n) if rerank_top_n > 0 else retrieval_depth
        retrieved_raw = retrieve_top_k(
            question,
            retriever_state,
            chunks,
            k=retrieval_depth,
            hybrid_alpha=float(payload.get("hybrid_alpha") or 0.5),
        )
        retrieved = _chat_rerank_by_question_evidence(
            question=question,
            rows=retrieved_raw,
            top_k=generation_top_k,
        )
        runtime_result = runtime_retrieval_evaluation(question=question, retrieved=retrieved)
        answer_mode = "extractive_answer"
        answer = ""
        llm_result = None
        if bool(payload.get("llm_enable")):
            llm_result = generate_answer_with_llm(
                question,
                retrieved,
                LLMConfig(
                    enabled=True,
                    model=str(payload.get("llm_model") or "gpt-4.1-mini"),
                    api_key_env=str(payload.get("openai_api_key_env") or "OPENAI_API_KEY"),
                    temperature=float(payload.get("llm_temperature") or 0.0),
                ),
            )
            if llm_result.answer:
                answer = llm_result.answer
                answer_mode = "llm_grounded_answer"
            else:
                answer_mode = f"extractive_fallback_after_{llm_result.status}"
        if not answer:
            if bool(payload.get("abstain_on_weak_evidence")) and runtime_result.get("action") == "abstain":
                answer = "I do not have enough reliable context to answer this from the selected sources."
                answer_mode = "abstained_on_weak_evidence"
            else:
                answer = keyword_extractive_answer(question, retrieved)

    return {
        "mode": mode,
        "question": question,
        "answer": answer,
        "metrics": _chat_metric_cards(
            runtime_result=runtime_result,
            retrieved=retrieved,
            answer=answer,
        ),
        "evidence_spans": _chat_evidence_spans(answer=answer, retrieved=retrieved),
        "retrieved": _chat_retrieved_payload(retrieved),
        "diagnostic": {
            "primary_error_reason": runtime_result.get("status"),
            "secondary_error_reason": runtime_result.get("action"),
            "explanation": runtime_result.get("reason"),
        },
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Evaluation Admin</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172026;
      --muted: #5c6870;
      --line: #d7dde2;
      --panel: #f6f8fa;
      --accent: #006a67;
      --accent-2: #8a4b08;
      --bad: #b42318;
      --good: #147a42;
      --bg: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 22px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 18px;
      min-width: 0;
    }
    .app-nav {
      display: inline-flex;
      gap: 4px;
      padding: 3px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .app-tab {
      min-height: 30px;
      border: 0;
      background: transparent;
      color: var(--muted);
      padding: 5px 10px;
    }
    .app-tab.active {
      background: var(--accent);
      color: #fff;
      font-weight: 650;
    }
    h1, h2, h3 { margin: 0; letter-spacing: 0; }
    h1 { font-size: 20px; }
    h2 { font-size: 16px; margin-bottom: 12px; }
    h3 { font-size: 14px; margin-bottom: 8px; }
    main {
      display: grid;
      grid-template-columns: minmax(330px, 420px) minmax(0, 1fr);
      min-height: calc(100vh - 58px);
    }
    aside {
      border-right: 1px solid var(--line);
      padding: 18px;
      background: var(--panel);
    }
    section.workspace { padding: 18px 22px; min-width: 0; }
    .band {
      border-bottom: 1px solid var(--line);
      padding: 0 0 18px;
      margin-bottom: 18px;
    }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    .label-row { display: flex; align-items: center; gap: 6px; }
    .hint {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 17px;
      height: 17px;
      border-radius: 50%;
      border: 1px solid var(--line);
      color: var(--accent);
      background: #fff;
      font-size: 11px;
      font-weight: 700;
      cursor: help;
      flex: 0 0 auto;
    }
    .hint::after {
      content: attr(title);
      position: absolute;
      left: 50%;
      bottom: calc(100% + 8px);
      transform: translateX(-50%);
      display: none;
      width: max-content;
      max-width: min(320px, 72vw);
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #172026;
      color: #fff;
      box-shadow: 0 8px 22px rgba(23, 32, 38, 0.18);
      font-size: 12px;
      font-weight: 500;
      line-height: 1.35;
      text-align: left;
      white-space: normal;
      z-index: 20;
      pointer-events: none;
    }
    .hint::before {
      content: "";
      position: absolute;
      left: 50%;
      bottom: calc(100% + 3px);
      transform: translateX(-50%) rotate(45deg);
      display: none;
      width: 9px;
      height: 9px;
      background: #172026;
      z-index: 19;
      pointer-events: none;
    }
    .hint:hover::after,
    .hint:focus::after,
    .hint:hover::before,
    .hint:focus::before {
      display: block;
    }
    input, select, textarea, button {
      font: inherit;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      min-height: 36px;
    }
    input, select, textarea { padding: 7px 9px; width: 100%; }
    textarea { min-height: 70px; resize: vertical; }
    button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 7px 12px;
      cursor: pointer;
      background: #fff;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
      font-weight: 650;
    }
    button:disabled { opacity: 0.58; cursor: not-allowed; }
    .checks { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-top: 10px; }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 32px;
      font-size: 13px;
      color: var(--ink);
    }
    .check input { width: auto; min-height: auto; }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .segmented {
      display: inline-flex;
      gap: 4px;
      padding: 3px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .segmented label {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 5px 9px;
      border-radius: 6px;
      color: var(--muted);
      cursor: pointer;
    }
    .segmented input { display: none; }
    .segmented label:has(input:checked) {
      background: var(--accent);
      color: #fff;
      font-weight: 650;
    }
    .sweep-builder {
      display: grid;
      gap: 12px;
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .sweep-options {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .option-panel {
      display: grid;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: var(--panel);
    }
    .chip-grid { display: flex; flex-wrap: wrap; gap: 7px; }
    .chip-check {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      padding: 5px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-size: 12px;
    }
    .chip-check input { width: auto; min-height: auto; }
    .combo-list { display: grid; gap: 7px; }
    .combo-row {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: var(--panel);
    }
    .combo-actions { display: flex; gap: 5px; }
    .combo-actions button { min-height: 28px; padding: 4px 8px; }
    .runs {
      display: grid;
      gap: 8px;
      max-height: 320px;
      overflow: auto;
      padding-right: 3px;
    }
    .run-row {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 10px;
      cursor: pointer;
    }
    .run-row.active { border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }
    .run-title { display: flex; justify-content: space-between; gap: 8px; font-weight: 650; }
    .meta { color: var(--muted); font-size: 12px; margin-top: 3px; overflow-wrap: anywhere; }
    .status {
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: #fff;
      white-space: nowrap;
    }
    .status.completed { color: var(--good); border-color: #a7d8bd; }
    .status.failed { color: var(--bad); border-color: #f0b4ae; }
    .status.running, .status.queued, .status.cancelling { color: var(--accent-2); border-color: #e6c489; }
    .status.cancelled { color: var(--muted); border-color: var(--line); }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
    }
    .metric {
      border-top: 3px solid var(--accent);
      background: var(--panel);
      padding: 10px;
      min-height: 74px;
    }
    .metric .value { font-size: 22px; font-weight: 720; margin-top: 4px; }
    .metric .metric-key {
      font-size: 12px;
      color: var(--ink);
      overflow-wrap: anywhere;
    }
    .metric .metric-source {
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
    }
    .tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
    .tab.active { border-color: var(--accent); color: var(--accent); font-weight: 650; }
    .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 6px; max-height: 460px; }
    table { border-collapse: collapse; width: 100%; min-width: 780px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }
    th { background: var(--panel); position: sticky; top: 0; z-index: 1; }
    td { max-width: 360px; overflow-wrap: anywhere; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #101820;
      color: #edf4f4;
      border-radius: 6px;
      padding: 12px;
      max-height: 360px;
      overflow: auto;
    }
    .file-picker {
      display: grid;
      gap: 6px;
      max-height: 190px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 8px;
    }
    .file-picker label {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      color: var(--ink);
      font-size: 12px;
      line-height: 1.3;
    }
    .file-picker input { width: auto; min-height: auto; margin-top: 2px; }
    .artifact-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }
    .artifact-card {
      display: grid;
      gap: 8px;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 10px;
      min-height: 126px;
    }
    .artifact-card a { color: var(--accent); overflow-wrap: anywhere; font-weight: 650; }
    .artifact-preview {
      display: grid;
      place-items: center;
      min-height: 68px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 4px;
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
    }
    .artifact-preview img { width: 100%; max-height: 160px; object-fit: contain; }
    .recommendations { display: grid; gap: 10px; }
    .recommendation {
      border-left: 3px solid var(--accent);
      background: var(--panel);
      padding: 10px 12px;
    }
    .recommendation strong { color: var(--ink); }
    .recommendation .source-badge {
      display: inline-block;
      margin-bottom: 6px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #e7ece7;
      color: #415241;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .chat-shell {
      min-height: calc(100vh - 70px);
      background: #f2f6f6;
      padding: 14px;
      overflow: visible;
    }
    .chat-page {
      display: grid;
      grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
      gap: 18px;
      max-width: 1500px;
      margin: 0 auto;
      align-items: start;
    }
    .chat-config,
    .chat-main {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .chat-config {
      position: sticky;
      top: 84px;
      padding: 14px;
      align-self: start;
      display: grid;
      gap: 12px;
      max-height: calc(100vh - 104px);
      overflow: auto;
    }
    .chat-main {
      display: grid;
      grid-template-rows: auto auto auto;
      min-height: 620px;
      overflow: visible;
    }
    .chat-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    .chat-thread {
      grid-row: 3;
      overflow: visible;
      padding: 18px;
      display: grid;
      align-content: start;
      gap: 14px;
    }
    .chat-turn {
      display: grid;
      gap: 10px;
    }
    .bubble {
      max-width: min(820px, 100%);
      border-radius: 8px;
      padding: 11px 13px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .bubble.user {
      justify-self: end;
      background: var(--accent);
      color: #fff;
    }
    .bubble.assistant {
      justify-self: start;
      background: #eef4f1;
      border: 1px solid #cddbd6;
    }
    .chat-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
      gap: 8px;
      max-width: 980px;
    }
    .chat-metrics .metric {
      background: #fff;
      border: 1px solid var(--line);
      border-top: 3px solid var(--accent-2);
      min-height: 70px;
    }
    .chat-sources {
      max-width: 980px;
      display: grid;
      gap: 7px;
    }
    .chat-doc-config,
    .chat-api-config {
      display: grid;
      gap: 10px;
    }
    .source-toggle {
      width: fit-content;
      min-height: 30px;
      font-size: 12px;
      color: var(--accent);
    }
    .retrieved-list { display: grid; gap: 8px; }
    .retrieved-item { border: 1px solid var(--line); border-radius: 6px; padding: 9px; background: #fff; }
    .chat-composer {
      grid-row: 2;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 14px;
      background: #fbfcfd;
    }
    .composer-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: end;
    }
    .composer-row textarea {
      min-height: 52px;
      max-height: 170px;
      border-radius: 8px;
    }
    .hidden { display: none !important; }
    .inactive-field { display: none !important; }
    .empty { color: var(--muted); padding: 16px 0; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .sweep-options { grid-template-columns: 1fr; }
      .brand { align-items: flex-start; flex-direction: column; gap: 8px; }
      .chat-shell { min-height: calc(100vh - 92px); overflow: visible; }
      .chat-page { grid-template-columns: 1fr; min-height: 680px; }
      .chat-main { min-height: 620px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>RAG Evaluation</h1>
      <nav class="app-nav" aria-label="Application mode">
        <button class="app-tab active" id="adminTabBtn" type="button">Admin</button>
        <button class="app-tab" id="chatTabBtn" type="button">Chat</button>
      </nav>
    </div>
    <div class="toolbar">
      <button id="refreshBtn" title="Refresh runs">Refresh</button>
      <button id="loadDefaultsBtn" title="Reload detected files">Detect files</button>
    </div>
  </header>
  <main id="adminApp">
    <aside>
      <div class="band">
        <h2>Run settings</h2>
        <form id="runForm">
          <div class="grid">
            <label><span class="label-row">Run name <span class="hint" title="Default format is web_eval_YYYYMMDD_HHMMSS. Leave as is or rename before launch.">i</span></span><input name="run_name" /></label>
            <label><span class="label-row">Mode <span class="hint" title="classifier runs one selected classifier; sweep compares several RAG retrieval/chunking settings.">i</span></span><select name="mode"><option value="classifier">classifier</option><option value="sweep">sweep</option></select></label>
            <label><span class="label-row">Classifier <span class="hint" title="Choose the evaluation target. Only settings relevant to this classifier are shown below.">i</span></span><select name="classifier_type"><option value="document_qa">document_qa_files</option><option value="ted_cpv">ted_cpv</option><option value="api_classifier">api_classifier</option><option value="prepared_rag_results">prepared_rag_results</option></select></label>
            <label><span class="label-row">Top K <span class="hint" title="How many candidates/chunks to retrieve for each question. Higher values improve recall but add noise.">i</span></span><input name="top_k" type="number" min="1" value="8" /></label>
          </div>

          <div class="setting-group" data-show-for="document_qa examination_regulations sweep" style="margin-top:10px">
            <label><span class="label-row">PDF files <span class="hint" title="Select the exact regulation PDFs for this run. The server passes these files directly to --docs.">i</span></span><div class="file-picker" id="pdfPicker"></div></label>
            <div class="toolbar" style="margin-top:8px">
              <input type="file" id="fileUploadInput" multiple accept=".pdf,.pdfa,application/pdf" />
              <button type="button" id="uploadDocsBtn">Upload PDFs</button>
              <button type="button" id="selectAllDocsBtn">Select all files</button>
              <button type="button" id="clearDocsBtn">Clear</button>
            </div>
          </div>

          <div class="grid setting-group" data-show-for="document_qa examination_regulations" style="margin-top:10px">
            <label><span class="label-row">Questions <span class="hint" title="JSON file with evaluation questions and gold answers/keywords.">i</span></span><input name="questions" value="data/questions_by_file.json" /></label>
            <label><span class="label-row">Retriever <span class="hint" title="Retrieval backend. auto compares several retrievers during sweep.">i</span></span><select name="retriever"><option>tfidf</option><option>bm25</option><option>dense</option><option>hybrid</option><option>auto</option></select></label>
            <label><span class="label-row">Chunking <span class="hint" title="How documents are split before retrieval. auto compares several strategies during sweep.">i</span></span><select name="chunking"><option>fixed_words</option><option>fixed_tokens</option><option>by_section</option><option>by_paragraph</option><option>auto</option></select></label>
            <label data-show-if="chunking:fixed_words|fixed_tokens"><span class="label-row">Chunk size <span class="hint" title="Size of fixed chunks. Ignored for section/paragraph chunking.">i</span></span><input name="chunk_size" type="number" min="0" value="450" /></label>
            <label data-show-if="chunking:fixed_words|fixed_tokens"><span class="label-row">Overlap <span class="hint" title="Token/word overlap between adjacent fixed chunks.">i</span></span><input name="overlap" type="number" min="0" value="60" /></label>
            <label data-show-if="retriever:hybrid|auto" class="inactive-field"><span class="label-row">Hybrid alpha <span class="hint" title="Dense score weight for hybrid retrieval; BM25 gets the remaining weight.">i</span></span><input name="hybrid_alpha" type="number" step="0.05" value="0.5" /></label>
            <label data-show-if="chunking:auto" class="inactive-field"><span class="label-row">Auto sizes <span class="hint" title="Comma-separated chunk sizes used when chunking=auto.">i</span></span><input name="auto_chunk_sizes" value="256,450" /></label>
            <label data-show-if="chunking:auto" class="inactive-field"><span class="label-row">Auto overlaps <span class="hint" title="Comma-separated overlaps used when chunking=auto.">i</span></span><input name="auto_overlaps" value="0,60" /></label>
            <label data-show-if="retriever:auto" class="inactive-field"><span class="label-row">Auto retrievers <span class="hint" title="Comma-separated retrievers used when retriever=auto.">i</span></span><input name="auto_retrievers" value="tfidf,bm25,dense,hybrid" /></label>
          </div>

          <div class="setting-group" data-show-for="sweep">
            <div class="grid" style="margin-top:10px">
              <label><span class="label-row">Questions <span class="hint" title="JSON file with evaluation questions and gold answers/keywords.">i</span></span><input name="sweep_questions" value="data/questions_by_file.json" /></label>
              <label><span class="label-row">Hybrid alpha <span class="hint" title="Dense score weight for hybrid retrieval; BM25 gets the remaining weight.">i</span></span><input name="sweep_hybrid_alpha" type="number" step="0.05" value="0.5" /></label>
            </div>
            <div class="sweep-builder">
              <div class="toolbar" style="justify-content:space-between">
                <div>
                  <h3>Sweep Configurations</h3>
                  <div class="meta">Run all selected combinations, or build an ordered list manually.</div>
                </div>
                <div class="segmented">
                  <label><input type="radio" name="sweep_builder_mode" value="all" checked /> All selected</label>
                  <label><input type="radio" name="sweep_builder_mode" value="ordered" /> Ordered list</label>
                </div>
              </div>
              <div class="sweep-options">
                <div class="option-panel">
                  <div class="toolbar" style="justify-content:space-between"><strong>Chunking</strong><button type="button" data-select-sweep="chunking">All</button></div>
                  <div class="chip-grid">
                    <label class="chip-check"><input type="checkbox" data-sweep-chunking value="fixed_words" checked /> fixed_words</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-chunking value="fixed_tokens" checked /> fixed_tokens</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-chunking value="by_section" checked /> by_section</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-chunking value="by_paragraph" checked /> by_paragraph</label>
                  </div>
                </div>
                <div class="option-panel">
                  <div class="toolbar" style="justify-content:space-between"><strong>Retriever</strong><button type="button" data-select-sweep="retriever">All</button></div>
                  <div class="chip-grid">
                    <label class="chip-check"><input type="checkbox" data-sweep-retriever value="tfidf" checked /> tfidf</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-retriever value="bm25" checked /> bm25</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-retriever value="dense" checked /> dense</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-retriever value="hybrid" checked /> hybrid</label>
                  </div>
                </div>
                <div class="option-panel">
                  <div class="toolbar" style="justify-content:space-between"><strong>Chunk sizes</strong><span class="meta">fixed only</span></div>
                  <div class="chip-grid">
                    <label class="chip-check"><input type="checkbox" data-sweep-size value="256" checked /> 256</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-size value="450" checked /> 450</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-size value="700" /> 700</label>
                  </div>
                </div>
                <div class="option-panel">
                  <div class="toolbar" style="justify-content:space-between"><strong>Overlaps</strong><span class="meta">fixed only</span></div>
                  <div class="chip-grid">
                    <label class="chip-check"><input type="checkbox" data-sweep-overlap value="0" checked /> 0</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-overlap value="60" checked /> 60</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-overlap value="120" /> 120</label>
                  </div>
                </div>
                <div class="option-panel">
                  <div class="toolbar" style="justify-content:space-between"><strong>Answer mode</strong><button type="button" data-select-sweep="answer-mode">All</button></div>
                  <div class="chip-grid">
                    <label class="chip-check"><input type="checkbox" data-sweep-answer-mode value="" checked /> profile default</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-answer-mode value="extractive" /> extractive</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-answer-mode value="grounded_llm" /> grounded_llm</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-answer-mode value="cite_first" /> cite_first</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-answer-mode value="claim_checklist" /> claim_checklist</label>
                  </div>
                </div>
                <div class="option-panel">
                  <div class="toolbar" style="justify-content:space-between"><strong>Context mode</strong><button type="button" data-select-sweep="context-mode">All</button></div>
                  <div class="chip-grid">
                    <label class="chip-check"><input type="checkbox" data-sweep-context-mode value="" checked /> profile default</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-context-mode value="ranked" /> ranked</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-context-mode value="kg_first" /> kg_first</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-context-mode value="kg_organized" /> kg_organized</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-context-mode value="group_by_doc" /> group_by_doc</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-context-mode value="dedupe_section" /> dedupe_section</label>
                  </div>
                </div>
                <div class="option-panel">
                  <div class="toolbar" style="justify-content:space-between"><strong>Feature toggles</strong><span class="meta">document QA only</span></div>
                  <div class="chip-grid">
                    <label class="chip-check"><input type="checkbox" data-sweep-kg-enabled value="true" /> KG on</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-kg-enabled value="false" checked /> KG off</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-judge-enable value="true" /> Judge on</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-judge-enable value="false" checked /> Judge off</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-abstain value="true" /> Abstain on</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-abstain value="false" checked /> Abstain off</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-retry value="true" /> Retry on</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-retry value="false" checked /> Retry off</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-critique value="true" /> Critique on</label>
                    <label class="chip-check"><input type="checkbox" data-sweep-critique value="false" checked /> Critique off</label>
                  </div>
                </div>
              </div>
              <div class="ordered-sweep-controls hidden">
                <div class="grid">
                  <label>Chunking<select id="customSweepChunking"><option>fixed_words</option><option>fixed_tokens</option><option>by_section</option><option>by_paragraph</option></select></label>
                  <label>Retriever<select id="customSweepRetriever"><option>tfidf</option><option>bm25</option><option>dense</option><option>hybrid</option></select></label>
                  <label>Chunk size<input id="customSweepSize" type="number" value="450" min="0" /></label>
                  <label>Overlap<input id="customSweepOverlap" type="number" value="60" min="0" /></label>
                  <label>Answer mode<select id="customSweepAnswerMode"><option value="" selected>profile default</option><option>extractive</option><option>grounded_llm</option><option>cite_first</option><option>claim_checklist</option></select></label>
                  <label>Context mode<select id="customSweepContextMode"><option value="" selected>profile default</option><option>ranked</option><option>kg_first</option><option>kg_organized</option><option>group_by_doc</option><option>dedupe_section</option></select></label>
                  <label>KG<select id="customSweepKgEnabled"><option value="true">on</option><option value="false">off</option></select></label>
                  <label>Judge<select id="customSweepJudgeEnable"><option value="false" selected>off</option><option value="true">on</option></select></label>
                  <label>Abstain<select id="customSweepAbstain"><option value="false" selected>off</option><option value="true">on</option></select></label>
                  <label>Retry<select id="customSweepRetry"><option value="false" selected>off</option><option value="true">on</option></select></label>
                  <label>Critique<select id="customSweepCritique"><option value="false" selected>off</option><option value="true">on</option></select></label>
                </div>
                <div class="toolbar" style="margin-top:8px">
                  <button type="button" id="addSweepComboBtn">Add combination</button>
                  <button type="button" id="addSelectedSweepCombosBtn">Add selected grid</button>
                  <button type="button" id="clearSweepCombosBtn">Clear list</button>
                </div>
                <div class="combo-list" id="customSweepList"></div>
              </div>
              <div class="meta" id="sweepPreview"></div>
            </div>
          </div>

          <div class="grid setting-group" data-show-for="ted_cpv api_classifier" style="margin-top:10px">
            <label><span class="label-row">TED corpus export <span class="hint" title="teddata corpus export CSV used to rebuild CPV profiles on the fly. Notice examples are always taken from the local SQLite database.">i</span></span><input name="cpv_catalog" value="data/teddata_corpus_export.csv" /></label>
            <label><span class="label-row">CPV queries <span class="hint" title="TED/CPV test queries JSON used for classifier evaluation.">i</span></span><input name="cpv_queries" value="data/cpv_ted_test_queries.json" /></label>
            <label><span class="label-row">Retriever <span class="hint" title="Retriever used by the local CPV classifier. API classifier ignores this.">i</span></span><select name="cpv_retriever"><option>tfidf</option><option>bm25</option><option>dense</option><option>hybrid</option></select></label>
          </div>

          <div class="grid setting-group" data-show-for="ted_cpv" style="margin-top:10px">
            <label class="check" style="grid-column:1/-1"><input type="checkbox" name="cpv_notice_examples_channel" /> Notice examples retrieval channel <span class="hint" title="Retrieve over cpv_notice_examples as a separate TED/CPV channel instead of relying only on cpv_profiles.">i</span></label>
            <label class="check" style="grid-column:1/-1"><input type="checkbox" name="disable_self_exclusion" /> Disable self-exclusion for notice examples <span class="hint" title="Leave this unchecked for honest evaluation. When unchecked, the current query notice is excluded from notice-example retrieval by publication_number.">i</span></label>
          </div>

          <div class="grid setting-group" data-show-for="api_classifier" style="margin-top:10px">
            <label><span class="label-row">API classifier URL <span class="hint" title="HTTP endpoint for external classifier predictions.">i</span></span><input name="api_classifier_url" placeholder="https://..." /></label>
            <label><span class="label-row">API token env <span class="hint" title="Environment variable containing a Bearer token for the classifier API.">i</span></span><input name="api_auth_token_env" value="API_CLASSIFIER_TOKEN" /></label>
          </div>
          <div class="setting-group" data-show-for="api_classifier prepared_rag_results" style="margin-top:10px">
            <div class="meta">Field aliases accepted: id/query/question, expected/expected_answer/gold/reference, prediction/predicted_answer/answer/candidate/label/code, score/confidence/probability/retrieval_score/reranker_score. Ranked files may use suffixes like #1, _1, or rank1.</div>
          </div>

          <div class="grid setting-group" data-show-for="prepared_rag_results" style="margin-top:10px">
            <label><span class="label-row">Prepared results <span class="hint" title="Excel file with existing RAG/classifier outputs. Multiple rows with one ID are treated as top-k candidates.">i</span></span><input name="prepared_results" value="data/eval_dataset.xlsx" /></label>
            <label><span class="label-row">TED corpus export <span class="hint" title="teddata corpus export CSV used to rebuild CPV candidates and hierarchy labels.">i</span></span><input name="prepared_cpv_catalog" value="data/teddata_corpus_export.csv" /></label>
          </div>

          <div class="grid setting-group" data-show-for="document_qa examination_regulations ted_cpv sweep" style="margin-top:10px">
            <label><span class="label-row">Rerank top N <span class="hint" title="Optional lexical reranking window. 0 disables reranking.">i</span></span><input name="rerank_top_n" type="number" min="0" value="0" /></label>
            <label data-show-if="rerank_top_n>0" class="inactive-field"><span class="label-row">Rerank weight <span class="hint" title="How strongly lexical reranking influences scores.">i</span></span><input name="rerank_weight" type="number" step="0.05" value="0.25" /></label>
          </div>

          <div class="grid setting-group" data-show-for="ted_cpv" style="margin-top:10px">
            <label class="check" style="grid-column:1/-1"><input type="checkbox" name="cross_encoder_rerank" /> Cross-encoder rerank <span class="hint" title="Rerank the top-N CPV candidates with a cross-encoder after retrieval and KG.">i</span></label>
            <label data-show-if="cross_encoder_rerank" class="inactive-field"><span class="label-row">Cross-encoder top N <span class="hint" title="How many top candidates the cross-encoder may inspect.">i</span></span><input name="cross_encoder_top_n" type="number" min="1" value="10" /></label>
            <label data-show-if="cross_encoder_rerank" class="inactive-field"><span class="label-row">Cross-encoder model <span class="hint" title="HuggingFace cross-encoder model. Leave empty for the default multilingual model.">i</span></span><input name="cross_encoder_model" placeholder="Alibaba-NLP/gte-multilingual-reranker-base" /></label>
            <label class="check" style="grid-column:1/-1"><input type="checkbox" name="llm_rerank" /> LLM rerank <span class="hint" title="Rerank top CPV candidates with the configured OpenAI model after retrieval and optional cross-encoder.">i</span></label>
            <label data-show-if="llm_rerank" class="inactive-field"><span class="label-row">LLM rerank top N <span class="hint" title="How many top candidates to pass to the LLM reranker. Shortlists around 5-8 are usually safer than 30.">i</span></span><input name="llm_rerank_top_n" type="number" min="1" value="8" /></label>
            <label data-show-if="llm_rerank" class="inactive-field"><span class="label-row">LLM rerank weight <span class="hint" title="How strongly the LLM reranker can override retrieval, KG, and cross-encoder scores. Lower values are more conservative.">i</span></span><input name="llm_rerank_weight" type="number" min="0" max="1" step="0.05" value="0.4" /></label>
          </div>

          <div class="grid setting-group" data-show-for="document_qa examination_regulations ted_cpv sweep" style="margin-top:10px">
            <label data-show-if="llm_enable|llm_rerank|judge_enable|query_augmentation:llm|query_augmentation:hyde|query_augmentation:translate_en|self_rag_retry_on_weak_evidence|self_rag_critique" class="inactive-field"><span class="label-row">LLM model <span class="hint" title="OpenAI model used for LLM answer generation, LLM reranking, query augmentation, HyDE, or Self-RAG steps.">i</span></span><input name="llm_model" value="gpt-5.4" /></label>
            <label data-show-if="llm_enable|llm_rerank|judge_enable|query_augmentation:llm|query_augmentation:hyde|query_augmentation:translate_en|self_rag_retry_on_weak_evidence|self_rag_critique" class="inactive-field"><span class="label-row">API key env <span class="hint" title="Environment variable containing the OpenAI API key. Required for LLM, reranking, HyDE, or Self-RAG steps.">i</span></span><input name="openai_api_key_env" value="OPENAI_API_KEY" /></label>
            <label data-show-if="judge_enable" class="inactive-field"><span class="label-row">Judge model <span class="hint" title="Optional separate OpenAI model for claim-level judging.">i</span></span><input name="judge_model" placeholder="defaults to LLM model" /></label>
          </div>
          <div class="grid setting-group" data-show-for="document_qa examination_regulations ted_cpv sweep" style="margin-top:10px">
            <label><span class="label-row">Query augmentation <span class="hint" title="Use structured procurement query enrichment (translate_en), LLM expansion, or HyDE before CPV or document search.">i</span></span><select name="query_augmentation"><option value="">profile/default none</option><option>none</option><option>translate_en</option><option>llm</option><option>hyde</option></select></label>
            <label data-show-if="query_augmentation:llm|query_augmentation:hyde|query_augmentation:translate_en" class="inactive-field"><span class="label-row">Augment terms <span class="hint" title="Maximum English terms for LLM expansion. translate_en and HyDE ignore this.">i</span></span><input name="query_augmentation_max_terms" type="number" min="1" value="8" /></label>
          </div>
          <div class="grid setting-group" data-show-for="document_qa examination_regulations sweep" style="margin-top:10px">
            <label><span class="label-row">Answer mode <span class="hint" title="Optional answer synthesis override.">i</span></span><select name="answer_mode"><option value="" selected>profile default</option><option>extractive</option><option>grounded_llm</option><option>cite_first</option><option>claim_checklist</option></select></label>
            <label><span class="label-row">Context mode <span class="hint" title="Optional context assembly override. kg_organized orders context by KG evidence paths.">i</span></span><select name="context_mode"><option value="" selected>profile default</option><option>ranked</option><option>kg_first</option><option>kg_organized</option><option>group_by_doc</option><option>dedupe_section</option></select></label>
            <label><span class="label-row">Max context chunks <span class="hint" title="Optional limit on chunks passed to answer and metrics.">i</span></span><input name="max_context_chunks" type="number" min="1" placeholder="profile default" /></label>
            <label><span class="label-row">Max context chars <span class="hint" title="Optional total character budget for assembled context.">i</span></span><input name="max_context_chars" type="number" min="1" placeholder="profile default" /></label>
            <label><span class="label-row">Min confidence <span class="hint" title="Decision policy threshold for auto-accept.">i</span></span><input name="decision_min_confidence" type="number" step="0.05" min="0" max="1" placeholder="profile default" /></label>
            <label><span class="label-row">Min claim recall <span class="hint" title="Decision policy threshold for context claim coverage.">i</span></span><input name="decision_min_context_claim_recall" type="number" step="0.05" min="0" max="1" placeholder="profile default" /></label>
            <label><span class="label-row">Min grounded <span class="hint" title="Decision policy threshold for grounded claim ratio.">i</span></span><input name="decision_min_grounded_claim_ratio" type="number" step="0.05" min="0" max="1" value="1.0" /></label>
          </div>
          <div class="checks setting-group" data-show-for="document_qa examination_regulations ted_cpv sweep">
            <label class="check"><input type="checkbox" name="kg_enable" /> KG retrieval <span class="hint" title="Build and use a lightweight knowledge graph for graph-augmented retrieval.">i</span></label>
            <label class="check"><input type="checkbox" name="export_kg_for_neo4j" /> Export KG for Neo4j <span class="hint" title="Write CPV KG nodes/edges CSV plus an import.cypher script into the run outputs.">i</span></label>
          </div>
          <div class="grid setting-group inactive-field" data-show-for="document_qa examination_regulations ted_cpv sweep" data-show-if="kg_enable" style="margin-top:10px">
            <label><span class="label-row">KG weight <span class="hint" title="Weight of graph signals when KG retrieval is enabled.">i</span></span><input type="number" step="0.05" min="0" max="1" name="kg_graph_weight" value="0.35" /></label>
            <label><span class="label-row">KG profile <span class="hint" title="Preset for KG pool expansion. safe_branch only expands locally when the retrieved pool already shows branch support and is the safest TED/CPV option.">i</span></span><select name="kg_profile"><option>safe_branch</option><option>selection</option><option>balanced</option><option>conservative</option><option>exploratory</option><option>ppr_only</option><option>direct_only</option></select></label>
            <label><span class="label-row">KG algorithm <span class="hint" title="Optional override for the graph traversal algorithm.">i</span></span><select name="kg_algorithm"><option value="">profile default</option><option>direct</option><option>ppr</option><option>ppr_direct</option></select></label>
            <label><span class="label-row">KG max added <span class="hint" title="Optional max graph-only chunks or candidates added by KG.">i</span></span><input name="kg_max_added_chunks" type="number" min="0" placeholder="profile default" /></label>
            <label><span class="label-row">PPR iterations <span class="hint" title="Optional Personalized PageRank-style propagation iterations for PDF KG.">i</span></span><input name="kg_ppr_iterations" type="number" min="0" placeholder="profile default" /></label>
            <label><span class="label-row">PPR damping <span class="hint" title="Optional propagation damping factor for PDF KG.">i</span></span><input name="kg_ppr_damping" type="number" step="0.01" min="0" max="1" placeholder="profile default" /></label>
            <label><span class="label-row">KG quality min <span class="hint" title="Optional minimum quality factor for graph-only chunks.">i</span></span><input name="kg_quality_threshold" type="number" step="0.05" min="0" max="1" placeholder="profile default" /></label>
            <label><span class="label-row">Intent weight <span class="hint" title="Optional contribution of relation intent matching to KG ranking.">i</span></span><input name="kg_intent_weight" type="number" step="0.05" min="0" max="1" placeholder="profile default" /></label>
            <label><span class="label-row">Edge dropouts <span class="hint" title="Comma-separated KG edge dropout rates for incompleteness testing, e.g. 0.1,0.3.">i</span></span><input name="kg_ablation_edge_dropouts" placeholder="off" /></label>
          </div>
          <div class="checks setting-group" data-show-for="document_qa examination_regulations sweep">
            <label class="check"><input type="checkbox" name="llm_enable" /> LLM answers <span class="hint" title="Generate grounded answers from retrieved chunks instead of extractive fallback only.">i</span></label>
            <label class="check"><input type="checkbox" name="judge_enable" /> LLM judge <span class="hint" title="Use an LLM judge for claim-level support and contradiction metrics.">i</span></label>
            <label class="check"><input type="checkbox" name="abstain_on_weak_evidence" /> Abstain weak <span class="hint" title="Abstain when runtime retrieval signals say evidence is weak or missing.">i</span></label>
            <label class="check"><input type="checkbox" name="self_rag_retry_on_weak_evidence" /> Self-RAG retry <span class="hint" title="Rewrite the query and retrieve again when runtime evidence is weak before answering.">i</span></label>
            <label class="check"><input type="checkbox" name="self_rag_critique" /> Self-RAG critique <span class="hint" title="Critique and revise generated answers against retrieved context before scoring.">i</span></label>
          </div>
          <div class="grid setting-group" data-show-for="document_qa examination_regulations sweep" style="margin-top:10px">
            <label data-show-if="self_rag_retry_on_weak_evidence" class="inactive-field"><span class="label-row">Retry attempts <span class="hint" title="Maximum query rewrite/retrieval retries for weak runtime evidence.">i</span></span><input name="self_rag_retry_max_attempts" type="number" min="1" value="1" /></label>
          </div>
          <div class="toolbar" style="margin-top:14px">
            <button class="primary" type="submit" id="startBtn">Start evaluation</button>
            <span class="meta" id="submitStatus"></span>
          </div>
        </form>
      </div>
      <div>
        <h2>Runs</h2>
        <div class="runs" id="runsList"></div>
      </div>
    </aside>
    <section class="workspace">
      <div id="details" class="empty">Select a run to view metrics, outputs, logs, and CSV previews.</div>
    </section>
  </main>
  <section id="chatApp" class="chat-shell hidden">
    <div id="chatView"></div>
  </section>
  <script>
    const state = { runs: [], selected: null, details: null, activeTable: null, options: null, workspace: "admin", chatTurns: [], customSweepConfigs: [] };
    const csvCandidates = [
      "experiment_ranking_csv",
      "design_trace_csv",
      "design_factor_effects_csv",
      "paired_design_comparisons_csv",
      "failure_attribution_csv",
      "group_failure_explanations_csv",
      "component_attribution_csv",
      "failure_taxonomy_csv",
      "task_type_attribution_csv",
      "design_pareto_frontier_csv",
      "cost_latency_attribution_csv",
      "parsing_diagnostics_csv",
      "rag_results_csv",
      "retrieval_metrics_csv",
      "answer_metrics_csv",
      "diagnostics_csv",
      "retrieved_chunks_csv"
    ];

    function defaultRunName() {
      const now = new Date();
      const pad = value => String(value).padStart(2, "0");
      return `web_eval_${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    }

    function fmt(value) {
      if (value === null || value === undefined || value === "") return "n/a";
      if (typeof value === "number") return Math.abs(value) < 1 ? value.toFixed(3) : value.toLocaleString();
      return String(value);
    }

    function badge(status) {
      const safe = (status || "unknown").toLowerCase();
      return `<span class="status ${safe}">${safe}</span>`;
    }

    async function api(path, options) {
      const response = await fetch(path, options);
      const text = await response.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch { data = text; }
      if (!response.ok) throw new Error(data && data.error ? data.error : response.statusText);
      return data;
    }

    function evalShowIf(form, spec) {
      if (!spec) return true;
      if (spec.includes("|")) {
        return spec.split("|").some(part => evalShowIf(form, part.trim()));
      }
      const gt = spec.match(/^([a-z0-9_]+)>(-?\\d+(?:\\.\\d+)?)$/i);
      if (gt) {
        const el = form.elements[gt[1]];
        return Number(el?.value || 0) > Number(gt[2]);
      }
      const colon = spec.indexOf(":");
      if (colon >= 0) {
        const name = spec.slice(0, colon);
        const values = spec.slice(colon + 1).split(",").filter(Boolean);
        const el = form.elements[name];
        if (!el) return false;
        if (el.type === "checkbox") {
          return values.some(value => (value === "true" || value === "1") && el.checked);
        }
        return values.includes(el.value);
      }
      const el = form.elements[spec];
      if (!el) return false;
      return el.type === "checkbox" ? el.checked : Boolean(String(el.value || "").trim());
    }

    function applyConditionalVisibility() {
      const form = document.getElementById("runForm");
      document.querySelectorAll("[data-show-if]").forEach(node => {
        const visible = evalShowIf(form, node.dataset.showIf);
        node.classList.toggle("inactive-field", !visible);
        node.querySelectorAll("input, select, textarea").forEach(control => {
          control.disabled = !visible;
        });
      });
    }

    function pruneInactivePayload(payload, form) {
      const inactive = new Set();
      form.querySelectorAll(".setting-group.hidden [name], .inactive-field [name]").forEach(el => {
        if (el.name) inactive.add(el.name);
      });
      for (const name of inactive) {
        if (name === "selected_docs") continue;
        if (name === "retriever" && ["ted_cpv", "api_classifier"].includes(payload.classifier_type)) continue;
        delete payload[name];
      }
      for (const box of form.querySelectorAll('input[type="checkbox"]')) {
        if (inactive.has(box.name)) payload[box.name] = false;
      }
      if (!payload.kg_enable) {
        ["kg_graph_weight","kg_profile","kg_algorithm","kg_max_added_chunks","kg_ppr_iterations","kg_ppr_damping","kg_quality_threshold","kg_intent_weight","kg_ablation_edge_dropouts"].forEach(key => delete payload[key]);
      }
      if (!payload.cross_encoder_rerank) {
        delete payload.cross_encoder_model;
        delete payload.cross_encoder_top_n;
      }
      if (!payload.llm_rerank) {
        delete payload.llm_rerank_top_n;
        delete payload.llm_rerank_weight;
      }
      const chunking = String(payload.chunking || "");
      if (chunking !== "auto") {
        delete payload.auto_chunk_sizes;
        delete payload.auto_overlaps;
      }
      if (!["fixed_words", "fixed_tokens"].includes(chunking)) {
        delete payload.chunk_size;
        delete payload.overlap;
      }
      const retriever = String(payload.retriever || "");
      if (!["hybrid", "auto"].includes(retriever)) delete payload.hybrid_alpha;
      if (retriever !== "auto") delete payload.auto_retrievers;
      if (Number(payload.rerank_top_n || 0) <= 0) delete payload.rerank_weight;
      if (!payload.self_rag_retry_on_weak_evidence) delete payload.self_rag_retry_max_attempts;
      const queryAug = String(payload.query_augmentation || "").trim().toLowerCase();
      if (!["llm", "hyde", "translate_en"].includes(queryAug)) delete payload.query_augmentation_max_terms;
      if (!queryAug || queryAug === "none") delete payload.query_augmentation;
      const llmNeeded = payload.llm_enable || payload.llm_rerank || payload.judge_enable || ["llm", "hyde", "translate_en"].includes(queryAug) || payload.self_rag_retry_on_weak_evidence || payload.self_rag_critique;
      if (!(payload.llm_enable || payload.llm_rerank || payload.judge_enable || ["llm", "hyde", "translate_en"].includes(queryAug) || payload.self_rag_retry_on_weak_evidence || payload.self_rag_critique)) {
        delete payload.llm_model;
        delete payload.llm_temperature;
      }
      if (!payload.judge_enable) {
        delete payload.judge_model;
        delete payload.judge_temperature;
      }
      if (!llmNeeded) delete payload.openai_api_key_env;
    }

    function formPayload(form) {
      applyVisibility();
      applyConditionalVisibility();
      const data = new FormData(form);
      const payload = {};
      for (const [key, value] of data.entries()) payload[key] = value;
      for (const box of form.querySelectorAll('input[type="checkbox"]')) {
        if (box.disabled) {
          payload[box.name] = false;
          continue;
        }
        payload[box.name] = box.checked;
      }
      payload.selected_docs = Array.from(form.querySelectorAll('input[name="selected_docs"]:checked')).map(input => input.value);
      if (!payload.run_name) payload.run_name = defaultRunName();
      if (payload.classifier_type === "ted_cpv" || payload.classifier_type === "api_classifier") {
        payload.retriever = payload.cpv_retriever || "tfidf";
      }
      if (payload.classifier_type === "prepared_rag_results") {
        payload.cpv_catalog = payload.prepared_cpv_catalog || payload.cpv_catalog;
      }
      if (payload.mode === "sweep") {
        payload.questions = payload.sweep_questions || payload.questions;
        payload.hybrid_alpha = payload.sweep_hybrid_alpha || payload.hybrid_alpha;
        const builderMode = data.get("sweep_builder_mode") || "all";
        if (builderMode === "ordered") {
          if (!state.customSweepConfigs.length) throw new Error("Add at least one ordered sweep combination.");
          payload.sweep_configs_json = JSON.stringify(state.customSweepConfigs);
          payload.chunking = "fixed_words";
          payload.retriever = "tfidf";
        } else {
          const configs = generatedSweepConfigs();
          if (!configs.length) throw new Error("Select at least one sweep combination.");
          payload.sweep_configs_json = JSON.stringify(configs);
          payload.chunking = "fixed_words";
          payload.retriever = "tfidf";
        }
      }
      for (const key of ["top_k","chunk_size","overlap","rerank_top_n","cross_encoder_top_n","llm_rerank_top_n","self_rag_retry_max_attempts"]) payload[key] = Number(payload[key] || 0);
      for (const key of ["rerank_weight","hybrid_alpha","llm_rerank_weight"]) {
        if (key in payload) payload[key] = Number(payload[key] || 0);
      }
      pruneInactivePayload(payload, form);
      return payload;
    }

    function classifierContext() {
      const form = document.getElementById("runForm");
      const mode = form.elements.mode.value;
      return mode === "sweep" ? "sweep" : form.elements.classifier_type.value;
    }

    function applyVisibility() {
      const context = classifierContext();
      document.querySelectorAll(".setting-group").forEach(group => {
        const allowed = (group.dataset.showFor || "").split(/\s+/);
        group.classList.toggle("hidden", !allowed.includes(context));
      });
      applyConditionalVisibility();
      updateSweepBuilder();
    }

    function selectedSweepValues(kind) {
      return Array.from(document.querySelectorAll(`input[data-sweep-${kind}]:checked`)).map(input => input.value);
    }

    function generatedSweepConfigs() {
      const chunking = selectedSweepValues("chunking");
      const retrievers = selectedSweepValues("retriever");
      const sizes = selectedSweepValues("size").map(Number);
      const overlaps = selectedSweepValues("overlap").map(Number);
      const answerModes = selectedSweepValues("answer-mode");
      const contextModes = selectedSweepValues("context-mode");
      const kgEnabledValues = selectedSweepValues("kg-enabled");
      const judgeEnabledValues = selectedSweepValues("judge-enable");
      const abstainValues = selectedSweepValues("abstain");
      const retryValues = selectedSweepValues("retry");
      const critiqueValues = selectedSweepValues("critique");
      const configs = [];
      for (const strategy of chunking) {
        for (const retriever of retrievers) {
          for (const answer_mode of (answerModes.length ? answerModes : ["cite_first"])) {
            for (const context_mode of (contextModes.length ? contextModes : ["dedupe_section"])) {
              for (const kg_enabled of (kgEnabledValues.length ? kgEnabledValues : ["false"])) {
                for (const judge_enable of (judgeEnabledValues.length ? judgeEnabledValues : ["false"])) {
                  for (const abstain_on_weak_evidence of (abstainValues.length ? abstainValues : ["false"])) {
                    for (const self_rag_retry_on_weak_evidence of (retryValues.length ? retryValues : ["false"])) {
                      for (const self_rag_critique of (critiqueValues.length ? critiqueValues : ["false"])) {
                        const baseConfig = {
                          chunking: strategy,
                          retriever,
                          answer_mode,
                          context_mode,
                          kg_enabled: kg_enabled === "true",
                          judge_enable: judge_enable === "true",
                          abstain_on_weak_evidence: abstain_on_weak_evidence === "true",
                          self_rag_retry_on_weak_evidence: self_rag_retry_on_weak_evidence === "true",
                          self_rag_critique: self_rag_critique === "true",
                        };
                        if (["fixed_words", "fixed_tokens"].includes(strategy)) {
                          for (const chunkSize of (sizes.length ? sizes : [450])) {
                            for (const overlap of (overlaps.length ? overlaps : [60])) {
                              configs.push({ ...baseConfig, chunk_size: chunkSize, overlap });
                            }
                          }
                        } else {
                          configs.push({ ...baseConfig, chunk_size: 0, overlap: 0 });
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      return configs;
    }

    function comboLabel(config, index) {
      const fixed = ["fixed_words", "fixed_tokens"].includes(config.chunking);
      const size = fixed ? ` · size ${config.chunk_size} · overlap ${config.overlap}` : "";
      const extras = [
        `ans ${config.answer_mode || "profile_default"}`,
        `ctx ${config.context_mode || "profile_default"}`,
        `kg ${config.kg_enabled ? "on" : "off"}`,
        `judge ${config.judge_enable ? "on" : "off"}`,
        `abstain ${config.abstain_on_weak_evidence ? "on" : "off"}`,
        `retry ${config.self_rag_retry_on_weak_evidence ? "on" : "off"}`,
        `critique ${config.self_rag_critique ? "on" : "off"}`,
      ];
      return `${index + 1}. ${config.chunking}${size} · ${config.retriever} · ${extras.join(" · ")}`;
    }

    function renderCustomSweepList() {
      const list = document.getElementById("customSweepList");
      if (!list) return;
      list.innerHTML = state.customSweepConfigs.map((config, index) => `
        <div class="combo-row">
          <strong>#${index + 1}</strong>
          <div>${htmlEscape(comboLabel(config, index).replace(/^\d+\.\s*/, ""))}</div>
          <div class="combo-actions">
            <button type="button" onclick="moveSweepCombo(${index}, -1)" ${index === 0 ? "disabled" : ""}>Up</button>
            <button type="button" onclick="moveSweepCombo(${index}, 1)" ${index === state.customSweepConfigs.length - 1 ? "disabled" : ""}>Down</button>
            <button type="button" onclick="removeSweepCombo(${index})">Remove</button>
          </div>
        </div>
      `).join("") || '<div class="empty">No ordered combinations yet.</div>';
    }

    function updateSweepBuilder() {
      const selectedMode = document.querySelector('input[name="sweep_builder_mode"]:checked')?.value || "all";
      document.querySelectorAll(".ordered-sweep-controls").forEach(node => node.classList.toggle("hidden", selectedMode !== "ordered"));
      const preview = document.getElementById("sweepPreview");
      if (!preview) return;
      if (selectedMode === "ordered") {
        renderCustomSweepList();
        preview.textContent = `${state.customSweepConfigs.length} ordered combination${state.customSweepConfigs.length === 1 ? "" : "s"} will run exactly in this order.`;
      } else {
        const configs = generatedSweepConfigs();
        preview.textContent = `${configs.length} combination${configs.length === 1 ? "" : "s"} selected. Fixed strategies combine selected sizes and overlaps; section/paragraph run once per retriever.`;
      }
    }

    function renderPdfPicker(pdfs, targetId="pdfPicker", inputName="selected_docs", selectedPaths=null) {
      const picker = document.getElementById(targetId);
      if (!picker) return;
      const selected = new Set(
        Array.isArray(selectedPaths) && selectedPaths.length
          ? selectedPaths
          : (Array.isArray(state.options?.default_selected_pdfs) && state.options.default_selected_pdfs.length
              ? state.options.default_selected_pdfs
              : (pdfs || []))
      );
      picker.innerHTML = (pdfs || []).map(path => `
        <label><input type="checkbox" name="${inputName}" value="${htmlEscape(path)}" ${selected.has(path) ? "checked" : ""} /> <span>${htmlEscape(path.replace(/^data\/files\//, ""))}</span></label>
      `).join("") || '<div class="empty">No PDFs found in data/files/.</div>';
    }

    function setWorkspace(view) {
      state.workspace = view;
      document.getElementById("adminTabBtn").classList.toggle("active", view === "admin");
      document.getElementById("chatTabBtn").classList.toggle("active", view === "chat");
      document.getElementById("adminApp").classList.toggle("hidden", view !== "admin");
      document.getElementById("chatApp").classList.toggle("hidden", view !== "chat");
      if (view === "chat") renderChat();
    }

    async function refreshRuns(selectLatest=false) {
      state.runs = await api("/api/runs");
      const list = document.getElementById("runsList");
      list.innerHTML = state.runs.map(run => `
        <div class="run-row ${state.selected === run.run_name ? "active" : ""}" data-run="${run.run_name}">
          <div class="run-title"><span>${run.run_name}</span>${badge(run.status)}</div>
          <div class="meta">${fmt(run.mode)} ${run.classifier_type ? " / " + run.classifier_type : ""}</div>
          <div class="meta">${fmt(run.best_experiment || run.retriever_mode || run.chunking_mode)} · ${fmt(run.n_questions)} questions</div>
        </div>
      `).join("") || '<div class="empty">No runs yet.</div>';
      list.querySelectorAll(".run-row").forEach(row => row.addEventListener("click", () => loadRun(row.dataset.run)));
      if (selectLatest && state.runs[0]) await loadRun(state.runs[0].run_name);
    }

    function flattenOutputs(summary) {
      const outputs = Object.assign({}, summary.outputs || {});
      const classifier = summary.classifier_summary || {};
      Object.assign(outputs, classifier.outputs || {});
      for (const experiment of summary.experiments || []) {
        for (const [key, value] of Object.entries(experiment.outputs || {})) {
          outputs[`${experiment.experiment}:${key}`] = value;
        }
      }
      return outputs;
    }

    function collectRecommendations(summary) {
      const design = summary.design_attribution || (summary.classifier_summary && summary.classifier_summary.design_attribution) || {};
      const rootCauses = design.root_cause_recommendations || [];
      const advisor = (summary.classifier_summary && summary.classifier_summary.advisor) || {};
      const classifierSummary = summary.classifier_summary || {};
      const evidenceGraph = classifierSummary.evidence_graph || {};
      const evidenceRates = evidenceGraph.rates || {};
      const componentSignals = evidenceGraph.component_signals || {};
      const experiments = summary.experiments || [];
      function inferRecommendationSource(item) {
        if (item.recommendation_source) return item.recommendation_source;
        const component = String(item.component || "");
        const sameDivisionRate = Number(evidenceRates.same_division_error_rate || 0);
        const hierarchySignal = Number(componentSignals.hierarchy || 0);
        const selectionSignal = Number(componentSignals.selection || 0);
        const querySignal = Number(componentSignals.query_enrichment || 0);
        if (component === "query_enrichment" && querySignal > 0) return "Graph-aware";
        if (component === "hierarchy" && (hierarchySignal > 0 || sameDivisionRate >= 0.2)) return "Graph-aware";
        if ((component === "selection" || component === "reranking") && (selectionSignal > 0 || sameDivisionRate >= 0.2)) return "Graph-aware";
        if (component === "retriever" && Number(componentSignals.retriever || 0) > 0) return "Graph-aware";
        if (component === "examples" && Number(componentSignals.examples || 0) > 0) return "Graph-aware";
        if (component === "deduplication" && Number(componentSignals.deduplication || 0) > 0) return "Graph-aware";
        const text = [
          item.issue || "",
          item.recommendation || "",
          item.evidence || "",
          item.implementation_hint || "",
          item.next_experiment || "",
        ].join(" ").toLowerCase();
        if (text.includes("evidence graph")) return "Graph-aware";
        return "Advisor";
      }
      let items = [];
      if (rootCauses.length) {
        items = items.concat(rootCauses.map(item => Object.assign({ recommendation_source: "Design attribution" }, item)));
      }
      const bestName = summary.best_experiment && summary.best_experiment.experiment;
      if (bestName && experiments.length) {
        const bestExperiment = experiments.find(experiment => experiment.experiment === bestName);
        const bestAdvisor = (bestExperiment && bestExperiment.advisor) || {};
        items = items
          .concat((bestAdvisor.summary_recommendations || []).map(item => Object.assign({ experiment: bestName }, item)))
          .concat((bestAdvisor.top_recommendations || []).map(item => Object.assign({ experiment: bestName }, item)));
      }
      if (!items.length) {
        items = []
          .concat(advisor.summary_recommendations || [])
          .concat(advisor.top_recommendations || []);
      } else {
        items = items
          .concat(advisor.summary_recommendations || [])
          .concat(advisor.top_recommendations || []);
      }
      const seen = new Set();
      return items.filter(item => {
        const source = inferRecommendationSource(item);
        if (!["Graph-aware", "Design attribution"].includes(source)) return false;
        const key = `${item.priority}|${item.component}|${item.issue}|${item.recommendation}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      }).map(item => Object.assign({ recommendation_source: inferRecommendationSource(item) }, item)).slice(0, 14);
    }

    function artifactKind(path) {
      const clean = String(path || "").toLowerCase();
      if (clean.endsWith(".svg")) return "svg";
      if (clean.endsWith(".csv")) return "csv";
      if (clean.endsWith(".json") || clean.endsWith(".jsonl")) return "json";
      if (clean.endsWith(".md")) return "md";
      return "file";
    }

    function artifactCard(runName, key, value) {
      const href = `/api/runs/${encodeURIComponent(runName)}/artifact?path=${encodeURIComponent(value)}`;
      const kind = artifactKind(value);
      const preview = kind === "svg"
        ? `<div class="artifact-preview"><img src="${href}" alt="${htmlEscape(key)} preview" /></div>`
        : `<div class="artifact-preview">${kind.toUpperCase()}</div>`;
      return `<div class="artifact-card">${preview}<a target="_blank" href="${href}">${htmlEscape(key)}</a><div class="meta">${htmlEscape(String(value).split("/").slice(-2).join("/"))}</div></div>`;
    }

    function renderRecommendations(summary) {
      const recommendations = collectRecommendations(summary);
      if (!recommendations.length) return '<div class="empty">No recommendations were generated for this run.</div>';
      return `<div class="recommendations">${recommendations.map(item => `
        <div class="recommendation">
          <div class="source-badge">${htmlEscape(item.recommendation_source || "Advisor")}</div>
          <strong>${htmlEscape(item.priority || "P?")} · ${htmlEscape(item.component || "general")} · ${htmlEscape(item.issue || "Recommendation")}</strong>
          ${item.experiment ? `<div class="meta">Best config: ${htmlEscape(item.experiment)}</div>` : ""}
          ${item.n_cases ? `<div class="meta">Errors in category: ${fmt(item.n_cases)}</div>` : ""}
          <div>${htmlEscape(item.recommendation || "")}</div>
          ${item.evidence ? `<div class="meta">Evidence: ${htmlEscape(item.evidence)}</div>` : ""}
          ${item.likely_causes ? `<div class="meta">Likely causes: ${htmlEscape(item.likely_causes).replaceAll(" | ", "<br />")}</div>` : ""}
          ${item.next_experiment ? `<div class="meta">Next: ${htmlEscape(item.next_experiment)}</div>` : ""}
        </div>
      `).join("")}</div>`;
    }

    function bestExperimentSummary(summary) {
      const bestName = summary.best_experiment && summary.best_experiment.experiment;
      const experiments = summary.experiments || [];
      if (bestName && experiments.length) {
        return experiments.find(experiment => experiment.experiment === bestName) || summary.classifier_summary || {};
      }
      return summary.classifier_summary || {};
    }

    function renderOutcomeCounts(summary) {
      const classifier = bestExperimentSummary(summary);
      const answerMetrics = classifier.answer_metrics || {};
      const classifierType = classifier.classifier?.type || "";
      const isCpv = ["ted_cpv", "api_classifier", "prepared_rag_results"].includes(classifierType);
      const correct = classifier.n_correct ?? answerMetrics.n_correct;
      const incorrect = classifier.n_incorrect ?? answerMetrics.n_incorrect;
      const partial = classifier.n_partial_correct ?? answerMetrics.n_partially_correct ?? 0;
      return `
        <div class="metrics">
          <div class="metric"><div class="meta">Correct</div><div class="value">${fmt(correct)}</div></div>
          <div class="metric"><div class="meta">Incorrect</div><div class="value">${fmt(incorrect)}</div></div>
          ${!isCpv && partial ? `<div class="metric"><div class="meta">Partial correct</div><div class="value">${fmt(partial)}</div></div>` : ""}
        </div>
      `;
    }

    function metricCard(key, value, help, formatter = fmt, source = "") {
      return `
        <div class="metric">
          <div class="meta"><span class="metric-key">${htmlEscape(key)}</span> <span class="hint" title="${htmlEscape(help)}">i</span></div>
          <div class="value">${formatter(value)}</div>
          ${source ? `<div class="metric-source">${htmlEscape(source)}</div>` : ""}
        </div>
      `;
    }

    function pct(value) {
      if (value === null || value === undefined || value === "") return "—";
      const number = Number(value);
      if (!Number.isFinite(number)) return fmt(value);
      return `${(number * 100).toFixed(1)}%`;
    }

    function renderKeyMetrics(summary) {
      const classifier = bestExperimentSummary(summary);
      const answerMetrics = classifier.answer_metrics || {};
      const ranking = classifier.classifier?.ranking_metrics || {};
      const cpvDiagnostics = classifier.classifier?.cpv_diagnostics || {};
      const retrieval = classifier.retrieval_metrics || {};
      const calibration = classifier.classifier?.calibration || {};
      const classifierType = classifier.classifier?.type || "";
      const isCpv = ["ted_cpv", "api_classifier", "prepared_rag_results"].includes(classifierType);
      const cards = isCpv ? [
        metricCard("Top answer exactly correct", answerMetrics.accuracy, "Correct only when the first predicted answer exactly equals the expected answer.", pct, "answer_metrics"),
        metricCard("First ranked candidate correct", ranking.exact_top1_accuracy, "Share of records where the first returned candidate exactly equals the expected answer.", pct, "classifier.ranking_metrics"),
        metricCard("Expected answer appears in candidates", ranking.hit_at_k, "Share of records where the expected answer appears anywhere in the returned top-k candidates.", pct, "classifier.ranking_metrics"),
        metricCard("Expected answer appears early", retrieval.mean_mrr_at_k, "Mean reciprocal rank of the first relevant answer or candidate. Rank 1 gives 1.0, rank 2 gives 0.5, missing gives 0.", fmt, "retrieval_metrics"),
        metricCard("Candidate ranking quality", retrieval.mean_ndcg_at_k, "Ranking quality for returned candidates. Higher means useful candidates appear closer to the top.", fmt, "retrieval_metrics"),
        metricCard("Relevant candidate coverage", retrieval.mean_recall_at_k, "Average share of relevant candidates found in top-k.", pct, "retrieval_metrics"),
        metricCard("Top answer structural closeness", ranking.mean_hierarchy_score_top1, "Diagnostic closeness of the top answer to the expected answer. Exact correctness is still evaluated separately.", fmt, "classifier.ranking_metrics"),
        metricCard("Candidate generation ceiling", cpvDiagnostics.gold_present_at_k_rate, "Share of records where the expected answer was available anywhere in top-k. Low value points to candidate-generation problems.", pct, "classifier.diagnostics"),
        metricCard("Ambiguous top decision rate", cpvDiagnostics.low_margin_decision_rate, "Share of records where rank 1 and rank 2 scores were close. These are good candidates for reranking or manual review.", pct, "classifier.diagnostics"),
        metricCard("High-confidence wrong rate", cpvDiagnostics.high_confidence_wrong_rate, "Share of records where the classifier was wrong despite high confidence. Lower is safer.", pct, "classifier.diagnostics"),
        metricCard("Confidence reliability error", calibration.expected_calibration_error, "Calibration error for prediction scores. Lower is better; high values mean confidence is not reliable.", fmt, "classifier.calibration"),
      ] : [
        metricCard("First useful source appears early", retrieval.mean_mrr_at_k, "How early the first relevant retrieved source appears on average.", fmt, "retrieval_metrics"),
        metricCard("Source ranking quality", retrieval.mean_ndcg_at_k, "Ranking quality for retrieved sources, rewarding useful sources near the top.", fmt, "retrieval_metrics"),
        metricCard("Useful source coverage", retrieval.mean_recall_at_k, "Share of relevant sources found in top-k.", pct, "retrieval_metrics"),
        metricCard("Questions with useful source", retrieval.questions_with_relevant_chunk, "Number of questions with at least one relevant retrieved item.", fmt, "retrieval_metrics"),
        metricCard("Questions with expected source", retrieval.questions_with_target_doc_at_k, "Number of questions where top-k includes the expected source or answer.", fmt, "retrieval_metrics"),
      ];
      return `<div class="metrics">${cards.join("")}</div>`;
    }

    async function loadRun(runName) {
      state.selected = runName;
      state.details = await api(`/api/runs/${encodeURIComponent(runName)}`);
      state.activeTable = null;
      await refreshRuns(false);
      renderDetails();
    }

    function renderDetails() {
      const root = document.getElementById("details");
      const data = state.details;
      if (!data) return;
      const run = data.run;
      const summary = data.summary || {};
      const outputs = flattenOutputs(summary);
      const tablePaths = csvCandidates
        .map(key => [key, outputs[key]])
        .filter(([, value]) => value);
      if (!state.activeTable && tablePaths.length) state.activeTable = tablePaths[0][0];
      const activePath = tablePaths.find(([key]) => key === state.activeTable)?.[1];
      root.className = "";
      root.innerHTML = `
        <div class="band">
          <div class="toolbar" style="justify-content:space-between">
            <div>
              <h2>${run.run_name} ${badge(run.status)}</h2>
              <div class="meta">${htmlEscape(run.run_dir)}</div>
            </div>
            <div class="toolbar">
              ${["running", "queued", "cancelling"].includes(run.status) ? `<button onclick="stopRun('${run.run_name}')" ${run.status === "cancelling" ? "disabled" : ""}>Stop</button>` : ""}
              <button onclick="loadRun('${run.run_name}')">Reload run</button>
            </div>
          </div>
        </div>
        <div class="band metrics">
          <div class="metric"><div class="meta">Best score</div><div class="value">${fmt(run.best_score)}</div></div>
          <div class="metric"><div class="meta">Best experiment</div><div class="value" style="font-size:15px">${fmt(run.best_experiment)}</div></div>
          <div class="metric"><div class="meta">Questions</div><div class="value">${fmt(run.n_questions)}</div></div>
          <div class="metric"><div class="meta">Experiments</div><div class="value">${fmt(run.n_experiments || 1)}</div></div>
        </div>
        <div class="band">
          <h2>Outcome Counts</h2>
          ${renderOutcomeCounts(summary)}
        </div>
        <div class="band">
          <h2>Key Metrics</h2>
          ${renderKeyMetrics(summary)}
        </div>
        <div class="band">
          <h2>CSV previews</h2>
          <div class="tabs">${tablePaths.map(([key]) => `<button class="tab ${key === state.activeTable ? "active" : ""}" onclick="selectTable('${key}')">${key.replace(/_csv$/, "")}</button>`).join("") || '<span class="empty">No CSV outputs.</span>'}</div>
          <div id="tableTarget" class="empty">Loading table...</div>
        </div>
        <div class="band">
          <h2>Artifacts</h2>
          <div class="artifact-grid">${Object.entries(outputs).filter(([, v]) => v).map(([key, value]) => artifactCard(run.run_name, key, value)).join("") || '<span class="empty">No artifacts.</span>'}</div>
        </div>
        <div class="band">
          <h2>Log</h2>
          <pre>${htmlEscape(data.log || "No log yet.")}</pre>
        </div>
        <div class="band">
          <h2>Recommendations</h2>
          ${renderRecommendations(summary)}
        </div>
      `;
      if (activePath) renderTable(activePath);
    }

    async function renderTable(path) {
      const target = document.getElementById("tableTarget");
      try {
        const table = await api(`/api/runs/${encodeURIComponent(state.selected)}/table?path=${encodeURIComponent(path)}&limit=80`);
        target.className = "table-wrap";
        target.innerHTML = `<table><thead><tr>${table.columns.map(col => `<th>${htmlEscape(col)}</th>`).join("")}</tr></thead><tbody>${table.rows.map(row => `<tr>${table.columns.map(col => `<td>${htmlEscape(row[col] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
      } catch (error) {
        target.className = "empty";
        target.textContent = error.message;
      }
    }

    window.selectTable = function(key) {
      state.activeTable = key;
      renderDetails();
    }

    window.stopRun = async function(runName) {
      await api(`/api/runs/${encodeURIComponent(runName)}/stop`, { method: "POST" });
      await loadRun(runName);
    }

    window.moveSweepCombo = function(index, direction) {
      const nextIndex = index + direction;
      if (nextIndex < 0 || nextIndex >= state.customSweepConfigs.length) return;
      const copy = state.customSweepConfigs.slice();
      [copy[index], copy[nextIndex]] = [copy[nextIndex], copy[index]];
      state.customSweepConfigs = copy;
      updateSweepBuilder();
    }

    window.removeSweepCombo = function(index) {
      state.customSweepConfigs = state.customSweepConfigs.filter((_, rowIndex) => rowIndex !== index);
      updateSweepBuilder();
    }

    function htmlEscape(value) {
      return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
    }

    function valueFmtMetric(metric) {
      const value = metric.value;
      if (metric.kind === "boolean") return value ? "yes" : "no";
      if (metric.kind === "percent") return pct(value);
      return fmt(value);
    }

    function renderChat() {
      const root = document.getElementById("chatView");
      root.innerHTML = `
        <div class="chat-page">
          <aside class="chat-config">
            <div>
              <h2>Chat Mode</h2>
              <div class="meta">Ask one question, get a grounded answer, then inspect live metrics for the same turn.</div>
            </div>
            <label><span class="label-row">Mode <span class="hint" title="document_qa_files answers from selected PDFs; api_classifier sends the question to an external classifier endpoint.">i</span></span><select id="chatMode"><option value="document_qa">document_qa_files</option><option value="api_classifier">api_classifier</option></select></label>
            <div class="grid">
              <label><span class="label-row">Top K <span class="hint" title="Initial number of chunks/candidates retrieved before reranking.">i</span></span><input id="chatTopK" type="number" min="1" value="5" /></label>
              <label><span class="label-row">Retriever <span class="hint" title="Retrieval backend for document mode.">i</span></span><select id="chatRetriever"><option>tfidf</option><option>bm25</option><option>dense</option><option>hybrid</option></select></label>
              <label><span class="label-row">Chunking <span class="hint" title="Chunking strategy for document mode.">i</span></span><select id="chatChunking"><option>fixed_words</option><option>fixed_tokens</option><option>by_section</option><option>by_paragraph</option></select></label>
              <label><span class="label-row">Chunk size <span class="hint" title="Size for fixed chunking strategies.">i</span></span><input id="chatChunkSize" type="number" min="0" value="450" /></label>
              <label><span class="label-row">Overlap <span class="hint" title="Overlap for fixed chunking strategies.">i</span></span><input id="chatOverlap" type="number" min="0" value="60" /></label>
              <label><span class="label-row">Rerank top N <span class="hint" title="How many best chunks to keep after reranking the initial Top K. Use 0 to keep all retrieved Top K.">i</span></span><input id="chatRerankTopN" type="number" min="0" value="0" /></label>
            </div>
            <label class="check"><input type="checkbox" id="chatLlmEnable" /> LLM answer <span class="hint" title="Use the configured OpenAI model when the API key env var is available; otherwise fallback to extractive answer.">i</span></label>
            <div class="grid">
              <label><span class="label-row">LLM model <span class="hint" title="Used only when LLM answer is enabled.">i</span></span><input id="chatLlmModel" value="gpt-4.1-mini" /></label>
              <label><span class="label-row">API key env <span class="hint" title="Environment variable containing the OpenAI API key.">i</span></span><input id="chatOpenaiEnv" value="OPENAI_API_KEY" /></label>
            </div>
            <div class="chat-doc-config">
              <label><span class="label-row">Files <span class="hint" title="PDFs used by document_qa_files for this chat turn.">i</span></span><div class="file-picker" id="chatPdfPicker"></div></label>
              <div class="toolbar" style="margin-top:8px">
                <input type="file" id="chatFileUploadInput" multiple accept=".pdf,.pdfa,application/pdf" />
                <button type="button" id="chatUploadDocsBtn">Upload PDFs</button>
              </div>
            </div>
            <div class="chat-api-config hidden">
              <label><span class="label-row">API classifier URL <span class="hint" title="HTTP endpoint that returns answer or ranked predictions for one query.">i</span></span><input id="chatApiUrl" placeholder="https://..." /></label>
              <label><span class="label-row">API token env <span class="hint" title="Environment variable containing a Bearer token for this API.">i</span></span><input id="chatApiTokenEnv" value="API_CLASSIFIER_TOKEN" /></label>
            </div>
          </aside>
          <section class="chat-main">
            <div class="chat-head">
              <div>
                <h2>Ask the classifier</h2>
                <div class="meta" id="chatModeSummary">document_qa_files · metrics after every answer</div>
              </div>
              <button type="button" id="clearChatBtn">Clear chat</button>
            </div>
            <div id="chatTurns" class="chat-thread"></div>
            <div class="chat-composer">
              <div class="composer-row">
                <textarea id="chatQuestion" placeholder="Ask a question about the selected sources..."></textarea>
                <button class="primary" type="button" id="chatSendBtn">Send</button>
              </div>
              <div class="meta" id="chatStatus"></div>
            </div>
          </section>
        </div>
      `;
      renderPdfPicker((state.options && state.options.pdfs) || [], "chatPdfPicker", "chat_selected_docs");
      document.getElementById("chatSendBtn").addEventListener("click", submitChatTurn);
      document.getElementById("clearChatBtn").addEventListener("click", () => {
        state.chatTurns = [];
        renderChatTurns();
      });
      document.getElementById("chatUploadDocsBtn").addEventListener("click", () => uploadDocs("chatFileUploadInput"));
      document.getElementById("chatMode").addEventListener("change", applyChatModeVisibility);
      document.getElementById("chatQuestion").addEventListener("keydown", event => {
        if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) submitChatTurn();
      });
      applyChatModeVisibility();
      renderChatTurns();
    }

    function renderChatTurns() {
      const root = document.getElementById("chatTurns");
      if (!root) return;
      root.innerHTML = state.chatTurns.map(turn => `
        <div class="chat-turn">
          <div class="bubble user">${htmlEscape(turn.question)}</div>
          <div class="bubble assistant">${htmlEscape(turn.answer || "")}</div>
          <div class="chat-metrics">${(turn.metrics || []).map(metric => metricCard(metric.title, metric.value, metric.help, () => valueFmtMetric(metric))).join("")}</div>
          <div class="chat-sources">
            <div class="meta">Sources</div>
            <div class="retrieved-list">
            ${(turn.retrieved || []).map(row => `
              <div class="retrieved-item">
                <strong>#${fmt(row.rank)} ${htmlEscape(row.title || row.chunk_id || "source")}</strong>
                <div class="meta">${htmlEscape(row.doc_id || "")} · score ${fmt(row.score)}</div>
                <div>${htmlEscape(String(row.text || "").slice(0, 700))}</div>
              </div>
            `).join("")}
            </div>
          </div>
        </div>
      `).join("") || '<div class="empty">Ask a question to see the answer, metrics, and sources here.</div>';
      root.scrollTop = root.scrollHeight;
    }

    function applyChatModeVisibility() {
      const mode = document.getElementById("chatMode").value;
      document.querySelectorAll(".chat-doc-config").forEach(node => node.classList.toggle("hidden", mode !== "document_qa"));
      document.querySelectorAll(".chat-api-config").forEach(node => node.classList.toggle("hidden", mode !== "api_classifier"));
      document.getElementById("chatModeSummary").textContent = `${mode === "document_qa" ? "document_qa_files" : "api_classifier"} · metrics after every answer`;
    }

    async function submitChatTurn() {
      const button = document.getElementById("chatSendBtn");
      const status = document.getElementById("chatStatus");
      button.disabled = true;
      status.textContent = "Evaluating...";
      try {
        const payload = {
          chat_mode: document.getElementById("chatMode").value,
          question: document.getElementById("chatQuestion").value,
          top_k: Number(document.getElementById("chatTopK").value || 5),
          retriever: document.getElementById("chatRetriever").value,
          chunking: document.getElementById("chatChunking").value,
          chunk_size: Number(document.getElementById("chatChunkSize").value || 450),
          overlap: Number(document.getElementById("chatOverlap").value || 60),
          rerank_top_n: Number(document.getElementById("chatRerankTopN").value || 0),
          llm_enable: document.getElementById("chatLlmEnable").checked,
          llm_model: document.getElementById("chatLlmModel").value,
          openai_api_key_env: document.getElementById("chatOpenaiEnv").value,
          api_classifier_url: document.getElementById("chatApiUrl").value,
          api_auth_token_env: document.getElementById("chatApiTokenEnv").value,
          selected_docs: Array.from(document.querySelectorAll('input[name="chat_selected_docs"]:checked')).map(input => input.value)
        };
        const result = await api("/api/chat/evaluate", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        state.chatTurns.push(result);
        status.textContent = "Done";
        document.getElementById("chatQuestion").value = "";
        renderChatTurns();
      } catch (error) {
        status.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    }

    function fileToBase64(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || "").split(",")[1] || "");
        reader.onerror = () => reject(reader.error || new Error("Could not read file."));
        reader.readAsDataURL(file);
      });
    }

    async function uploadDocs(inputId="fileUploadInput") {
      const input = document.getElementById(inputId);
      const files = Array.from(input.files || []);
      if (!files.length) return;
      const button = inputId === "chatFileUploadInput" ? document.getElementById("chatUploadDocsBtn") : document.getElementById("uploadDocsBtn");
      button.disabled = true;
      try {
        const payload = { files: await Promise.all(files.map(async file => ({ name: file.name, content_base64: await fileToBase64(file) }))) };
        const result = await api("/api/uploads", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        state.options = result.options;
        renderPdfPicker(state.options.pdfs || []);
        if (state.workspace === "chat") renderChat();
        input.value = "";
      } finally {
        button.disabled = false;
      }
    }

    document.getElementById("runForm").addEventListener("submit", async event => {
      event.preventDefault();
      const button = document.getElementById("startBtn");
      const status = document.getElementById("submitStatus");
      button.disabled = true;
      status.textContent = "Starting...";
      try {
        const result = await api("/api/evaluations", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(formPayload(event.currentTarget))
        });
        status.textContent = `Started ${result.run_name}`;
        await refreshRuns(false);
        await loadRun(result.run_name);
      } catch (error) {
        status.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });

    document.getElementById("refreshBtn").addEventListener("click", () => refreshRuns(false));
    document.getElementById("adminTabBtn").addEventListener("click", () => setWorkspace("admin"));
    document.getElementById("chatTabBtn").addEventListener("click", () => setWorkspace("chat"));
    document.getElementById("runForm").addEventListener("change", applyConditionalVisibility);
    document.getElementById("runForm").elements.mode.addEventListener("change", applyVisibility);
    document.getElementById("runForm").elements.classifier_type.addEventListener("change", applyVisibility);
    document.getElementById("runForm").elements.chunking.addEventListener("change", applyConditionalVisibility);
    document.getElementById("runForm").elements.retriever.addEventListener("change", applyConditionalVisibility);
    document.getElementById("uploadDocsBtn").addEventListener("click", uploadDocs);
    document.getElementById("selectAllDocsBtn").addEventListener("click", () => {
      document.querySelectorAll('input[name="selected_docs"]').forEach(input => input.checked = true);
    });
    document.getElementById("clearDocsBtn").addEventListener("click", () => {
      document.querySelectorAll('input[name="selected_docs"]').forEach(input => input.checked = false);
    });
    document.querySelectorAll('input[name="sweep_builder_mode"]').forEach(input => {
      input.addEventListener("change", updateSweepBuilder);
    });
    document.querySelectorAll("[data-sweep-chunking], [data-sweep-retriever], [data-sweep-size], [data-sweep-overlap]").forEach(input => {
      input.addEventListener("change", updateSweepBuilder);
    });
    document.querySelectorAll("[data-select-sweep]").forEach(button => {
      button.addEventListener("click", () => {
        const kind = button.dataset.selectSweep;
        const inputs = Array.from(document.querySelectorAll(`input[data-sweep-${kind}]`));
        const shouldCheck = inputs.some(input => !input.checked);
        inputs.forEach(input => input.checked = shouldCheck);
        updateSweepBuilder();
      });
    });
    document.getElementById("addSweepComboBtn").addEventListener("click", () => {
      const chunking = document.getElementById("customSweepChunking").value;
      const fixed = ["fixed_words", "fixed_tokens"].includes(chunking);
      state.customSweepConfigs.push({
        chunking,
        retriever: document.getElementById("customSweepRetriever").value,
        chunk_size: fixed ? Number(document.getElementById("customSweepSize").value || 450) : 0,
        overlap: fixed ? Number(document.getElementById("customSweepOverlap").value || 60) : 0,
        answer_mode: document.getElementById("customSweepAnswerMode").value,
        context_mode: document.getElementById("customSweepContextMode").value,
        kg_enabled: document.getElementById("customSweepKgEnabled").value === "true",
        judge_enable: document.getElementById("customSweepJudgeEnable").value === "true",
        abstain_on_weak_evidence: document.getElementById("customSweepAbstain").value === "true",
        self_rag_retry_on_weak_evidence: document.getElementById("customSweepRetry").value === "true",
        self_rag_critique: document.getElementById("customSweepCritique").value === "true",
      });
      updateSweepBuilder();
    });
    document.getElementById("addSelectedSweepCombosBtn").addEventListener("click", () => {
      state.customSweepConfigs.push(...generatedSweepConfigs());
      updateSweepBuilder();
    });
    document.getElementById("clearSweepCombosBtn").addEventListener("click", () => {
      state.customSweepConfigs = [];
      updateSweepBuilder();
    });
    document.getElementById("loadDefaultsBtn").addEventListener("click", async () => {
      const options = await api("/api/options");
      state.options = options;
      renderPdfPicker(options.pdfs || []);
      if (state.workspace === "chat") renderChat();
      const form = document.getElementById("runForm");
      if (options.questions[0]) form.elements.questions.value = options.questions[0];
      if (options.questions[0]) form.elements.sweep_questions.value = options.questions[0];
      if (options.csv.includes("data/teddata_corpus_export.csv")) form.elements.cpv_catalog.value = "data/teddata_corpus_export.csv";
      if (options.csv.includes("data/teddata_corpus_export.csv")) form.elements.prepared_cpv_catalog.value = "data/teddata_corpus_export.csv";
      if (options.json.includes("data/cpv_ted_test_queries.json")) form.elements.cpv_queries.value = "data/cpv_ted_test_queries.json";
      if (options.xlsx && options.xlsx.includes("data/eval_dataset.xlsx")) form.elements.prepared_results.value = "data/eval_dataset.xlsx";
    });

    async function boot() {
      document.getElementById("runForm").elements.run_name.value = defaultRunName();
      state.options = await api("/api/options");
      renderPdfPicker(state.options.pdfs || []);
      applyVisibility();
      updateSweepBuilder();
      await refreshRuns(true);
    }

    boot();
    setInterval(() => {
      if (state.runs.some(run => ["running", "queued"].includes(run.status))) refreshRuns(false);
      if (state.selected) {
        const active = state.runs.find(run => run.run_name === state.selected);
        if (active && ["running", "queued"].includes(active.status)) loadRun(state.selected);
      }
    }, 5000);
  </script>
</body>
</html>
"""


class EvaluationAdminHandler(BaseHTTPRequestHandler):
    server_version = "RAGEvalAdmin/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/":
                return text_response(self, INDEX_HTML, content_type="text/html; charset=utf-8")
            if path == "/api/options":
                return json_response(self, list_data_files())
            if path == "/api/runs":
                return json_response(self, list_runs())
            if path.startswith("/api/runs/"):
                return self.handle_run_get(path, parse_qs(parsed.query))
            return json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            return json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - keeps admin server responsive
            return json_response(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/stop"):
            try:
                parts = parsed.path.strip("/").split("/")
                if len(parts) != 4:
                    return json_response(self, {"error": "Run name is required."}, HTTPStatus.BAD_REQUEST)
                return json_response(self, stop_job(unquote(parts[2])))
            except ValueError as exc:
                return json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        if parsed.path == "/api/uploads":
            try:
                payload = self.read_json_body()
                return json_response(self, save_uploaded_files(payload), HTTPStatus.CREATED)
            except (json.JSONDecodeError, ValueError) as exc:
                return json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        if parsed.path == "/api/chat/evaluate":
            try:
                payload = self.read_json_body()
                return json_response(self, evaluate_chat_turn(payload), HTTPStatus.OK)
            except (json.JSONDecodeError, ValueError) as exc:
                return json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover - chat endpoint should report runtime failures cleanly
                return json_response(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        if parsed.path != "/api/evaluations":
            return json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)
        try:
            payload = self.read_json_body()
            run_name = safe_run_name(payload.get("run_name") or default_run_name())
            run_dir = DEFAULT_OUTPUT_DIR / run_name
            if run_dir.exists() and (run_dir / "run_summary.json").exists():
                run_name = safe_run_name(f"{run_name}_{datetime.now().strftime('%H%M%S')}")
                run_dir = DEFAULT_OUTPUT_DIR / run_name
            command = build_command(payload, run_name, DEFAULT_OUTPUT_DIR)
            job = EvaluationJob(
                run_name=run_name,
                command=command,
                run_dir=run_dir,
                log_path=run_dir / "web_eval.log",
            )
            with JOBS_LOCK:
                JOBS[run_name] = job
            thread = threading.Thread(target=run_job, args=(job,), daemon=True)
            thread.start()
            return json_response(self, job.to_dict(), HTTPStatus.ACCEPTED)
        except (json.JSONDecodeError, ValueError) as exc:
            return json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    def handle_run_get(self, path: str, query: dict[str, list[str]]) -> None:
        parts = path.split("/")
        if len(parts) < 4:
            return json_response(self, {"error": "Run name is required."}, HTTPStatus.BAD_REQUEST)
        run_name = parts[3]
        run_dir = resolve_run_path(run_name)
        if not run_dir.exists():
            return json_response(self, {"error": "Run not found."}, HTTPStatus.NOT_FOUND)

        if len(parts) == 4:
            run = summarize_run(run_dir)
            summary = read_json(run_dir / "run_summary.json") or {}
            log_path = Path(run.get("job", {}).get("log_path") or run_dir / "web_eval.log")
            return json_response(
                self,
                {
                    "run": run,
                    "summary": summary,
                    "log": read_tail(log_path),
                },
            )

        action = parts[4]
        if action == "table":
            raw_path = query.get("path", [""])[0]
            limit = int(query.get("limit", ["80"])[0])
            artifact = resolve_artifact_path(run_dir, raw_path)
            return json_response(self, preview_csv(artifact, max(1, min(limit, 500))))
        if action == "artifact":
            raw_path = query.get("path", [""])[0]
            artifact = resolve_artifact_path(run_dir, raw_path)
            if not artifact.exists():
                return json_response(self, {"error": "Artifact not found."}, HTTPStatus.NOT_FOUND)
            content_type = "text/plain; charset=utf-8"
            if artifact.suffix == ".json":
                content_type = "application/json; charset=utf-8"
            elif artifact.suffix == ".csv":
                content_type = "text/csv; charset=utf-8"
            elif artifact.suffix == ".svg":
                content_type = "image/svg+xml; charset=utf-8"
            data = artifact.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return json_response(self, {"error": "Not found."}, HTTPStatus.NOT_FOUND)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), EvaluationAdminHandler)
    print(f"RAG Evaluation Admin is running at http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the RAG evaluation web/admin interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
