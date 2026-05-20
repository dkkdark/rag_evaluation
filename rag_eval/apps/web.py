from __future__ import annotations

import csv
import html
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
RUN_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


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


def safe_run_name(raw_name: str | None) -> str:
    if raw_name:
        cleaned = RUN_NAME_RE.sub("_", raw_name.strip()).strip("._-")
    else:
        cleaned = ""
    if not cleaned:
        cleaned = "web_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    return cleaned[:80]


def default_run_name() -> str:
    return "web_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def evaluation_python() -> str:
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def bool_flag(payload: dict[str, Any], key: str) -> bool:
    return bool(payload.get(key))


def add_value_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    command.extend([flag, str(value)])


def build_command(payload: dict[str, Any], run_name: str, output_dir: Path) -> list[str]:
    classifier_type = str(payload.get("classifier_type", "examination_regulations"))
    selected_docs = payload.get("selected_docs") or []
    if isinstance(selected_docs, str):
        selected_docs = [selected_docs]
    if classifier_type == "examination_regulations":
        if not selected_docs:
            raise ValueError("Select at least one PDF file for examination_regulations.")
        docs_value = ",".join(str(path) for path in selected_docs)
    else:
        docs_value = payload.get("docs", "data/files/**/*.pdf*")

    command = [
        evaluation_python(),
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
        "--cpv-catalog": payload.get("cpv_catalog", "data/cpv_ted_train_catalog.csv"),
        "--cpv-queries": payload.get("cpv_queries", "data/cpv_ted_test_queries.json"),
        "--prepared-results": payload.get("prepared_results", "data/eval_dataset.xlsx"),
        "--api-classifier-url": payload.get("api_classifier_url"),
        "--api-auth-token-env": payload.get("api_auth_token_env"),
        "--chunking": payload.get("chunking", "fixed_words"),
        "--chunk-size": payload.get("chunk_size", 450),
        "--overlap": payload.get("overlap", 60),
        "--top-k": payload.get("top_k", 5),
        "--retriever": payload.get("retriever", "tfidf"),
        "--embedding-model": payload.get("embedding_model"),
        "--auto-chunk-sizes": payload.get("auto_chunk_sizes"),
        "--auto-overlaps": payload.get("auto_overlaps"),
        "--auto-retrievers": payload.get("auto_retrievers"),
        "--hybrid-alpha": payload.get("hybrid_alpha"),
        "--rerank-top-n": payload.get("rerank_top_n"),
        "--rerank-weight": payload.get("rerank_weight"),
        "--weight-answer": payload.get("weight_answer"),
        "--weight-correctness": payload.get("weight_correctness"),
        "--weight-retrieval": payload.get("weight_retrieval"),
        "--llm-model": payload.get("llm_model"),
        "--openai-api-key-env": payload.get("openai_api_key_env"),
        "--llm-temperature": payload.get("llm_temperature"),
        "--judge-model": payload.get("judge_model"),
        "--judge-temperature": payload.get("judge_temperature"),
    }
    if classifier_type != "examination_regulations":
        simple_args["--docs-root"] = payload.get("docs_root", "data/files")
    for flag, value in simple_args.items():
        add_value_arg(command, flag, value)

    command.append("--create-strategy-showcase")
    flags = {
        "--cpv-use-examples": "cpv_use_examples",
        "--llm-enable": "llm_enable",
        "--judge-enable": "judge_enable",
        "--disable-runtime-retrieval-evaluator": "disable_runtime_retrieval_evaluator",
        "--abstain-on-weak-evidence": "abstain_on_weak_evidence",
        "--kg-enable": "kg_enable",
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
            job.process = subprocess.Popen(
                job.command,
                cwd=str(PROJECT_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
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
    return {
        "pdfs": sorted(
            str(path.relative_to(PROJECT_ROOT))
            for path in (PROJECT_ROOT / "data" / "files").glob("**/*.pdf*")
            if path.is_file()
        ),
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
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
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
    .hidden { display: none !important; }
    .empty { color: var(--muted); padding: 16px 0; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>RAG Evaluation Admin</h1>
    <div class="toolbar">
      <button id="refreshBtn" title="Refresh runs">Refresh</button>
      <button id="loadDefaultsBtn" title="Reload detected files">Detect files</button>
    </div>
  </header>
  <main>
    <aside>
      <div class="band">
        <h2>Run settings</h2>
        <form id="runForm">
          <div class="grid">
            <label><span class="label-row">Run name <span class="hint" title="Default format is web_eval_YYYYMMDD_HHMMSS. Leave as is or rename before launch.">i</span></span><input name="run_name" /></label>
            <label><span class="label-row">Mode <span class="hint" title="classifier runs one selected classifier; sweep compares several RAG retrieval/chunking settings.">i</span></span><select name="mode"><option value="classifier">classifier</option><option value="sweep">sweep</option></select></label>
            <label><span class="label-row">Classifier <span class="hint" title="Choose the evaluation target. Only settings relevant to this classifier are shown below.">i</span></span><select name="classifier_type"><option value="examination_regulations">examination_regulations</option><option value="ted_cpv">ted_cpv</option><option value="api_classifier">api_classifier</option><option value="prepared_rag_results">prepared_rag_results</option></select></label>
            <label><span class="label-row">Top K <span class="hint" title="How many candidates/chunks to retrieve for each question. Higher values improve recall but add noise.">i</span></span><input name="top_k" type="number" min="1" value="5" /></label>
          </div>

          <div class="setting-group" data-show-for="examination_regulations sweep" style="margin-top:10px">
            <label><span class="label-row">PDF files <span class="hint" title="Select the exact regulation PDFs for this run. The server passes these files directly to --docs.">i</span></span><div class="file-picker" id="pdfPicker"></div></label>
            <div class="toolbar" style="margin-top:8px">
              <button type="button" id="selectAllDocsBtn">Select all files</button>
              <button type="button" id="clearDocsBtn">Clear</button>
            </div>
          </div>

          <div class="grid setting-group" data-show-for="examination_regulations sweep" style="margin-top:10px">
            <label><span class="label-row">Questions <span class="hint" title="JSON file with evaluation questions and gold answers/keywords.">i</span></span><input name="questions" value="data/questions_by_file.json" /></label>
            <label><span class="label-row">Retriever <span class="hint" title="Retrieval backend. auto compares several retrievers during sweep.">i</span></span><select name="retriever"><option>tfidf</option><option>bm25</option><option>dense</option><option>hybrid</option><option>auto</option></select></label>
            <label><span class="label-row">Chunking <span class="hint" title="How documents are split before retrieval. auto compares several strategies during sweep.">i</span></span><select name="chunking"><option>fixed_words</option><option>fixed_tokens</option><option>by_section</option><option>by_paragraph</option><option>auto</option></select></label>
            <label><span class="label-row">Chunk size <span class="hint" title="Size of fixed chunks. Ignored for section/paragraph chunking.">i</span></span><input name="chunk_size" type="number" min="0" value="450" /></label>
            <label><span class="label-row">Overlap <span class="hint" title="Token/word overlap between adjacent fixed chunks.">i</span></span><input name="overlap" type="number" min="0" value="60" /></label>
            <label><span class="label-row">Hybrid alpha <span class="hint" title="Dense score weight for hybrid retrieval; BM25 gets the remaining weight.">i</span></span><input name="hybrid_alpha" type="number" step="0.05" value="0.5" /></label>
            <label><span class="label-row">Auto sizes <span class="hint" title="Comma-separated chunk sizes used when chunking=auto.">i</span></span><input name="auto_chunk_sizes" value="256,450" /></label>
            <label><span class="label-row">Auto overlaps <span class="hint" title="Comma-separated overlaps used when chunking=auto.">i</span></span><input name="auto_overlaps" value="0,60" /></label>
            <label><span class="label-row">Auto retrievers <span class="hint" title="Comma-separated retrievers used when retriever=auto.">i</span></span><input name="auto_retrievers" value="tfidf,bm25,dense,hybrid" /></label>
          </div>

          <div class="grid setting-group" data-show-for="ted_cpv api_classifier" style="margin-top:10px">
            <label><span class="label-row">CPV catalog <span class="hint" title="Training/catalog CSV with CPV codes and descriptions.">i</span></span><input name="cpv_catalog" value="data/cpv_ted_train_catalog.csv" /></label>
            <label><span class="label-row">CPV queries <span class="hint" title="TED/CPV test queries JSON used for classifier evaluation.">i</span></span><input name="cpv_queries" value="data/cpv_ted_test_queries.json" /></label>
            <label><span class="label-row">Retriever <span class="hint" title="Retriever used by the local CPV classifier. API classifier ignores this.">i</span></span><select name="cpv_retriever"><option>tfidf</option><option>bm25</option><option>dense</option><option>hybrid</option></select></label>
            <label class="check"><input type="checkbox" name="cpv_use_examples" /> Use examples <span class="hint" title="Append real TED examples to CPV labels before ranking.">i</span></label>
          </div>

          <div class="grid setting-group" data-show-for="api_classifier" style="margin-top:10px">
            <label><span class="label-row">API classifier URL <span class="hint" title="HTTP endpoint for external classifier predictions.">i</span></span><input name="api_classifier_url" placeholder="https://..." /></label>
            <label><span class="label-row">API token env <span class="hint" title="Environment variable containing a Bearer token for the classifier API.">i</span></span><input name="api_auth_token_env" value="API_CLASSIFIER_TOKEN" /></label>
          </div>

          <div class="grid setting-group" data-show-for="prepared_rag_results" style="margin-top:10px">
            <label><span class="label-row">Prepared results <span class="hint" title="Excel file with existing RAG/classifier outputs. Multiple rows with one ID are treated as top-k candidates.">i</span></span><input name="prepared_results" value="data/eval_dataset.xlsx" /></label>
            <label><span class="label-row">CPV catalog <span class="hint" title="CPV catalog is used to attach labels and compute hierarchy-aware metrics.">i</span></span><input name="prepared_cpv_catalog" value="data/cpv_ted_train_catalog.csv" /></label>
          </div>

          <div class="grid setting-group" data-show-for="examination_regulations ted_cpv sweep" style="margin-top:10px">
            <label><span class="label-row">Rerank top N <span class="hint" title="Optional lexical reranking window. 0 disables reranking.">i</span></span><input name="rerank_top_n" type="number" min="0" value="0" /></label>
            <label><span class="label-row">Rerank weight <span class="hint" title="How strongly lexical reranking influences scores.">i</span></span><input name="rerank_weight" type="number" step="0.05" value="0.25" /></label>
          </div>

          <div class="grid setting-group" data-show-for="examination_regulations sweep" style="margin-top:10px">
            <label><span class="label-row">LLM model <span class="hint" title="OpenAI model used when LLM answer generation or judging is enabled.">i</span></span><input name="llm_model" value="gpt-4.1-mini" /></label>
            <label><span class="label-row">API key env <span class="hint" title="Environment variable containing the OpenAI API key.">i</span></span><input name="openai_api_key_env" value="OPENAI_API_KEY" /></label>
            <label><span class="label-row">Judge model <span class="hint" title="Optional separate OpenAI model for claim-level judging.">i</span></span><input name="judge_model" placeholder="defaults to LLM model" /></label>
          </div>
          <div class="checks setting-group" data-show-for="examination_regulations sweep">
            <label class="check"><input type="checkbox" name="llm_enable" /> LLM answers <span class="hint" title="Generate grounded answers from retrieved chunks instead of extractive fallback only.">i</span></label>
            <label class="check"><input type="checkbox" name="judge_enable" /> LLM judge <span class="hint" title="Use an LLM judge for claim-level support and contradiction metrics.">i</span></label>
            <label class="check"><input type="checkbox" name="abstain_on_weak_evidence" /> Abstain weak <span class="hint" title="Abstain when runtime retrieval signals say evidence is weak or missing.">i</span></label>
            <label class="check"><input type="checkbox" name="kg_enable" /> KG retrieval <span class="hint" title="Build and use a lightweight knowledge graph for graph-augmented retrieval.">i</span></label>
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
  <script>
    const state = { runs: [], selected: null, details: null, activeTable: null, options: null };
    const csvCandidates = [
      "experiment_ranking_csv",
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

    function formPayload(form) {
      const data = new FormData(form);
      const payload = {};
      for (const [key, value] of data.entries()) payload[key] = value;
      for (const box of form.querySelectorAll('input[type="checkbox"]')) payload[box.name] = box.checked;
      payload.selected_docs = Array.from(form.querySelectorAll('input[name="selected_docs"]:checked')).map(input => input.value);
      if (!payload.run_name) payload.run_name = defaultRunName();
      if (payload.classifier_type === "ted_cpv" || payload.classifier_type === "api_classifier") {
        payload.retriever = payload.cpv_retriever || "tfidf";
      }
      if (payload.classifier_type === "prepared_rag_results") {
        payload.cpv_catalog = payload.prepared_cpv_catalog || payload.cpv_catalog;
      }
      for (const key of ["top_k","chunk_size","overlap","rerank_top_n"]) payload[key] = Number(payload[key] || 0);
      for (const key of ["rerank_weight","hybrid_alpha"]) payload[key] = Number(payload[key] || 0);
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
    }

    function renderPdfPicker(pdfs) {
      const picker = document.getElementById("pdfPicker");
      picker.innerHTML = (pdfs || []).map(path => `
        <label><input type="checkbox" name="selected_docs" value="${htmlEscape(path)}" checked /> <span>${htmlEscape(path.replace(/^data\/files\//, ""))}</span></label>
      `).join("") || '<div class="empty">No PDFs found in data/files/.</div>';
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
      const advisor = (summary.classifier_summary && summary.classifier_summary.advisor) || {};
      const experiments = summary.experiments || [];
      const items = []
        .concat(advisor.summary_recommendations || [])
        .concat(advisor.top_recommendations || []);
      for (const experiment of experiments) {
        const expAdvisor = experiment.advisor || {};
        items.push(...(expAdvisor.summary_recommendations || []));
        items.push(...(expAdvisor.top_recommendations || []));
      }
      const seen = new Set();
      return items.filter(item => {
        const key = `${item.priority}|${item.component}|${item.issue}|${item.recommendation}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      }).slice(0, 12);
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
          <strong>${htmlEscape(item.priority || "P?")} · ${htmlEscape(item.component || "general")} · ${htmlEscape(item.issue || "Recommendation")}</strong>
          <div>${htmlEscape(item.recommendation || "")}</div>
          ${item.evidence ? `<div class="meta">Evidence: ${htmlEscape(item.evidence)}</div>` : ""}
          ${item.next_experiment ? `<div class="meta">Next: ${htmlEscape(item.next_experiment)}</div>` : ""}
        </div>
      `).join("")}</div>`;
    }

    function renderOutcomeCounts(summary) {
      const classifier = summary.classifier_summary || {};
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
      const classifier = summary.classifier_summary || {};
      const answerMetrics = classifier.answer_metrics || {};
      const ranking = classifier.classifier?.ranking_metrics || {};
      const cpvDiagnostics = classifier.classifier?.cpv_diagnostics || {};
      const retrieval = classifier.retrieval_metrics || {};
      const calibration = classifier.classifier?.calibration || {};
      const classifierType = classifier.classifier?.type || "";
      const isCpv = ["ted_cpv", "api_classifier", "prepared_rag_results"].includes(classifierType);
      const cards = isCpv ? [
        metricCard("accuracy", answerMetrics.accuracy, "Exact top-1 correctness rate: correct only when the predicted answer exactly equals the expected answer.", pct, "answer_metrics"),
        metricCard("exact_top1_accuracy", ranking.exact_top1_accuracy, "Share of records where the first predicted answer exactly equals the expected answer.", pct, "classifier.ranking_metrics"),
        metricCard("hit_at_k", ranking.hit_at_k, "Share of records where the expected answer appears anywhere in the returned top-k candidates.", pct, "classifier.ranking_metrics"),
        metricCard("mean_mrr_at_k", retrieval.mean_mrr_at_k, "Mean reciprocal rank of the first relevant answer or candidate. Rank 1 gives 1.0, rank 2 gives 0.5, missing gives 0.", fmt, "retrieval_metrics"),
        metricCard("mean_ndcg_at_k", retrieval.mean_ndcg_at_k, "Mean ranking quality for returned candidates. Higher means better candidates appear closer to the top.", fmt, "retrieval_metrics"),
        metricCard("mean_recall_at_k", retrieval.mean_recall_at_k, "Average share of relevant candidates found in top-k.", pct, "retrieval_metrics"),
        metricCard("mean_hierarchy_score_top1", ranking.mean_hierarchy_score_top1, "Diagnostic closeness of the top answer to the expected answer. Higher means the prediction is structurally closer, but exact correctness is still separate.", fmt, "classifier.ranking_metrics"),
        metricCard("expected_answer_present_at_k_rate", cpvDiagnostics.gold_present_at_k_rate, "Share of records where the expected answer was available anywhere in top-k. Low value points to retriever/candidate-generation problems.", pct, "classifier.diagnostics"),
        metricCard("low_margin_decision_rate", cpvDiagnostics.low_margin_decision_rate, "Share of records where rank 1 and rank 2 scores were close. These are good candidates for reranking or manual review.", pct, "classifier.diagnostics"),
        metricCard("high_confidence_wrong_rate", cpvDiagnostics.high_confidence_wrong_rate, "Share of records where the classifier was wrong despite high confidence. Lower is safer.", pct, "classifier.diagnostics"),
        metricCard("expected_calibration_error", calibration.expected_calibration_error, "Expected calibration error for prediction scores. Lower is better; high values mean confidence is not reliable.", fmt, "classifier.calibration"),
      ] : [
        metricCard("mean_mrr_at_k", retrieval.mean_mrr_at_k, "How early the first relevant retrieved chunk appears on average.", fmt, "retrieval_metrics"),
        metricCard("mean_ndcg_at_k", retrieval.mean_ndcg_at_k, "Ranking quality for retrieved chunks, rewarding relevant chunks near the top.", fmt, "retrieval_metrics"),
        metricCard("mean_recall_at_k", retrieval.mean_recall_at_k, "Share of relevant chunks found in top-k.", pct, "retrieval_metrics"),
        metricCard("questions_with_relevant_chunk", retrieval.questions_with_relevant_chunk, "Number of questions with at least one relevant retrieved item.", fmt, "retrieval_metrics"),
        metricCard("questions_with_target_doc_at_k", retrieval.questions_with_target_doc_at_k, "Number of questions where top-k includes the expected source or answer.", fmt, "retrieval_metrics"),
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

    function htmlEscape(value) {
      return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
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
    document.getElementById("runForm").elements.mode.addEventListener("change", applyVisibility);
    document.getElementById("runForm").elements.classifier_type.addEventListener("change", applyVisibility);
    document.getElementById("selectAllDocsBtn").addEventListener("click", () => {
      document.querySelectorAll('input[name="selected_docs"]').forEach(input => input.checked = true);
    });
    document.getElementById("clearDocsBtn").addEventListener("click", () => {
      document.querySelectorAll('input[name="selected_docs"]').forEach(input => input.checked = false);
    });
    document.getElementById("loadDefaultsBtn").addEventListener("click", async () => {
      const options = await api("/api/options");
      state.options = options;
      renderPdfPicker(options.pdfs || []);
      const form = document.getElementById("runForm");
      if (options.questions[0]) form.elements.questions.value = options.questions[0];
      if (options.csv.includes("data/cpv_ted_train_catalog.csv")) form.elements.cpv_catalog.value = "data/cpv_ted_train_catalog.csv";
      if (options.csv.includes("data/cpv_ted_train_catalog.csv")) form.elements.prepared_cpv_catalog.value = "data/cpv_ted_train_catalog.csv";
      if (options.json.includes("data/cpv_ted_test_queries.json")) form.elements.cpv_queries.value = "data/cpv_ted_test_queries.json";
      if (options.xlsx && options.xlsx.includes("data/eval_dataset.xlsx")) form.elements.prepared_results.value = "data/eval_dataset.xlsx";
    });

    async function boot() {
      document.getElementById("runForm").elements.run_name.value = defaultRunName();
      state.options = await api("/api/options");
      renderPdfPicker(state.options.pdfs || []);
      applyVisibility();
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
        if parsed.path != "/api/evaluations":
            return json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
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
