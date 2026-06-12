#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os

from rag_eval.data.ted_data import (
    TED_DEFAULT_CORPUS_EXPORT_PATH,
    TED_DEFAULT_QUERY,
    TED_DEFAULT_FIELDS,
    TED_MULTILINGUAL_FIELDS,
    fetch_ted_notice_iteration,
    load_ted_corpus_export_notices,
    normalize_ted_notice_record,
)
from rag_eval.data.ted_notice_store import (
    count_rows,
    open_ted_notice_db,
    top_languages,
    upsert_ted_notices,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load TED notices from teddata corpus export or the TED API and store them in a local SQLite database for CPV retrieval enrichment."
    )
    parser.add_argument(
        "--source",
        choices=["auto", "file", "api"],
        default="auto",
        help="Where TED notices should come from. 'auto' prefers the local teddata corpus export if present, otherwise falls back to the API.",
    )
    parser.add_argument(
        "--corpus-export-path",
        default=TED_DEFAULT_CORPUS_EXPORT_PATH,
        help="Path to teddata_corpus_export CSV used when --source=file or when --source=auto finds the file.",
    )
    parser.add_argument(
        "--query",
        default=TED_DEFAULT_QUERY,
        help="TED expert query used to fetch notices.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=250,
        help="Maximum notices per API call (TED Search API iteration mode max is 250).",
    )
    parser.add_argument(
        "--max-notices",
        type=int,
        default=20000,
        help="Maximum number of notices to fetch into the local database.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.35,
        help="Delay between TED Search API iteration calls.",
    )
    parser.add_argument(
        "--db-path",
        default=".rag_eval_indices/ted_notices.sqlite",
        help="Local SQLite file used to store normalized TED notices and CPV notice examples.",
    )
    parser.add_argument(
        "--raw-out",
        default="data/ted_notices_raw_sample.json",
        help="Optional path for a raw sample dump of fetched notices.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    corpus_exists = bool(args.corpus_export_path) and os.path.exists(args.corpus_export_path)
    if args.source == "file":
        notices = load_ted_corpus_export_notices(args.corpus_export_path)
        requested_fields = ["PUB_NUMBER", "TITLE", "TEXT", "BUYER", "LANGUAGE", "CONTRACT_TYPE", "MAIN_CPV"]
        field_profile = f"file:{args.corpus_export_path}"
    elif args.source == "auto" and corpus_exists:
        notices = load_ted_corpus_export_notices(args.corpus_export_path)
        requested_fields = ["PUB_NUMBER", "TITLE", "TEXT", "BUYER", "LANGUAGE", "CONTRACT_TYPE", "MAIN_CPV"]
        field_profile = f"file:{args.corpus_export_path}"
    else:
        try:
            notices = fetch_ted_notice_iteration(
                query=args.query,
                fields=TED_MULTILINGUAL_FIELDS,
                limit=args.limit,
                max_notices=args.max_notices,
                pause_seconds=args.pause_seconds,
            )
            requested_fields = TED_MULTILINGUAL_FIELDS
            field_profile = "api_multilingual"
        except Exception as exc:
            notices = fetch_ted_notice_iteration(
                query=args.query,
                fields=TED_DEFAULT_FIELDS,
                limit=args.limit,
                max_notices=args.max_notices,
                pause_seconds=args.pause_seconds,
            )
            requested_fields = TED_DEFAULT_FIELDS
            field_profile = f"api_fallback_default_fields_after_error:{exc}"
    normalized = [normalize_ted_notice_record(notice) for notice in notices]
    conn = open_ted_notice_db(args.db_path)
    try:
        ingest_summary = upsert_ted_notices(conn, normalized)
        summary = {
            "db_path": args.db_path,
            "query": args.query,
            "field_profile": field_profile,
            "requested_fields": list(requested_fields),
            "n_notices_fetched": len(notices),
            "n_notices_normalized": len(normalized),
            **ingest_summary,
            "ted_notices_total": count_rows(conn, "ted_notices"),
            "cpv_notice_examples_total": count_rows(conn, "cpv_notice_examples"),
            "top_languages": top_languages(conn, limit=25),
        }
    finally:
        conn.close()

    if args.raw_out:
        with open(args.raw_out, "w", encoding="utf-8") as f:
            json.dump({"notices": notices[: min(200, len(notices))]}, f, ensure_ascii=False, indent=2)
        summary["raw_out"] = args.raw_out

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
