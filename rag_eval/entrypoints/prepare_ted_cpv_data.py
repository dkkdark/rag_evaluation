#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from rag_eval.data.ted_data import (
    TED_DEFAULT_QUERY,
    fetch_ted_notice_pages,
    transform_ted_notices_to_cpv_split_dataset,
    write_cpv_catalog_csv,
    write_queries_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a real CPV/TED baseline dataset from the official TED Search API."
    )
    parser.add_argument(
        "--query",
        default=TED_DEFAULT_QUERY,
        help="TED expert query used to fetch notices.",
    )
    parser.add_argument("--limit", type=int, default=100, help="How many TED notices to fetch per page.")
    parser.add_argument("--start-page", type=int, default=1, help="First Search API page number.")
    parser.add_argument("--pages", type=int, default=4, help="How many Search API pages to fetch.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Stable notice-level split ratio used for catalog training examples.")
    parser.add_argument(
        "--catalog-out",
        default="data/cpv_ted_train_catalog.csv",
        help="Output CSV for the generated CPV catalog.",
    )
    parser.add_argument(
        "--queries-out",
        default="data/cpv_ted_test_queries.json",
        help="Output JSON for generated material queries.",
    )
    parser.add_argument(
        "--raw-out",
        default="data/ted_notice_sample.json",
        help="Optional raw API response dump.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    notices = fetch_ted_notice_pages(
        query=args.query,
        limit=args.limit,
        start_page=args.start_page,
        pages=args.pages,
    )
    catalog, queries, split_summary = transform_ted_notices_to_cpv_split_dataset(
        notices,
        train_ratio=args.train_ratio,
    )
    write_cpv_catalog_csv(args.catalog_out, catalog)
    write_queries_json(args.queries_out, queries)
    with open(args.raw_out, "w", encoding="utf-8") as f:
        json.dump({"notices": notices}, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "n_notices_fetched": len(notices),
                "pages": args.pages,
                "start_page": args.start_page,
                "n_cpv_entries": len(catalog),
                "n_queries": len(queries),
                "split_summary": split_summary,
                "catalog_out": args.catalog_out,
                "queries_out": args.queries_out,
                "raw_out": args.raw_out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
