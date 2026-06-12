from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, Iterable, List, Sequence


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def open_ted_notice_db(path: str) -> sqlite3.Connection:
    ensure_parent_dir(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_ted_notice_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ted_notices (
          publication_number TEXT PRIMARY KEY,
          main_cpv_code TEXT,
          all_cpv_codes_json TEXT NOT NULL,
          notice_title TEXT,
          title_proc TEXT,
          description_proc TEXT,
          description_lot TEXT,
          buyer_name TEXT,
          official_languages_json TEXT NOT NULL,
          language_count INTEGER NOT NULL,
          combined_text TEXT,
          raw_notice_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cpv_notice_examples (
          publication_number TEXT NOT NULL,
          cpv_code TEXT NOT NULL,
          language TEXT NOT NULL,
          source_field TEXT NOT NULL,
          example_text TEXT NOT NULL,
          PRIMARY KEY (publication_number, cpv_code, language, source_field, example_text),
          FOREIGN KEY (publication_number) REFERENCES ted_notices(publication_number)
        );

        CREATE INDEX IF NOT EXISTS idx_ted_notices_main_cpv_code
          ON ted_notices(main_cpv_code);

        CREATE INDEX IF NOT EXISTS idx_cpv_notice_examples_cpv_code
          ON cpv_notice_examples(cpv_code);

        CREATE INDEX IF NOT EXISTS idx_cpv_notice_examples_language
          ON cpv_notice_examples(language);

        CREATE TABLE IF NOT EXISTS cpv_profiles (
          code TEXT PRIMARY KEY,
          label TEXT NOT NULL,
          description_en TEXT,
          parent_code TEXT,
          parent_label TEXT,
          procurement_type TEXT,
          keywords_en TEXT,
          generated_synonyms_en TEXT,
          use_when_text TEXT,
          do_not_use_when_text TEXT,
          common_tender_phrases TEXT,
          children_codes_json TEXT NOT NULL,
          children_labels_json TEXT NOT NULL,
          sibling_codes_json TEXT NOT NULL,
          sibling_labels_json TEXT NOT NULL,
          examples_json TEXT NOT NULL,
          notice_examples_json TEXT NOT NULL,
          text TEXT NOT NULL,
          search_text_en TEXT NOT NULL,
          search_text_multilingual TEXT NOT NULL,
          source_fingerprint TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_cpv_profiles_parent_code
          ON cpv_profiles(parent_code);
        """
    )
    conn.commit()


def _notice_examples_from_record(record: Dict[str, object]) -> List[Dict[str, str]]:
    publication_number = str(record.get("publication_number") or "").strip()
    cpv_codes = [str(code).strip() for code in record.get("all_cpv_codes", []) if str(code).strip()]
    if not cpv_codes and str(record.get("main_cpv_code") or "").strip():
        cpv_codes = [str(record.get("main_cpv_code") or "").strip()]
    languages = [str(language).strip() for language in record.get("official_languages", []) if str(language).strip()]
    if not languages:
        languages = ["und"]
    examples: List[Dict[str, str]] = []
    for field in ["notice_title", "title_proc", "description_proc", "description_lot"]:
        text = str(record.get(field) or "").strip()
        if len(text) < 10:
            continue
        for cpv_code in cpv_codes:
            for language in languages:
                examples.append(
                    {
                        "publication_number": publication_number,
                        "cpv_code": cpv_code,
                        "language": language,
                        "source_field": field,
                        "example_text": text,
                    }
                )
    return examples


def upsert_ted_notices(conn: sqlite3.Connection, records: Sequence[Dict[str, object]]) -> Dict[str, int]:
    ensure_ted_notice_schema(conn)
    inserted = 0
    example_rows = 0
    for record in records:
        publication_number = str(record.get("publication_number") or "").strip()
        if not publication_number:
            continue
        conn.execute(
            """
            INSERT INTO ted_notices (
              publication_number, main_cpv_code, all_cpv_codes_json, notice_title, title_proc,
              description_proc, description_lot, buyer_name, official_languages_json,
              language_count, combined_text, raw_notice_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(publication_number) DO UPDATE SET
              main_cpv_code=excluded.main_cpv_code,
              all_cpv_codes_json=excluded.all_cpv_codes_json,
              notice_title=excluded.notice_title,
              title_proc=excluded.title_proc,
              description_proc=excluded.description_proc,
              description_lot=excluded.description_lot,
              buyer_name=excluded.buyer_name,
              official_languages_json=excluded.official_languages_json,
              language_count=excluded.language_count,
              combined_text=excluded.combined_text,
              raw_notice_json=excluded.raw_notice_json
            """,
            (
                publication_number,
                str(record.get("main_cpv_code") or "").strip(),
                json.dumps(list(record.get("all_cpv_codes", [])), ensure_ascii=False),
                str(record.get("notice_title") or ""),
                str(record.get("title_proc") or ""),
                str(record.get("description_proc") or ""),
                str(record.get("description_lot") or ""),
                str(record.get("buyer_name") or ""),
                json.dumps(list(record.get("official_languages", [])), ensure_ascii=False),
                int(record.get("language_count") or 0),
                str(record.get("combined_text") or ""),
                json.dumps(record.get("raw_notice") or {}, ensure_ascii=False),
            ),
        )
        inserted += 1
        for example in _notice_examples_from_record(record):
            conn.execute(
                """
                INSERT OR IGNORE INTO cpv_notice_examples (
                  publication_number, cpv_code, language, source_field, example_text
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    example["publication_number"],
                    example["cpv_code"],
                    example["language"],
                    example["source_field"],
                    example["example_text"],
                ),
            )
            example_rows += 1
    conn.commit()
    return {"notice_rows_upserted": inserted, "example_rows_seen": example_rows}


def count_rows(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS n FROM {table_name}").fetchone()
    return int(row["n"]) if row is not None else 0


def top_languages(conn: sqlite3.Connection, *, limit: int = 25) -> List[Dict[str, object]]:
    rows = conn.execute(
        """
        SELECT language, COUNT(*) AS n
        FROM cpv_notice_examples
        GROUP BY language
        ORDER BY n DESC, language ASC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    return [{"language": str(row["language"]), "count": int(row["n"])} for row in rows]


def load_notice_examples_by_cpv(path: str, *, max_examples_per_cpv: int = 8) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return {}
    conn = open_ted_notice_db(path)
    try:
        rows = conn.execute(
            """
            SELECT cpv_code, example_text
            FROM cpv_notice_examples
            ORDER BY cpv_code ASC, publication_number ASC
            """
        ).fetchall()
    finally:
        conn.close()
    examples_by_cpv: Dict[str, List[str]] = {}
    for row in rows:
        cpv_code = str(row["cpv_code"]).strip()
        text = str(row["example_text"]).strip()
        if not cpv_code or not text:
            continue
        bucket = examples_by_cpv.setdefault(cpv_code, [])
        if text in bucket or len(bucket) >= max_examples_per_cpv:
            continue
        bucket.append(text)
    return examples_by_cpv


def upsert_cpv_profiles(
    conn: sqlite3.Connection,
    profiles: Sequence[Dict[str, object]],
    *,
    source_fingerprint: str | None = None,
) -> Dict[str, int]:
    ensure_ted_notice_schema(conn)
    inserted = 0
    for profile in profiles:
        code = str(profile.get("code") or "").strip()
        if not code:
            continue
        conn.execute(
            """
            INSERT INTO cpv_profiles (
              code, label, description_en, parent_code, parent_label, procurement_type,
              keywords_en, generated_synonyms_en, use_when_text, do_not_use_when_text,
              common_tender_phrases, children_codes_json, children_labels_json,
              sibling_codes_json, sibling_labels_json, examples_json, notice_examples_json,
              text, search_text_en, search_text_multilingual, source_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
              label=excluded.label,
              description_en=excluded.description_en,
              parent_code=excluded.parent_code,
              parent_label=excluded.parent_label,
              procurement_type=excluded.procurement_type,
              keywords_en=excluded.keywords_en,
              generated_synonyms_en=excluded.generated_synonyms_en,
              use_when_text=excluded.use_when_text,
              do_not_use_when_text=excluded.do_not_use_when_text,
              common_tender_phrases=excluded.common_tender_phrases,
              children_codes_json=excluded.children_codes_json,
              children_labels_json=excluded.children_labels_json,
              sibling_codes_json=excluded.sibling_codes_json,
              sibling_labels_json=excluded.sibling_labels_json,
              examples_json=excluded.examples_json,
              notice_examples_json=excluded.notice_examples_json,
              text=excluded.text,
              search_text_en=excluded.search_text_en,
              search_text_multilingual=excluded.search_text_multilingual,
              source_fingerprint=excluded.source_fingerprint
            """,
            (
                code,
                str(profile.get("label") or ""),
                str(profile.get("description_en") or ""),
                str(profile.get("parent_code") or ""),
                str(profile.get("parent_label") or ""),
                str(profile.get("procurement_type") or ""),
                str(profile.get("keywords_en") or ""),
                str(profile.get("generated_synonyms_en") or ""),
                str(profile.get("use_when_text") or ""),
                str(profile.get("do_not_use_when_text") or ""),
                str(profile.get("common_tender_phrases") or ""),
                json.dumps(list(profile.get("children_codes") or []), ensure_ascii=False),
                json.dumps(list(profile.get("children_labels") or []), ensure_ascii=False),
                json.dumps(list(profile.get("sibling_codes") or []), ensure_ascii=False),
                json.dumps(list(profile.get("sibling_labels") or []), ensure_ascii=False),
                json.dumps(list(profile.get("examples") or []), ensure_ascii=False),
                json.dumps(list(profile.get("notice_examples") or []), ensure_ascii=False),
                str(profile.get("text") or ""),
                str(profile.get("search_text_en") or ""),
                str(profile.get("search_text_multilingual") or ""),
                str(source_fingerprint or profile.get("source_fingerprint") or ""),
            ),
        )
        inserted += 1
    conn.commit()
    return {"cpv_profiles_upserted": inserted}


def load_cpv_profiles(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []
    conn = open_ted_notice_db(path)
    try:
        ensure_ted_notice_schema(conn)
        rows = conn.execute(
            """
            SELECT code, label, description_en, parent_code, parent_label, procurement_type,
                   keywords_en, generated_synonyms_en, use_when_text, do_not_use_when_text,
                   common_tender_phrases, children_codes_json, children_labels_json,
                   sibling_codes_json, sibling_labels_json, examples_json, notice_examples_json,
                   text, search_text_en, search_text_multilingual, source_fingerprint
            FROM cpv_profiles
            ORDER BY code ASC
            """
        ).fetchall()
    finally:
        conn.close()
    profiles: List[Dict[str, object]] = []
    for row in rows:
        profiles.append(
            {
                "code": str(row["code"] or ""),
                "label": str(row["label"] or ""),
                "description_en": str(row["description_en"] or ""),
                "parent_code": str(row["parent_code"] or ""),
                "parent_label": str(row["parent_label"] or ""),
                "procurement_type": str(row["procurement_type"] or ""),
                "keywords_en": str(row["keywords_en"] or ""),
                "generated_synonyms_en": str(row["generated_synonyms_en"] or ""),
                "use_when_text": str(row["use_when_text"] or ""),
                "do_not_use_when_text": str(row["do_not_use_when_text"] or ""),
                "common_tender_phrases": str(row["common_tender_phrases"] or ""),
                "children_codes": json.loads(str(row["children_codes_json"] or "[]")),
                "children_labels": json.loads(str(row["children_labels_json"] or "[]")),
                "sibling_codes": json.loads(str(row["sibling_codes_json"] or "[]")),
                "sibling_labels": json.loads(str(row["sibling_labels_json"] or "[]")),
                "examples": json.loads(str(row["examples_json"] or "[]")),
                "notice_examples": json.loads(str(row["notice_examples_json"] or "[]")),
                "text": str(row["text"] or ""),
                "search_text_en": str(row["search_text_en"] or ""),
                "search_text_multilingual": str(row["search_text_multilingual"] or ""),
                "source_fingerprint": str(row["source_fingerprint"] or ""),
            }
        )
    return profiles
