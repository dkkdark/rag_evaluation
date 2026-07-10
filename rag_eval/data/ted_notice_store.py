from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Dict, List, Sequence


_EXAMPLE_SOURCE_WEIGHTS = {
    "title_plus_description_lead": 1.0,
    "title_proc_clean": 0.97,
    "description_proc_lead": 0.94,
    "description_lot_lead": 0.9,
    "title_proc": 0.88,
    "notice_title": 0.82,
    "description_proc": 0.72,
    "description_lot": 0.68,
}

_EXAMPLE_GENERIC_PATTERNS = [
    "framework agreement",
    "contract notice",
    "procurement of",
    "supply and delivery of",
    "invitation to tender",
    "call for tenders",
    "open procedure",
    "negotiated procedure",
]


def _clean_example_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\n", " ")).strip()


def _normalize_example_key(text: str) -> str:
    cleaned = _clean_example_text(text).casefold()
    cleaned = re.sub(r"\b[a-z]{1,4}\d{2,}[/_-]?\d+\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{4,}\b", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _strip_notice_prefix(text: str) -> str:
    cleaned = _clean_example_text(text)
    cleaned = re.sub(r"^[A-Z]{1,6}\d{0,4}(?:[/_-]\d{2,6})+(?:\s*[-:]\s*|\s+)", "", cleaned)
    cleaned = re.sub(r"^\d{4,}(?:[/_-]\d{2,6})?(?:\s*[-:]\s*|\s+)", "", cleaned)
    return cleaned.strip(" -:")


def _sentences(text: str) -> List[str]:
    cleaned = _clean_example_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _trim_example_length(text: str, *, max_chars: int = 320) -> str:
    cleaned = _clean_example_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    trimmed = cleaned[:max_chars]
    last_space = trimmed.rfind(" ")
    if last_space >= 120:
        trimmed = trimmed[:last_space]
    return trimmed.rstrip(" ,;:-") + "..."


def _collapse_repeated_lead(text: str) -> str:
    cleaned = _clean_example_text(text)
    match = re.match(r"^(.{8,160}?)(?:[.!?])\s+\1(?:[.!?])(\s+.*)?$", cleaned, flags=re.IGNORECASE)
    if match:
        tail = match.group(2) or ""
        cleaned = f"{match.group(1)}.{tail}".strip()
    return _clean_example_text(cleaned)


def _looks_low_signal_example(text: str) -> bool:
    cleaned = _clean_example_text(text)
    if len(cleaned) < 10:
        return True
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", cleaned)
    return len(words) < 2


def _build_notice_example_variants(record: Dict[str, object], field: str, text: str) -> List[tuple[str, str]]:
    cleaned = _clean_example_text(text)
    if not cleaned or _looks_low_signal_example(cleaned):
        return []
    variants: List[tuple[str, str]] = []
    if field in {"notice_title", "title_proc"}:
        stripped = _strip_notice_prefix(cleaned)
        variants.append((field, _collapse_repeated_lead(_trim_example_length(cleaned, max_chars=220))))
        if stripped and stripped != cleaned and not _looks_low_signal_example(stripped):
            variants.append(("title_proc_clean", _collapse_repeated_lead(_trim_example_length(stripped, max_chars=220))))
        lead_description = _clean_example_text(str(record.get("description_proc") or ""))
        lead_sentences = _sentences(lead_description)
        if lead_sentences:
            title_base = _strip_notice_prefix(cleaned) or cleaned
            lead = lead_sentences[0]
            if _normalize_example_key(lead).startswith(_normalize_example_key(title_base)):
                combined = lead
            else:
                combined = f"{title_base}. {lead}"
            variants.append(("title_plus_description_lead", _collapse_repeated_lead(_trim_example_length(combined, max_chars=260))))
    elif field in {"description_proc", "description_lot"}:
        lead_sentences = _sentences(cleaned)[:2]
        if lead_sentences:
            lead = " ".join(lead_sentences)
            source_name = "description_proc_lead" if field == "description_proc" else "description_lot_lead"
            variants.append((source_name, _collapse_repeated_lead(_trim_example_length(lead, max_chars=260))))
        variants.append((field, _collapse_repeated_lead(_trim_example_length(cleaned, max_chars=320))))
    else:
        variants.append((field, _trim_example_length(cleaned, max_chars=220)))
    deduped: List[tuple[str, str]] = []
    seen = set()
    for source_field, variant_text in variants:
        key = _normalize_example_key(variant_text)
        if not key or key in seen or _looks_low_signal_example(variant_text):
            continue
        seen.add(key)
        deduped.append((source_field, variant_text))
    return deduped


def _score_example_row(source_field: str, text: str) -> float:
    cleaned = _clean_example_text(text)
    source_weight = _EXAMPLE_SOURCE_WEIGHTS.get(source_field, 0.7)
    token_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", cleaned))
    token_bonus = min(token_count, 18) * 0.015
    ideal_length_penalty = abs(min(len(cleaned), 220) - 140) / 700.0
    generic_penalty = 0.0
    lowered = cleaned.casefold()
    for pattern in _EXAMPLE_GENERIC_PATTERNS:
        if pattern in lowered:
            generic_penalty += 0.035
    if re.match(r"^[A-Z0-9/_ -]{8,}$", cleaned):
        generic_penalty += 0.08
    return source_weight + token_bonus - ideal_length_penalty - generic_penalty


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

        DROP TABLE IF EXISTS cpv_profiles_base;
        DROP TABLE IF EXISTS cpv_profiles_examples;

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
    seen_rows = set()
    for field in ["notice_title", "title_proc", "description_proc", "description_lot"]:
        text = str(record.get(field) or "")
        variants = _build_notice_example_variants(record, field, text)
        if not variants:
            continue
        for cpv_code in cpv_codes:
            for language in languages:
                for source_field, example_text in variants:
                    row_key = (
                        publication_number,
                        cpv_code,
                        language,
                        _normalize_example_key(example_text),
                    )
                    if row_key in seen_rows:
                        continue
                    seen_rows.add(row_key)
                    examples.append(
                        {
                            "publication_number": publication_number,
                            "cpv_code": cpv_code,
                            "language": language,
                            "source_field": source_field,
                            "example_text": example_text,
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
        conn.execute(
            "DELETE FROM cpv_notice_examples WHERE publication_number = ?",
            (publication_number,),
        )
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


def load_notice_examples_by_cpv(path: str, *, max_examples_per_cpv: int = 12) -> Dict[str, List[str]]:
    detailed = load_notice_examples_by_cpv_detailed(path, max_examples_per_cpv=max_examples_per_cpv)
    return {
        cpv_code: [str(item.get("example_text") or "").strip() for item in items if str(item.get("example_text") or "").strip()]
        for cpv_code, items in detailed.items()
    }


def load_notice_examples_by_cpv_detailed(
    path: str,
    *,
    max_examples_per_cpv: int = 12,
) -> Dict[str, List[Dict[str, str]]]:
    if not os.path.exists(path):
        return {}
    conn = open_ted_notice_db(path)
    try:
        rows = conn.execute(
            """
            SELECT cpv_code, publication_number, source_field, example_text
            FROM cpv_notice_examples
            ORDER BY cpv_code ASC, publication_number ASC, source_field ASC
            """
        ).fetchall()
    finally:
        conn.close()
    examples_by_cpv: Dict[str, List[Dict[str, str]]] = {}
    seen_keys_by_cpv: Dict[str, set[str]] = {}
    rows_by_cpv: Dict[str, List[sqlite3.Row]] = {}
    for row in rows:
        cpv_code = str(row["cpv_code"]).strip()
        text = _clean_example_text(str(row["example_text"] or ""))
        if not cpv_code or not text or _looks_low_signal_example(text):
            continue
        rows_by_cpv.setdefault(cpv_code, []).append(row)
    for cpv_code, cpv_rows in rows_by_cpv.items():
        ranked_rows = sorted(
            cpv_rows,
            key=lambda row: (
                -_score_example_row(str(row["source_field"] or ""), str(row["example_text"] or "")),
                len(_clean_example_text(str(row["example_text"] or ""))),
                str(row["publication_number"] or ""),
            ),
        )
        bucket = examples_by_cpv.setdefault(cpv_code, [])
        seen_keys = seen_keys_by_cpv.setdefault(cpv_code, set())
        per_publication_counts: Dict[str, int] = {}
        for row in ranked_rows:
            if len(bucket) >= max_examples_per_cpv:
                break
            text = _clean_example_text(str(row["example_text"] or ""))
            normalized_key = _normalize_example_key(text)
            if not normalized_key or normalized_key in seen_keys:
                continue
            publication_number = str(row["publication_number"] or "").strip()
            if publication_number and per_publication_counts.get(publication_number, 0) >= 3:
                continue
            if publication_number:
                per_publication_counts[publication_number] = per_publication_counts.get(publication_number, 0) + 1
            seen_keys.add(normalized_key)
            bucket.append(
                {
                    "publication_number": publication_number,
                    "source_field": str(row["source_field"] or ""),
                    "example_text": text,
                }
            )
        if not bucket:
            continue
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
