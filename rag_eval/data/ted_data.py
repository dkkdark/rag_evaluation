from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence


TED_SEARCH_API_URL = "https://api.ted.europa.eu/v3/notices/search"
TED_DEFAULT_FIELDS = [
    "publication-number",
    "notice-title",
    "main-classification-proc",
    "title-proc",
    "description-proc",
    "description-lot",
]
TED_DEFAULT_QUERY = "main-classification-proc=* AND notice-title=*"
PREFERRED_LANGUAGES = ["eng", "deu", "fra", "ita", "spa"]
NOISE_PATTERNS = [
    "test",
    "please disregard",
    "lorem ipsum",
    "dolor sit amet",
    "galisum",
    "rem eveniet",
    "qui ipsa totam",
    "optio necessitatibus",
    "ipsam officia",
    "qui sequi excepturi",
]


def _post_json(url: str, payload: Dict, *, timeout: int = 60, retries: int = 5) -> Dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 10))
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("TED API request failed without an error.")


def _pick_language_value(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = _pick_language_value(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for language in PREFERRED_LANGUAGES:
            text = str(value.get(language, "")).strip()
            if text:
                return text
        for item in value.values():
            text = _pick_language_value(item)
            if text:
                return text
    return ""


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    return text


def _looks_noisy(text: str) -> bool:
    lowered = text.casefold()
    return any(pattern in lowered for pattern in NOISE_PATTERNS)


def _extract_cpv_label_from_notice_title(notice_title: str) -> str:
    if " – " in notice_title:
        parts = [part.strip() for part in notice_title.split(" – ") if part.strip()]
        if len(parts) >= 3:
            return parts[1]
    return notice_title.strip()


def fetch_ted_notice_batch(
    *,
    query: str = TED_DEFAULT_QUERY,
    fields: Sequence[str] = TED_DEFAULT_FIELDS,
    limit: int = 100,
    page: int = 1,
) -> List[Dict]:
    payload = {
        "query": query,
        "fields": list(fields),
        "limit": limit,
        "page": page,
        "paginationMode": "PAGE_NUMBER",
    }
    response = _post_json(TED_SEARCH_API_URL, payload)
    return list(response.get("notices", []))


def fetch_ted_notice_pages(
    *,
    query: str = TED_DEFAULT_QUERY,
    fields: Sequence[str] = TED_DEFAULT_FIELDS,
    limit: int = 100,
    start_page: int = 1,
    pages: int = 1,
    pause_seconds: float = 0.35,
) -> List[Dict]:
    all_notices: List[Dict] = []
    for page in range(start_page, start_page + pages):
        if page > start_page and pause_seconds > 0:
            time.sleep(pause_seconds)
        all_notices.extend(fetch_ted_notice_batch(query=query, fields=fields, limit=limit, page=page))
    return all_notices


def _iter_notice_text_candidates(notice: Dict) -> Iterable[str]:
    for field in ["title-proc", "description-proc", "description-lot"]:
        raw_value = notice.get(field)
        text = _pick_language_value(raw_value)
        cleaned = _clean_text(text)
        if cleaned:
            yield cleaned
    notice_title = _clean_text(_pick_language_value(notice.get("notice-title")))
    if notice_title:
        yield notice_title


def _notice_split_bucket(notice: Dict, train_ratio: float) -> str:
    notice_id = str(notice.get("publication-number", ""))
    digest = hashlib.sha256(notice_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if bucket < train_ratio else "test"


def _extract_notice_example(notice: Dict) -> Dict | None:
    cpv_codes = notice.get("main-classification-proc") or []
    if isinstance(cpv_codes, str):
        cpv_codes = [cpv_codes]
    if not cpv_codes:
        return None

    cpv_code = str(cpv_codes[0]).strip()
    notice_title = _clean_text(_pick_language_value(notice.get("notice-title")))
    if _looks_noisy(notice_title):
        return None
    cpv_label = _extract_cpv_label_from_notice_title(notice_title) if notice_title else cpv_code

    candidates = []
    for text in _iter_notice_text_candidates(notice):
        if _looks_noisy(text):
            continue
        if len(text) < 20:
            continue
        candidates.append(text)

    if not candidates:
        return None

    query_text = candidates[0]
    if _looks_noisy(query_text):
        return None

    return {
        "id": str(notice.get("publication-number", "")).strip(),
        "cpv_code": cpv_code,
        "cpv_label": cpv_label,
        "notice_title": notice_title,
        "examples": candidates,
        "query_text": query_text,
    }


def transform_ted_notices_to_cpv_split_dataset(
    notices: Sequence[Dict],
    *,
    train_ratio: float = 0.8,
    min_examples_per_cpv: int = 1,
    max_examples_per_cpv: int = 8,
    max_queries: int = 100,
) -> tuple[list[Dict], list[Dict], Dict[str, object]]:
    examples_by_cpv: Dict[str, List[str]] = defaultdict(list)
    cpv_labels: Dict[str, str] = {}
    train_notice_count = 0
    test_notice_count = 0
    test_candidates: List[Dict] = []

    for notice in notices:
        extracted = _extract_notice_example(notice)
        if extracted is None:
            continue

        split = _notice_split_bucket(notice, train_ratio)
        if split == "train":
            train_notice_count += 1
            cpv_code = extracted["cpv_code"]
            cpv_labels.setdefault(cpv_code, extracted["cpv_label"])
            for text in extracted["examples"][:max_examples_per_cpv]:
                if text not in examples_by_cpv[cpv_code]:
                    examples_by_cpv[cpv_code].append(text)
        else:
            test_notice_count += 1
            test_candidates.append(extracted)

    catalog: List[Dict] = []
    for cpv_code, examples in sorted(examples_by_cpv.items()):
        if len(examples) < min_examples_per_cpv:
            continue
        label = cpv_labels.get(cpv_code, cpv_code)
        description = examples[0]
        catalog.append(
            {
                "code": cpv_code,
                "label": label,
                "description": description,
                "parent_code": "",
                "examples": " || ".join(examples[:max_examples_per_cpv]),
            }
        )

    valid_codes = {row["code"] for row in catalog}
    queries: List[Dict] = []
    seen_query_keys = set()
    for row in test_candidates:
        if row["cpv_code"] not in valid_codes:
            continue
        query_key = (row["query_text"], row["cpv_code"])
        if query_key in seen_query_keys or len(queries) >= max_queries:
            continue
        seen_query_keys.add(query_key)
        queries.append(
            {
                "id": row["id"] or f"ted-test-{len(queries) + 1}",
                "query": row["query_text"],
                "gold_cpv_code": row["cpv_code"],
                "notes": row["notice_title"],
            }
        )

    split_summary = {
        "train_notice_count": train_notice_count,
        "test_notice_count": test_notice_count,
        "train_ratio": train_ratio,
        "n_catalog_entries": len(catalog),
        "n_test_queries": len(queries),
    }
    return catalog, queries, split_summary


def read_cpv_catalog_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(
                {
                    "code": str(row.get("code", "")).strip(),
                    "label": str(row.get("label", "")).strip(),
                    "description": str(row.get("description", "")).strip(),
                    "parent_code": str(row.get("parent_code", "")).strip(),
                    "examples": str(row.get("examples", "")).strip(),
                }
            )
        return rows


def _split_examples_field(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(" || ") if item.strip()]


def collect_ted_examples_by_cpv(
    notices: Sequence[Dict],
    *,
    max_examples_per_cpv: int = 8,
    valid_codes: set[str] | None = None,
) -> Dict[str, List[str]]:
    examples_by_cpv: Dict[str, List[str]] = defaultdict(list)
    for notice in notices:
        extracted = _extract_notice_example(notice)
        if extracted is None:
            continue
        cpv_code = extracted["cpv_code"]
        if valid_codes is not None and cpv_code not in valid_codes:
            continue
        bucket = examples_by_cpv[cpv_code]
        for text in extracted["examples"]:
            if text in bucket or len(bucket) >= max_examples_per_cpv:
                continue
            bucket.append(text)
    return dict(examples_by_cpv)


def enrich_cpv_catalog_with_ted_examples(
    base_rows: Sequence[Dict[str, str]],
    examples_by_cpv: Dict[str, Sequence[str]],
    *,
    max_examples_per_cpv: int = 8,
) -> List[Dict[str, str]]:
    enriched: List[Dict[str, str]] = []
    for row in base_rows:
        merged = _split_examples_field(str(row.get("examples", "")))
        for text in examples_by_cpv.get(row["code"], []):
            if text in merged or len(merged) >= max_examples_per_cpv:
                continue
            merged.append(text)
        enriched.append(
            {
                "code": row["code"],
                "label": row["label"],
                "description": row["description"],
                "parent_code": row.get("parent_code", ""),
                "examples": " || ".join(merged),
            }
        )
    return enriched


def write_cpv_catalog_csv(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["code", "label", "description", "parent_code", "examples"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_queries_json(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"queries": list(rows)}, f, ensure_ascii=False, indent=2)
