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


TED_SEARCH_API_URL = os.environ.get("TED_SEARCH_API_URL", "https://api.ted.europa.eu/v3/notices/search")
TED_DEFAULT_CORPUS_EXPORT_PATH = "data/teddata_corpus_export.csv"
TED_DEFAULT_FIELDS = [
    "publication-number",
    "notice-title",
    "main-classification-proc",
    "title-proc",
    "description-proc",
    "description-lot",
]
TED_MULTILINGUAL_FIELDS = [
    *TED_DEFAULT_FIELDS,
    "additional-classification-proc",
    "buyer-name",
    "official-language",
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


def _clean_corpus_export_cpv(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits[:8] if len(digits) >= 8 else ""


def load_ted_corpus_export_notices(path: str) -> List[Dict]:
    notices: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            publication_number = str(row.get("PUB_NUMBER", "")).strip()
            main_cpv = _clean_corpus_export_cpv(row.get("MAIN_CPV", ""))
            title = _clean_text(str(row.get("TITLE", "")).strip())
            text = _clean_text(str(row.get("TEXT", "")).strip())
            buyer = _clean_text(str(row.get("BUYER", "")).strip())
            language = str(row.get("LANGUAGE", "")).strip().casefold()
            contract_type = str(row.get("CONTRACT_TYPE", "")).strip().casefold()
            if not publication_number or not main_cpv or not (title or text):
                continue
            notices.append(
                {
                    "publication-number": publication_number,
                    "notice-title": title,
                    "title-proc": title,
                    "description-proc": text,
                    "description-lot": "",
                    "buyer-name": buyer,
                    "official-language": language,
                    "main-classification-proc": [main_cpv],
                    "additional-classification-proc": [],
                    "contract-type": contract_type,
                    "source-id": str(row.get("ID", "")).strip(),
                    "source": "teddata_corpus_export",
                }
            )
    return notices


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
            raw_error = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(
                f"TED API HTTP {exc.code} for payload {json.dumps(payload, ensure_ascii=False)}; response body: {raw_error}"
            )
            if exc.code == 429 and attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise last_error
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


def fetch_ted_notice_iteration(
    *,
    query: str = TED_DEFAULT_QUERY,
    fields: Sequence[str] = TED_MULTILINGUAL_FIELDS,
    limit: int = 250,
    max_notices: int = 20000,
    pause_seconds: float = 0.35,
) -> List[Dict]:
    notices: List[Dict] = []
    iteration_token: str | None = None
    while len(notices) < max_notices:
        payload = {
            "query": query,
            "fields": list(fields),
            "limit": min(int(limit), 250),
            "paginationMode": "ITERATION",
        }
        if iteration_token:
            payload["iterationNextToken"] = iteration_token
        response = _post_json(TED_SEARCH_API_URL, payload)
        batch = list(response.get("notices", []))
        if not batch:
            break
        notices.extend(batch)
        iteration_token = str(response.get("iterationNextToken") or "").strip() or None
        if not iteration_token:
            break
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return notices[:max_notices]


def _collect_language_values(value: object) -> List[str]:
    values: List[str] = []
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            values.append(cleaned)
    elif isinstance(value, list):
        for item in value:
            values.extend(_collect_language_values(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            key_clean = str(key).strip()
            if len(key_clean) in {2, 3} and key_clean.isalpha():
                values.append(key_clean)
            values.extend(_collect_language_values(item))
    deduped: List[str] = []
    seen = set()
    for item in values:
        norm = item.casefold()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(item)
    return deduped


def extract_notice_languages(notice: Dict) -> List[str]:
    languages: List[str] = []
    for key in [
        "official-language",
        "additional-notice-language",
        "notice-language",
        "languages",
        "lang",
    ]:
        languages.extend(_collect_language_values(notice.get(key)))
    deduped: List[str] = []
    seen = set()
    for item in languages:
        norm = item.casefold()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(item)
    return deduped


def normalize_ted_notice_record(notice: Dict) -> Dict[str, object]:
    cpv_codes = notice.get("main-classification-proc") or []
    if isinstance(cpv_codes, str):
        cpv_codes = [cpv_codes]
    additional_codes = notice.get("additional-classification-proc") or []
    if isinstance(additional_codes, str):
        additional_codes = [additional_codes]
    notice_title = _clean_text(_pick_language_value(notice.get("notice-title")))
    title_proc = _clean_text(_pick_language_value(notice.get("title-proc")))
    description_proc = _clean_text(_pick_language_value(notice.get("description-proc")))
    description_lot = _clean_text(_pick_language_value(notice.get("description-lot")))
    combined_text = "\n".join(
        part for part in [notice_title, title_proc, description_proc, description_lot] if part.strip()
    )
    languages = extract_notice_languages(notice)
    return {
        "publication_number": str(notice.get("publication-number", "")).strip(),
        "main_cpv_code": str(cpv_codes[0]).strip() if cpv_codes else "",
        "all_cpv_codes": [str(code).strip() for code in [*cpv_codes, *additional_codes] if str(code).strip()],
        "notice_title": notice_title,
        "title_proc": title_proc,
        "description_proc": description_proc,
        "description_lot": description_lot,
        "buyer_name": _clean_text(_pick_language_value(notice.get("buyer-name"))),
        "official_languages": languages,
        "language_count": len(languages),
        "combined_text": combined_text,
        "raw_notice": notice,
    }


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


def build_cpv_catalog_rows_from_notices(
    notices: Sequence[Dict],
    *,
    min_examples_per_cpv: int = 1,
    max_examples_per_cpv: int = 12,
) -> List[Dict[str, str]]:
    examples_by_cpv: Dict[str, List[str]] = defaultdict(list)
    cpv_labels: Dict[str, str] = {}

    for notice in notices:
        extracted = _extract_notice_example(notice)
        if extracted is None:
            continue
        cpv_code = extracted["cpv_code"]
        cpv_labels.setdefault(cpv_code, extracted["cpv_label"])
        for text in extracted["examples"][:max_examples_per_cpv]:
            if text not in examples_by_cpv[cpv_code]:
                examples_by_cpv[cpv_code].append(text)

    catalog: List[Dict[str, str]] = []
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
    return catalog


def build_cpv_catalog_rows_from_corpus_export(
    path: str,
    *,
    min_examples_per_cpv: int = 1,
    max_examples_per_cpv: int = 12,
) -> List[Dict[str, str]]:
    notices = load_ted_corpus_export_notices(path)
    return build_cpv_catalog_rows_from_notices(
        notices,
        min_examples_per_cpv=min_examples_per_cpv,
        max_examples_per_cpv=max_examples_per_cpv,
    )


def transform_ted_notices_to_cpv_split_dataset(
    notices: Sequence[Dict],
    *,
    train_ratio: float = 0.8,
    min_examples_per_cpv: int = 1,
    max_examples_per_cpv: int = 12,
    max_queries: int = 100,
) -> tuple[list[Dict], list[Dict], Dict[str, object]]:
    train_notice_count = 0
    test_notice_count = 0
    test_candidates: List[Dict] = []
    train_notices: List[Dict] = []

    for notice in notices:
        extracted = _extract_notice_example(notice)
        if extracted is None:
            continue

        split = _notice_split_bucket(notice, train_ratio)
        if split == "train":
            train_notice_count += 1
            train_notices.append(notice)
        else:
            test_notice_count += 1
            test_candidates.append(extracted)

    catalog = build_cpv_catalog_rows_from_notices(
        train_notices,
        min_examples_per_cpv=min_examples_per_cpv,
        max_examples_per_cpv=max_examples_per_cpv,
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
