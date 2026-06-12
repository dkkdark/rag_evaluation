from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from rag_eval.data.ted_notice_store import (
    load_cpv_profiles,
    load_notice_examples_by_cpv,
    open_ted_notice_db,
    upsert_cpv_profiles,
)
from rag_eval.data.ted_data import build_cpv_catalog_rows_from_corpus_export


@dataclass
class CPVRecord:
    code: str
    label: str
    description: str
    parent_code: str
    examples: List[str]
    multilingual_labels: List[str]


@dataclass
class MaterialQuery:
    id: str
    query: str
    gold_cpv_code: str
    notes: str


_ENGLISH_STOPWORDS = {
    "and", "or", "the", "of", "for", "to", "in", "with", "on", "by", "from", "other",
    "services", "service", "work", "works", "supply", "related", "parts",
}

_MULTILINGUAL_ALIAS_SEEDS = [
    ("heating", ["chauffage", "heizung", "calefaccion", "riscaldamento", "otoplenie"]),
    ("ventilation", ["ventilation", "beluftung", "ventilacion", "ventilazione", "ventilyatsiya"]),
    ("maintenance", ["maintenance", "wartung", "mantenimiento", "manutenzione", "obsluzhivanie"]),
    ("repair", ["repair", "reparacion", "reparation", "instandhaltung", "remont"]),
    ("installation", ["installation", "instalacion", "installazione", "ustanovka"]),
    ("software", ["software", "logiciel", "softwareentwicklung", "programmnoe obespechenie"]),
    ("network", ["lan", "network", "reseau", "red", "set"]),
    ("camera", ["camera", "camara", "kamera", "telecamera"]),
    ("security", ["security", "sicurezza", "sicherheits", "ochrona"]),
    ("advertising", ["advertising", "publicity", "publicitari", "pubblicitari"]),
    ("parking", ["parking", "pay and display", "cashless parking"]),
    ("janitorial", ["cleaning", "limpieza", "nettoyage", "reinigung", "celaduria"]),
    ("procurement", ["suministro", "fourniture", "lieferung", "supply", "delivery"]),
]

_PROCUREMENT_TYPE_PATTERNS = {
    "maintenance_repair_service": ["repair", "maintenance", "servicing", "upkeep", "wartung", "mantenimiento", "manutenzione"],
    "installation_work": ["installation", "assembly", "montage", "erection", "commissioning"],
    "consultancy_service": ["consultancy", "consulting", "advisory", "evaluation", "feasibility"],
    "software_it_service": ["software", "it service", "hosting", "cloud", "development", "digital", "network"],
    "medical_equipment_goods": ["prostheses", "implant", "camera", "medical", "device", "equipment", "machine", "tool"],
    "database_information_service": ["database", "information service", "electronic journals", "digital library"],
    "goods_supply": ["supply", "delivery", "equipment", "machine", "device", "materials", "furnishing"],
    "works_construction": ["construction", "building", "road works", "engineering works", "infrastructure"],
}


def _extract_keywords_en(*parts: str, limit: int = 8) -> str:
    tokens = []
    seen = set()
    for part in parts:
        for token in re.findall(r"[A-Za-z0-9]+", str(part or "").lower()):
            if len(token) <= 2 or token in _ENGLISH_STOPWORDS or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) >= limit:
                return " ".join(tokens)
    return " ".join(tokens)
def _multilingual_aliases(label: str, description: str, keywords_en: str) -> str:
    haystack = " ".join([label, description, keywords_en]).lower()
    aliases: List[str] = []
    seen = set()
    for english_term, variants in _MULTILINGUAL_ALIAS_SEEDS:
        if english_term not in haystack:
            continue
        for variant in variants:
            variant_norm = variant.strip().lower()
            if not variant_norm or variant_norm in seen:
                continue
            seen.add(variant_norm)
            aliases.append(variant.strip())
    return " | ".join(aliases)


def _unique_texts(values: Sequence[str], *, limit: int | None = None) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value or "").strip())
        if not normalized:
            continue
        marker = normalized.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        out.append(normalized)
        if limit is not None and len(out) >= limit:
            break
    return out


def _normalize_catalog_code(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits[:8] if len(digits) >= 8 else ""


def _load_legacy_cpv_rows(rows: Sequence[Dict[str, str]]) -> List[CPVRecord]:
    records: List[CPVRecord] = []
    for row in rows:
        code = _normalize_catalog_code(row.get("code", ""))
        if not code:
            continue
        examples_raw = str(row.get("examples", "")).strip()
        examples = [item.strip() for item in examples_raw.split(" || ") if item.strip()]
        records.append(
            CPVRecord(
                code=code,
                label=str(row.get("label", "")).strip(),
                description=str(row.get("description", "")).strip(),
                parent_code=_normalize_catalog_code(row.get("parent_code", "")),
                examples=examples,
                multilingual_labels=[],
            )
        )
    return records


def _load_multilingual_cpv_rows(rows: Sequence[Dict[str, str]]) -> List[CPVRecord]:
    records: List[CPVRecord] = []
    for row in rows:
        code = _normalize_catalog_code(row.get("CODE", ""))
        if not code:
            continue
        multilingual_labels = _unique_texts(
            [str(value).strip() for key, value in row.items() if key != "CODE" and str(value).strip()]
        )
        english_label = str(row.get("EN", "")).strip()
        label = english_label or (multilingual_labels[0] if multilingual_labels else "")
        if not label:
            continue
        records.append(
            CPVRecord(
                code=code,
                label=label,
                description=english_label or label,
                parent_code="",
                examples=[],
                multilingual_labels=multilingual_labels,
            )
        )
    return records


def _infer_procurement_type(*parts: str) -> str:
    haystack = " ".join(str(part or "").casefold() for part in parts)
    for procurement_type, patterns in _PROCUREMENT_TYPE_PATTERNS.items():
        if any(pattern in haystack for pattern in patterns):
            return procurement_type
    if "services" in haystack or "service" in haystack:
        return "general_service"
    return "goods_supply"


def _use_when_text(record: CPVRecord, *, parent_label: str, procurement_type: str, keywords_en: str) -> str:
    hints = _unique_texts(
        [
            record.label,
            record.description,
            keywords_en.replace(" ", ", "),
            parent_label,
        ]
    )
    if procurement_type.endswith("service"):
        return f"Use when the main contract is a service about {record.label.lower()}, not just equipment supply."
    if procurement_type == "installation_work":
        return f"Use when the primary scope is installation or works related to {record.label.lower()}."
    if procurement_type == "works_construction":
        return f"Use when the procurement is mainly construction or engineering works involving {record.label.lower()}."
    return f"Use when the main procured object is {record.label.lower()} or a close equipment category. Related terms: {', '.join(hints[:4])}."


def _do_not_use_when_text(record: CPVRecord, *, procurement_type: str) -> str:
    label = record.label.lower()
    if procurement_type in {"goods_supply", "medical_equipment_goods"}:
        return f"Do not use when the main contract is repair, maintenance, consultancy, or installation service and {label} is only secondary context."
    if procurement_type == "maintenance_repair_service":
        return f"Do not use when the tender is mainly about buying or delivering new {label} equipment."
    if procurement_type == "installation_work":
        return f"Do not use when the main scope is operating, repairing, or maintaining {label} rather than installing it."
    if procurement_type == "consultancy_service":
        return f"Do not use when the buyer mainly procures goods, software licenses, or installation works instead of advice about {label}."
    return f"Do not use when {label} appears only as a side topic and the procurement is mainly for a different contract type."


def load_cpv_catalog(path: str) -> List[CPVRecord]:
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return []

    columns = set(rows[0].keys())
    if {"code", "label"}.issubset(columns):
        return _load_legacy_cpv_rows(rows)
    if "CODE" in columns and "EN" in columns:
        return _load_multilingual_cpv_rows(rows)
    raise ValueError(
        "CPV catalog must use either legacy columns "
        "['code', 'label', ...] or multilingual columns ['CODE', 'EN', ...]."
    )


def load_queries(path: str) -> List[MaterialQuery]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["queries"] if isinstance(data, dict) and "queries" in data else data
    if not isinstance(items, list):
        raise ValueError("Queries should be a list or {'queries': [...]} object.")

    queries: List[MaterialQuery] = []
    for index, item in enumerate(items):
        if "query" not in item or "gold_cpv_code" not in item:
            raise ValueError(f"Query item at index {index} must include 'query' and 'gold_cpv_code'.")
        queries.append(
            MaterialQuery(
                id=str(item.get("id", f"q{index + 1}")),
                query=str(item["query"]).strip(),
                gold_cpv_code=str(item["gold_cpv_code"]).strip(),
                notes=str(item.get("notes", "")).strip(),
            )
        )
    return queries


DEFAULT_TED_NOTICE_DB_PATH = ".rag_eval_indices/ted_notices.sqlite"


def _catalog_fingerprint(records: Sequence[CPVRecord], *, use_examples: bool) -> str:
    digest = hashlib.sha1()
    digest.update(f"use_examples={int(bool(use_examples))}".encode("utf-8"))
    for record in records:
        digest.update(record.code.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(record.label.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(record.description.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(record.parent_code.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(" || ".join(record.multilingual_labels).encode("utf-8"))
        digest.update(b"\x1f")
        if use_examples:
            digest.update(" || ".join(record.examples).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _build_cpv_profiles(
    records: Sequence[CPVRecord],
    *,
    use_examples: bool,
    ted_notice_db_path: str | None = DEFAULT_TED_NOTICE_DB_PATH,
    ted_notice_examples_limit: int = 8,
) -> List[Dict[str, object]]:
    label_by_code = {record.code: record.label for record in records}
    children_by_parent: Dict[str, List[CPVRecord]] = {}
    for record in records:
        if record.parent_code:
            children_by_parent.setdefault(record.parent_code, []).append(record)
    ted_examples_by_cpv = (
        load_notice_examples_by_cpv(ted_notice_db_path, max_examples_per_cpv=ted_notice_examples_limit)
        if ted_notice_db_path and os.path.exists(ted_notice_db_path)
        else {}
    )
    profiles: List[Dict[str, object]] = []
    for record in records:
        parent_label = label_by_code.get(record.parent_code, "")
        children = children_by_parent.get(record.code, [])
        siblings = [child for child in children_by_parent.get(record.parent_code, []) if child.code != record.code]
        notice_examples = (
            ted_examples_by_cpv.get(record.code, [])
            if use_examples
            else []
        )
        merged_examples = _unique_texts(
            list(record.examples) + notice_examples,
            limit=ted_notice_examples_limit,
        ) if use_examples else []
        keywords_en = _extract_keywords_en(
            record.label,
            record.description,
            " ".join(record.multilingual_labels[:8]),
            " ".join(merged_examples[:3]),
            " ".join(child.label for child in children[:3]),
        )
        multilingual_aliases = _multilingual_aliases(record.label, record.description, keywords_en)
        procurement_type = _infer_procurement_type(record.label, record.description, keywords_en, parent_label)
        use_when = _use_when_text(record, parent_label=parent_label, procurement_type=procurement_type, keywords_en=keywords_en)
        do_not_use_when = _do_not_use_when_text(record, procurement_type=procurement_type)
        common_phrases = _unique_texts(merged_examples[:5], limit=5)
        children_labels = [child.label for child in children[:8]]
        children_codes = [child.code for child in children[:8]]
        sibling_labels = [sibling.label for sibling in siblings[:8]]
        sibling_codes = [sibling.code for sibling in siblings[:8]]
        text_parts = [
            f"Code: {record.code}",
            f"Label: {record.label}",
            f"Description: {record.description}" if record.description else "",
            f"Multilingual labels: {' | '.join(record.multilingual_labels[:8])}" if record.multilingual_labels else "",
            f"Parent: {parent_label}" if parent_label else "",
            f"Procurement type: {procurement_type}",
            use_when,
            do_not_use_when,
            f"Common tender phrases: {' | '.join(common_phrases)}" if common_phrases else "",
            f"Related terms: {keywords_en}" if keywords_en else "",
            f"Sibling distinctions: {' | '.join(sibling_labels[:5])}" if sibling_labels else "",
            f"Children: {' | '.join(children_labels[:5])}" if children_labels else "",
        ]
        search_text_en = "\n".join(
            part for part in [
                record.code,
                record.label,
                record.description,
                " | ".join(record.multilingual_labels),
                parent_label,
                procurement_type,
                keywords_en,
                use_when,
                do_not_use_when,
                " ".join(common_phrases),
                " ".join(children_labels[:5]),
                " ".join(sibling_labels[:5]),
            ] if str(part).strip()
        )
        search_text_multilingual = "\n".join(
            part for part in [
                search_text_en,
                " | ".join(record.multilingual_labels),
                multilingual_aliases,
                " | ".join(notice_examples[:4]) if use_examples else "",
            ] if str(part).strip()
        )
        profiles.append(
            {
                "code": record.code,
                "label": record.label,
                "description_en": record.description,
                "parent_code": record.parent_code,
                "parent_label": parent_label,
                "procurement_type": procurement_type,
                "keywords_en": keywords_en,
                "generated_synonyms_en": multilingual_aliases,
                "use_when_text": use_when,
                "do_not_use_when_text": do_not_use_when,
                "common_tender_phrases": " | ".join(common_phrases),
                "children_codes": children_codes,
                "children_labels": children_labels,
                "sibling_codes": sibling_codes,
                "sibling_labels": sibling_labels,
                "examples": merged_examples,
                "notice_examples": notice_examples[:ted_notice_examples_limit] if use_examples else [],
                "text": "\n".join(part for part in text_parts if part.strip()),
                "search_text_en": search_text_en,
                "search_text_multilingual": search_text_multilingual,
            }
        )
    return profiles


def sync_cpv_profiles_to_db(
    records: Sequence[CPVRecord],
    *,
    use_examples: bool,
    ted_notice_db_path: str = DEFAULT_TED_NOTICE_DB_PATH,
    ted_notice_examples_limit: int = 8,
) -> Dict[str, object]:
    profiles = _build_cpv_profiles(
        records,
        use_examples=use_examples,
        ted_notice_db_path=ted_notice_db_path,
        ted_notice_examples_limit=ted_notice_examples_limit,
    )
    fingerprint = _catalog_fingerprint(records, use_examples=use_examples)
    conn = open_ted_notice_db(ted_notice_db_path)
    try:
        result = upsert_cpv_profiles(conn, profiles, source_fingerprint=fingerprint)
    finally:
        conn.close()
    return {"source_fingerprint": fingerprint, **result}


def load_cpv_catalog_from_db(path: str) -> List[CPVRecord]:
    profiles = load_cpv_profiles(path)
    records: List[CPVRecord] = []
    for profile in profiles:
        records.append(
            CPVRecord(
                code=str(profile.get("code") or "").strip(),
                label=str(profile.get("label") or "").strip(),
                description=str(profile.get("description_en") or "").strip(),
                parent_code=str(profile.get("parent_code") or "").strip(),
                examples=[str(item).strip() for item in profile.get("examples", []) if str(item).strip()],
                multilingual_labels=[],
            )
        )
    return records


def load_cpv_catalog_from_ted_corpus_export(
    path: str,
    *,
    max_examples_per_cpv: int = 8,
) -> List[CPVRecord]:
    rows = build_cpv_catalog_rows_from_corpus_export(
        path,
        max_examples_per_cpv=max_examples_per_cpv,
    )
    return _load_legacy_cpv_rows(rows)


def build_cpv_chunks_from_db(path: str) -> List[Dict]:
    profiles = load_cpv_profiles(path)
    chunks: List[Dict] = []
    for index, profile in enumerate(profiles):
        examples = [str(item).strip() for item in profile.get("examples", []) if str(item).strip()]
        chunks.append(
            {
                "chunk_id": str(profile.get("code") or ""),
                "chunk_index": index,
                "doc_id": str(profile.get("code") or ""),
                "doc_path": "cpv_profiles_db",
                "program_id": "cpv",
                "program_name": "CPV",
                "section_id": str(profile.get("code") or ""),
                "title": str(profile.get("label") or ""),
                "text": str(profile.get("text") or ""),
                "chunking_strategy": "cpv_profile_db",
                "source_type": "cpv_profiles_db",
                "cpv_code": str(profile.get("code") or ""),
                "cpv_label": str(profile.get("label") or ""),
                "cpv_parent_code": str(profile.get("parent_code") or ""),
                "cpv_parent_label": str(profile.get("parent_label") or ""),
                "description_en": str(profile.get("description_en") or ""),
                "description_multilingual_aliases": str(profile.get("generated_synonyms_en") or ""),
                "generated_synonyms_en": str(profile.get("generated_synonyms_en") or ""),
                "generated_keywords_en": str(profile.get("keywords_en") or ""),
                "keywords_en": str(profile.get("keywords_en") or ""),
                "procurement_type": str(profile.get("procurement_type") or ""),
                "use_when_text": str(profile.get("use_when_text") or ""),
                "do_not_use_when_text": str(profile.get("do_not_use_when_text") or ""),
                "common_tender_phrases": str(profile.get("common_tender_phrases") or ""),
                "children_labels": " | ".join(str(item) for item in profile.get("children_labels", []) if str(item).strip()),
                "sibling_labels": " | ".join(str(item) for item in profile.get("sibling_labels", []) if str(item).strip()),
                "examples_text": " | ".join(examples),
                "notice_examples_count": len([item for item in profile.get("notice_examples", []) if str(item).strip()]),
                "search_text_en": str(profile.get("search_text_en") or ""),
                "search_text_multilingual": str(profile.get("search_text_multilingual") or ""),
            }
        )
    return chunks


def build_cpv_chunks(
    records: Sequence[CPVRecord],
    *,
    use_examples: bool,
    ted_notice_db_path: str | None = DEFAULT_TED_NOTICE_DB_PATH,
    ted_notice_examples_limit: int = 8,
) -> List[Dict]:
    if ted_notice_db_path:
        sync_cpv_profiles_to_db(
            records,
            use_examples=use_examples,
            ted_notice_db_path=ted_notice_db_path,
            ted_notice_examples_limit=ted_notice_examples_limit,
        )
        db_chunks = build_cpv_chunks_from_db(ted_notice_db_path)
        if db_chunks:
            return db_chunks
    profiles = _build_cpv_profiles(
        records,
        use_examples=use_examples,
        ted_notice_db_path=ted_notice_db_path,
        ted_notice_examples_limit=ted_notice_examples_limit,
    )
    temp_db = ted_notice_db_path if ted_notice_db_path and os.path.exists(ted_notice_db_path) else None
    if temp_db:
        return build_cpv_chunks_from_db(temp_db)
    chunks: List[Dict] = []
    for index, profile in enumerate(profiles):
        chunks.append(
            {
                "chunk_id": str(profile.get("code") or ""),
                "chunk_index": index,
                "doc_id": str(profile.get("code") or ""),
                "doc_path": "cpv_profiles_memory",
                "program_id": "cpv",
                "program_name": "CPV",
                "section_id": str(profile.get("code") or ""),
                "title": str(profile.get("label") or ""),
                "text": str(profile.get("text") or ""),
                "chunking_strategy": "cpv_profile_memory",
                "source_type": "cpv_profile_memory",
                "cpv_code": str(profile.get("code") or ""),
                "cpv_label": str(profile.get("label") or ""),
                "cpv_parent_code": str(profile.get("parent_code") or ""),
                "cpv_parent_label": str(profile.get("parent_label") or ""),
                "description_en": str(profile.get("description_en") or ""),
                "description_multilingual_aliases": str(profile.get("generated_synonyms_en") or ""),
                "generated_synonyms_en": str(profile.get("generated_synonyms_en") or ""),
                "generated_keywords_en": str(profile.get("keywords_en") or ""),
                "keywords_en": str(profile.get("keywords_en") or ""),
                "procurement_type": str(profile.get("procurement_type") or ""),
                "use_when_text": str(profile.get("use_when_text") or ""),
                "do_not_use_when_text": str(profile.get("do_not_use_when_text") or ""),
                "common_tender_phrases": str(profile.get("common_tender_phrases") or ""),
                "children_labels": " | ".join(str(item) for item in profile.get("children_labels", []) if str(item).strip()),
                "sibling_labels": " | ".join(str(item) for item in profile.get("sibling_labels", []) if str(item).strip()),
                "examples_text": " | ".join(str(item) for item in profile.get("examples", []) if str(item).strip()),
                "notice_examples_count": len([item for item in profile.get("notice_examples", []) if str(item).strip()]),
                "search_text_en": str(profile.get("search_text_en") or ""),
                "search_text_multilingual": str(profile.get("search_text_multilingual") or ""),
            }
        )
    return chunks


'''Build a lookup table of parent codes for each CPV code.
For example, if we have 33141800
then we can build the following lookup table:
{
  "33141800": "33141000",
  "33141000": "33140000",
  "33140000": "33100000"
}'''
def build_parent_lookup(records: Sequence[CPVRecord]) -> Dict[str, str]:
    parent_lookup: Dict[str, str] = {}

    def infer_parent_code(code: str) -> str:
        digits = "".join(char for char in code if char.isdigit())
        if len(digits) != 8:
            return ""
        for width in [7, 6, 5, 4, 3, 2]:
            prefix = digits[:width]
            if set(digits[width:]) == {"0"}:
                continue
            candidate = prefix + ("0" * (8 - width))
            if candidate != digits:
                return candidate
        return ""

    def register_ancestor_chain(code: str) -> None:
        current = code
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            parent = infer_parent_code(current)
            if not parent:
                break
            parent_lookup.setdefault(current, parent)
            current = parent

    for record in records:
        if record.parent_code:
            parent_lookup[record.code] = record.parent_code
        else:
            register_ancestor_chain(record.code)
    return parent_lookup
