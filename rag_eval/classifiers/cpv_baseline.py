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
    load_notice_examples_by_cpv_detailed,
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

_CANONICAL_PROFILE_OVERRIDES: Dict[str, Dict[str, str]] = {
    "72000000": {
        "description_en": (
            "IT services: consulting, software development, internet and support services. "
            "Use for broad information-technology service contracts when the tender is about software development, "
            "application services, cloud, hosting, digital systems, network services, support, maintenance, or managed IT operations."
        ),
        "keywords_en": (
            "it services software development application services cloud hosting digital systems "
            "network services support maintenance managed services"
        ),
        "use_when_text": (
            "Use when the main contract is for broad IT or software-related services, such as application development, "
            "system evolution, cloud or hosting services, network services, IT support, or managed digital operations."
        ),
        "do_not_use_when_text": (
            "Do not use when the procurement is mainly for buying hardware, communications equipment, packaged office software, "
            "or a more specific child IT service code clearly matches the service being bought."
        ),
        "common_tender_phrases": (
            "software development services | application maintenance and support | cloud and hosting services | "
            "digital platform services | network and infrastructure services | managed IT services"
        ),
    },
    "50000000": {
        "description_en": (
            "Repair and maintenance services. Use for broad service contracts whose main scope is repair, upkeep, servicing, "
            "preventive or corrective maintenance of equipment, systems, or infrastructure."
        ),
        "keywords_en": (
            "repair maintenance servicing upkeep preventive maintenance corrective maintenance maintenance contract "
            "service contract technical support"
        ),
        "use_when_text": (
            "Use when the main contract is a repair or maintenance service, especially when the tender is framed around servicing, "
            "upkeep, operational support, or maintenance of existing assets rather than buying new goods."
        ),
        "do_not_use_when_text": (
            "Do not use when the buyer mainly procures new equipment, installation works, or a more specific child maintenance-service code "
            "clearly describes the maintained asset type."
        ),
        "common_tender_phrases": (
            "repair and maintenance services | maintenance contract | servicing of existing equipment | technical maintenance support | "
            "preventive and corrective maintenance"
        ),
    },
    "51500000": {
        "description_en": (
            "Installation services of machinery and equipment. Use for service contracts where the main scope is installing, commissioning, "
            "assembling, fitting, integrating, or putting equipment into operation."
        ),
        "keywords_en": (
            "installation services commissioning assembly fitting integration equipment installation machine installation "
            "put into operation"
        ),
        "use_when_text": (
            "Use when the main contract is an installation or commissioning service for machinery, equipment, technical systems, or devices, "
            "including integration into an operational environment."
        ),
        "do_not_use_when_text": (
            "Do not use when the main scope is the purchase of equipment only, ongoing maintenance after installation, construction works, "
            "or a more specific child installation-service code clearly matches the equipment type."
        ),
        "common_tender_phrases": (
            "installation and commissioning services | equipment installation | machine assembly and fitting | "
            "system installation services | putting equipment into operation"
        ),
    },
    "79000000": {
        "description_en": (
            "Business services: law, marketing, consulting, recruitment, printing and security. "
            "Use for broad business-support service contracts, especially consultancy, communication, marketing, recruitment, "
            "administrative support, and similar professional services."
        ),
        "keywords_en": (
            "business services consultancy consulting communication marketing recruitment administrative support "
            "professional services printing legal services"
        ),
        "use_when_text": (
            "Use when the main contract is for broad business or professional support services, such as consultancy, project communication, "
            "marketing, recruitment, administrative support, or related business-service activities."
        ),
        "do_not_use_when_text": (
            "Do not use when the tender is mainly for IT development, construction works, equipment supply, or when a more specific child code "
            "for advertising, market research, recruitment, legal, printing, or security services clearly fits."
        ),
        "common_tender_phrases": (
            "business support services | consultancy services | project communication services | marketing services | "
            "professional advisory services | administrative support services"
        ),
    },
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


def _catalog_fingerprint(records: Sequence[CPVRecord]) -> str:
    digest = hashlib.sha1()
    digest.update(b"use_examples=always_on")
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
        digest.update(" || ".join(record.examples).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _ancestor_path_codes(code: str, parent_lookup: Dict[str, str]) -> List[str]:
    normalized = str(code or "").strip()
    if not normalized:
        return []
    path: List[str] = []
    seen: set[str] = set()
    current = normalized
    while current and current not in seen:
        seen.add(current)
        path.append(current)
        current = parent_lookup.get(current, "")
    path.reverse()
    return path


def _taxonomy_profile_text(profile: Dict[str, object]) -> str:
    return "\n".join(
        part
        for part in [
            f"Code: {str(profile.get('code') or '')}".strip(),
            f"Label: {str(profile.get('label') or '')}".strip(),
            f"Description: {str(profile.get('description_en') or '')}".strip() if str(profile.get("description_en") or "").strip() else "",
            f"Hierarchy path: {str(profile.get('hierarchy_path_text') or '')}".strip() if str(profile.get("hierarchy_path_text") or "").strip() else "",
            f"Parent: {str(profile.get('parent_label') or '')}".strip() if str(profile.get("parent_label") or "").strip() else "",
            f"Procurement type: {str(profile.get('procurement_type') or '')}".strip() if str(profile.get("procurement_type") or "").strip() else "",
            str(profile.get("use_when_text") or ""),
            str(profile.get("do_not_use_when_text") or ""),
            f"Related terms: {str(profile.get('keywords_en') or '')}".strip() if str(profile.get("keywords_en") or "").strip() else "",
            f"Multilingual labels: {str(profile.get('generated_synonyms_en') or '')}".strip() if str(profile.get("generated_synonyms_en") or "").strip() else "",
            f"Sibling distinctions: {' | '.join(str(item) for item in profile.get('sibling_labels', []) if str(item).strip())}" if profile.get("sibling_labels") else "",
            f"Children: {' | '.join(str(item) for item in profile.get('children_labels', []) if str(item).strip())}" if profile.get("children_labels") else "",
        ]
        if str(part).strip()
    )


def _build_cpv_profiles(
    records: Sequence[CPVRecord],
    *,
    ted_notice_db_path: str | None = DEFAULT_TED_NOTICE_DB_PATH,
    ted_notice_examples_limit: int = 12,
) -> List[Dict[str, object]]:
    label_by_code = {record.code: record.label for record in records}
    parent_lookup = build_parent_lookup(records)
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
        hierarchy_path_codes = _ancestor_path_codes(record.code, parent_lookup)
        hierarchy_path_labels = [label_by_code.get(code, code) for code in hierarchy_path_codes]
        hierarchy_path_text = " > ".join(
            f"{code} {label}".strip()
            for code, label in zip(hierarchy_path_codes, hierarchy_path_labels)
            if code or label
        )
        children = children_by_parent.get(record.code, [])
        siblings = [child for child in children_by_parent.get(record.parent_code, []) if child.code != record.code]
        notice_examples = ted_examples_by_cpv.get(record.code, [])
        merged_examples = _unique_texts(
            list(record.examples) + notice_examples,
            limit=ted_notice_examples_limit,
        )
        keywords_en = _extract_keywords_en(
            record.label,
            record.description,
            " ".join(record.multilingual_labels[:8]),
            " ".join(merged_examples[:5]),
            " ".join(child.label for child in children[:4]),
        )
        multilingual_aliases = _multilingual_aliases(record.label, record.description, keywords_en)
        procurement_type = _infer_procurement_type(record.label, record.description, keywords_en, parent_label)
        use_when = _use_when_text(record, parent_label=parent_label, procurement_type=procurement_type, keywords_en=keywords_en)
        do_not_use_when = _do_not_use_when_text(record, procurement_type=procurement_type)
        common_phrases = _unique_texts(merged_examples[:8], limit=8)
        children_labels = [child.label for child in children[:8]]
        children_codes = [child.code for child in children[:8]]
        sibling_labels = [sibling.label for sibling in siblings[:8]]
        sibling_codes = [sibling.code for sibling in siblings[:8]]
        representative_notice_example_text = notice_examples[0] if notice_examples else (merged_examples[0] if merged_examples else "")
        taxonomy_text_parts = [
            f"Code: {record.code}",
            f"Label: {record.label}",
            f"Description: {record.description}" if record.description else "",
            f"Hierarchy path: {hierarchy_path_text}" if hierarchy_path_text else "",
            f"Multilingual labels: {' | '.join(record.multilingual_labels[:8])}" if record.multilingual_labels else "",
            f"Parent: {parent_label}" if parent_label else "",
            f"Procurement type: {procurement_type}",
            use_when,
            do_not_use_when,
            f"Related terms: {keywords_en}" if keywords_en else "",
            f"Sibling distinctions: {' | '.join(sibling_labels[:8])}" if sibling_labels else "",
            f"Children: {' | '.join(children_labels[:8])}" if children_labels else "",
        ]
        text_parts = [
            *taxonomy_text_parts,
            f"Common tender phrases: {' | '.join(common_phrases)}" if common_phrases else "",
            f"Representative notice example: {representative_notice_example_text}" if representative_notice_example_text else "",
        ]
        search_text_en = "\n".join(
            part for part in [
                record.code,
                record.label,
                record.description,
                hierarchy_path_text,
                " | ".join(record.multilingual_labels),
                parent_label,
                procurement_type,
                keywords_en,
                use_when,
                do_not_use_when,
                " ".join(common_phrases),
                " ".join(children_labels[:8]),
                " ".join(sibling_labels[:8]),
            ] if str(part).strip()
        )
        search_text_multilingual = "\n".join(
            part for part in [
                search_text_en,
                " | ".join(record.multilingual_labels),
                multilingual_aliases,
                " | ".join(notice_examples[:8]),
            ] if str(part).strip()
        )
        override = _CANONICAL_PROFILE_OVERRIDES.get(record.code)
        if override:
            description_en = str(override.get("description_en") or record.description)
            keywords_en = str(override.get("keywords_en") or keywords_en)
            use_when = str(override.get("use_when_text") or use_when)
            do_not_use_when = str(override.get("do_not_use_when_text") or do_not_use_when)
            common_phrases = _unique_texts(
                str(override.get("common_tender_phrases") or "").split(" | "),
                limit=8,
            ) or common_phrases
            taxonomy_text_parts = [
                f"Code: {record.code}",
                f"Label: {record.label}",
                f"Description: {description_en}" if description_en else "",
                f"Hierarchy path: {hierarchy_path_text}" if hierarchy_path_text else "",
                f"Multilingual labels: {' | '.join(record.multilingual_labels[:8])}" if record.multilingual_labels else "",
                f"Parent: {parent_label}" if parent_label else "",
                f"Procurement type: {procurement_type}",
                use_when,
                do_not_use_when,
                f"Related terms: {keywords_en}" if keywords_en else "",
                f"Sibling distinctions: {' | '.join(sibling_labels[:8])}" if sibling_labels else "",
                f"Children: {' | '.join(children_labels[:8])}" if children_labels else "",
            ]
            text_parts = [
                *taxonomy_text_parts,
                f"Common tender phrases: {' | '.join(common_phrases)}" if common_phrases else "",
                f"Representative notice example: {representative_notice_example_text}" if representative_notice_example_text else "",
            ]
            search_text_en = "\n".join(
                part for part in [
                    record.code,
                    record.label,
                    description_en,
                    hierarchy_path_text,
                    " | ".join(record.multilingual_labels),
                    parent_label,
                    procurement_type,
                    keywords_en,
                    use_when,
                    do_not_use_when,
                    " ".join(common_phrases),
                    " ".join(children_labels[:8]),
                    " ".join(sibling_labels[:8]),
                ] if str(part).strip()
            )
            search_text_multilingual = "\n".join(
                part for part in [
                    search_text_en,
                    " | ".join(record.multilingual_labels),
                    multilingual_aliases,
                    " | ".join(notice_examples[:8]),
                ] if str(part).strip()
            )
        else:
            description_en = record.description
        profiles.append(
            {
                "code": record.code,
                "label": record.label,
                "description_en": description_en,
                "parent_code": record.parent_code,
                "parent_label": parent_label,
                "hierarchy_path_codes": hierarchy_path_codes,
                "hierarchy_path_labels": hierarchy_path_labels,
                "hierarchy_path_text": hierarchy_path_text,
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
                "taxonomy_text": "\n".join(part for part in taxonomy_text_parts if part.strip()),
                "examples": merged_examples,
                "notice_examples": notice_examples[:ted_notice_examples_limit],
                "representative_notice_example_text": representative_notice_example_text,
                "text": "\n".join(part for part in text_parts if part.strip()),
                "search_text_en": search_text_en,
                "search_text_multilingual": search_text_multilingual,
            }
        )
    return profiles


def sync_cpv_profiles_to_db(
    records: Sequence[CPVRecord],
    *,
    ted_notice_db_path: str = DEFAULT_TED_NOTICE_DB_PATH,
    ted_notice_examples_limit: int = 12,
) -> Dict[str, object]:
    profiles = _build_cpv_profiles(
        records,
        ted_notice_db_path=ted_notice_db_path,
        ted_notice_examples_limit=ted_notice_examples_limit,
    )
    fingerprint = _catalog_fingerprint(records)
    conn = open_ted_notice_db(ted_notice_db_path)
    try:
        result = upsert_cpv_profiles(
            conn,
            profiles,
            source_fingerprint=fingerprint,
        )
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
    max_examples_per_cpv: int = 12,
) -> List[CPVRecord]:
    rows = build_cpv_catalog_rows_from_corpus_export(
        path,
        max_examples_per_cpv=max_examples_per_cpv,
    )
    return _load_legacy_cpv_rows(rows)


def merge_cpv_catalogs(
    official_records: Sequence[CPVRecord],
    ted_records: Sequence[CPVRecord],
) -> List[CPVRecord]:
    official_by_code = {record.code: record for record in official_records if str(record.code).strip()}
    ted_by_code = {record.code: record for record in ted_records if str(record.code).strip()}
    merged_codes = sorted(set(official_by_code.keys()) | set(ted_by_code.keys()))
    merged: List[CPVRecord] = []
    for code in merged_codes:
        official = official_by_code.get(code)
        ted = ted_by_code.get(code)
        base = official or ted
        if base is None:
            continue
        merged.append(
            CPVRecord(
                code=code,
                label=(
                    str((official.label if official and official.label else "") or (ted.label if ted else "")).strip()
                    or code
                ),
                description=(
                    str((official.description if official and official.description else "") or (ted.description if ted else "")).strip()
                    or str((official.label if official and official.label else "") or (ted.label if ted else "")).strip()
                    or code
                ),
                parent_code=str((official.parent_code if official and official.parent_code else "") or (ted.parent_code if ted else "")).strip(),
                examples=_unique_texts(list(ted.examples if ted else []) + list(official.examples if official else []), limit=12),
                multilingual_labels=_unique_texts(list(official.multilingual_labels if official else []) + list(ted.multilingual_labels if ted else []), limit=24),
            )
        )
    return merged


def build_cpv_chunks_from_db(path: str) -> List[Dict]:
    profiles = load_cpv_profiles(path)
    chunks: List[Dict] = []
    for index, profile in enumerate(profiles):
        examples = [str(item).strip() for item in profile.get("examples", []) if str(item).strip()]
        taxonomy_text = str(profile.get("taxonomy_text") or _taxonomy_profile_text(profile))
        representative_notice_example_text = str(
            profile.get("representative_notice_example_text")
            or next((str(item).strip() for item in profile.get("notice_examples", []) if str(item).strip()), "")
            or (examples[0] if examples else "")
        )
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
                "text": str(profile.get("text") or taxonomy_text),
                "chunking_strategy": "cpv_profile_db",
                "source_type": "cpv_profiles_db",
                "cpv_code": str(profile.get("code") or ""),
                "cpv_label": str(profile.get("label") or ""),
                "cpv_parent_code": str(profile.get("parent_code") or ""),
                "cpv_parent_label": str(profile.get("parent_label") or ""),
                "cpv_hierarchy_path_codes": " | ".join(str(item) for item in profile.get("hierarchy_path_codes", []) if str(item).strip()),
                "cpv_hierarchy_path_labels": " | ".join(str(item) for item in profile.get("hierarchy_path_labels", []) if str(item).strip()),
                "cpv_hierarchy_path_text": str(profile.get("hierarchy_path_text") or ""),
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
                "representative_notice_example_text": representative_notice_example_text,
                "taxonomy_text": taxonomy_text,
                "notice_examples_count": len([item for item in profile.get("notice_examples", []) if str(item).strip()]),
                "search_text_en": str(profile.get("search_text_en") or ""),
                "search_text_multilingual": str(profile.get("search_text_multilingual") or ""),
            }
        )
    return chunks


def build_cpv_notice_example_chunks_from_db(
    path: str,
    *,
    max_examples_per_cpv: int = 12,
) -> List[Dict]:
    profiles = load_cpv_profiles(path)
    notice_examples_by_cpv = load_notice_examples_by_cpv_detailed(path, max_examples_per_cpv=max_examples_per_cpv)
    chunks: List[Dict] = []
    for profile_index, profile in enumerate(profiles):
        cpv_code = str(profile.get("code") or "").strip()
        if not cpv_code:
            continue
        example_rows = notice_examples_by_cpv.get(cpv_code, [])
        examples = [str(item.get("example_text") or "").strip() for item in example_rows if str(item.get("example_text") or "").strip()]
        publication_numbers = [str(item.get("publication_number") or "").strip() for item in example_rows]
        if not examples:
            continue
        for example_index, example_text in enumerate(examples):
            taxonomy_text = str(profile.get("taxonomy_text") or _taxonomy_profile_text(profile))
            chunks.append(
                {
                    "chunk_id": f"{cpv_code}::notice_example::{example_index + 1}",
                    "chunk_index": len(chunks),
                    "doc_id": cpv_code,
                    "doc_path": "cpv_notice_examples_db",
                    "publication_number": publication_numbers[example_index] if example_index < len(publication_numbers) else "",
                    "program_id": "cpv",
                    "program_name": "CPV",
                    "section_id": cpv_code,
                    "title": str(profile.get("label") or ""),
                    "text": example_text,
                    "chunking_strategy": "cpv_notice_example_db",
                    "source_type": "cpv_notice_examples_db",
                    "cpv_code": cpv_code,
                    "cpv_label": str(profile.get("label") or ""),
                    "cpv_parent_code": str(profile.get("parent_code") or ""),
                    "cpv_parent_label": str(profile.get("parent_label") or ""),
                    "cpv_hierarchy_path_codes": " | ".join(str(item) for item in profile.get("hierarchy_path_codes", []) if str(item).strip()),
                    "cpv_hierarchy_path_labels": " | ".join(str(item) for item in profile.get("hierarchy_path_labels", []) if str(item).strip()),
                    "cpv_hierarchy_path_text": str(profile.get("hierarchy_path_text") or ""),
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
                    "examples_text": example_text,
                    "representative_notice_example_text": example_text,
                    "taxonomy_text": taxonomy_text,
                    "notice_examples_count": len(examples),
                    "notice_example_publication_numbers": publication_numbers,
                    "search_text_en": "\n".join(
                        part
                        for part in [
                            cpv_code,
                            str(profile.get("label") or ""),
                            str(profile.get("hierarchy_path_text") or ""),
                            example_text,
                            str(profile.get("keywords_en") or ""),
                            str(profile.get("use_when_text") or ""),
                            str(profile.get("do_not_use_when_text") or ""),
                        ]
                        if str(part).strip()
                    ),
                    "search_text_multilingual": "\n".join(
                        part
                        for part in [
                            cpv_code,
                            str(profile.get("label") or ""),
                            taxonomy_text,
                            example_text,
                        ]
                        if str(part).strip()
                    ),
                    "profile_chunk_index": profile_index,
                    "notice_example_index": example_index,
                }
            )
    return chunks


def build_cpv_chunks(
    records: Sequence[CPVRecord],
    *,
    ted_notice_db_path: str | None = DEFAULT_TED_NOTICE_DB_PATH,
    ted_notice_examples_limit: int = 12,
) -> List[Dict]:
    if ted_notice_db_path:
        sync_cpv_profiles_to_db(
            records,
            ted_notice_db_path=ted_notice_db_path,
            ted_notice_examples_limit=ted_notice_examples_limit,
        )
        db_chunks = build_cpv_chunks_from_db(ted_notice_db_path)
        if db_chunks:
            return db_chunks
    profiles = _build_cpv_profiles(
        records,
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
                "representative_notice_example_text": str(profile.get("representative_notice_example_text") or ""),
                "taxonomy_text": str(profile.get("taxonomy_text") or ""),
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
