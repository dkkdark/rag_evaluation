from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class CPVRecord:
    code: str
    label: str
    description: str
    parent_code: str
    examples: List[str]


@dataclass
class MaterialQuery:
    id: str
    query: str
    gold_cpv_code: str
    notes: str


def load_cpv_catalog(path: str) -> List[CPVRecord]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"code", "label"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"CPV catalog is missing required columns: {sorted(missing)}")

    records: List[CPVRecord] = []
    for row in rows:
        examples_raw = row.get("examples", "").strip()
        examples = [item.strip() for item in examples_raw.split(" || ") if item.strip()]
        records.append(
            CPVRecord(
                code=row["code"].strip(),
                label=row["label"].strip(),
                description=row.get("description", "").strip(),
                parent_code=row.get("parent_code", "").strip(),
                examples=examples,
            )
        )
    return records


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


def build_cpv_chunks(records: Sequence[CPVRecord], *, use_examples: bool) -> List[Dict]:
    chunks: List[Dict] = []
    for index, record in enumerate(records):
        parts = [record.label, record.description]
        if use_examples and record.examples:
            parts.append("Examples: " + " | ".join(record.examples))
        text = "\n".join(part for part in parts if part.strip())
        chunks.append(
            {
                "chunk_id": record.code,
                "chunk_index": index,
                "doc_id": record.code,
                "doc_path": "cpv_catalog",
                "program_id": "cpv",
                "program_name": "CPV",
                "section_id": record.code,
                "title": record.label,
                "text": text,
                "chunking_strategy": "cpv_entry",
                "source_type": "cpv_catalog",
                "cpv_code": record.code,
                "cpv_label": record.label,
                "cpv_parent_code": record.parent_code,
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
