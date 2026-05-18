from __future__ import annotations

import json
from typing import List

from rag_eval.evaluation.core import EvaluationItem


def build_items_from_cpv_queries(path: str) -> List[EvaluationItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["queries"] if isinstance(data, dict) and "queries" in data else data
    return [
        EvaluationItem(
            id=str(row.get("id", f"q{index + 1}")),
            query=str(row["query"]),
            gold_labels=[str(row["gold_cpv_code"])],
            metadata={"notes": row.get("notes", "")},
        )
        for index, row in enumerate(rows)
    ]
