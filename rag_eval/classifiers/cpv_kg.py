from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

from rag_eval.classifiers.cpv_baseline import CPVRecord, build_parent_lookup
from rag_eval.evaluation.core import best_cpv_hierarchy_match, cpv_common_prefix_length
from rag_eval.retrieval.engines import lexical_overlap_score, min_max_normalize


CPV_LEVELS = [2, 4, 6, 8]
CPV_DEFAULT_POOL_SIZE = 40

CPV_SELECTION_WEIGHTS = {
    "base": 0.52,
    "lexical": 0.33,
    "graph": 0.15,
    "sibling_boost": 0.22,
    "class_cluster_boost": 0.16,
}
CPV_REFINEMENT_BASE_WINDOW = 15
CPV_REFINEMENT_SEED_WINDOW = 5
SAFE_BRANCH_PREFIX_LEVELS = (6, 4)
HIERARCHY_STAGE_TOP_COUNTS = {2: 6, 4: 4, 6: 3}
HIERARCHY_STAGE_RELATIVE_THRESHOLDS = {2: 0.82, 4: 0.86, 6: 0.88}


@dataclass
class CPVGraphNode:
    code: str
    label: str
    description: str
    parent_code: str
    path_codes: List[str]
    path_labels: List[str]
    examples: List[str]


@dataclass
class CPVKnowledgeGraph:
    nodes: Dict[str, CPVGraphNode]
    parent_lookup: Dict[str, str]
    children_lookup: Dict[str, List[str]]
    sibling_lookup: Dict[str, List[str]]


def normalize_cpv_code(value: str) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits if len(digits) == 8 else ""


def cpv_prefix_code(code: str, width: int) -> str:
    normalized = normalize_cpv_code(code)
    if not normalized or width > len(normalized):
        return ""
    return normalized[:width] + ("0" * (8 - width))


def code_level(code: str) -> int:
    normalized = normalize_cpv_code(code)
    if not normalized:
        return 0
    for width in CPV_LEVELS:
        candidate = cpv_prefix_code(normalized, width)
        if candidate == normalized:
            return width
    return 8


def build_children_lookup(parent_lookup: Dict[str, str]) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = {}
    for child, parent in parent_lookup.items():
        if not child or not parent:
            continue
        children.setdefault(parent, []).append(child)
    for rows in children.values():
        rows.sort()
    return children


def ancestor_codes(code: str, parent_lookup: Dict[str, str], *, include_self: bool = True) -> List[str]:
    current = normalize_cpv_code(code)
    if not current:
        return []
    out = [current] if include_self else []
    seen = set(out)
    while current:
        parent = parent_lookup.get(current, "")
        if not parent or parent in seen:
            break
        out.append(parent)
        seen.add(parent)
        current = parent
    return out


def build_cpv_knowledge_graph(records: Sequence[CPVRecord]) -> CPVKnowledgeGraph:
    parent_lookup = build_parent_lookup(records)
    records_by_code = {record.code: record for record in records}
    children_lookup = build_children_lookup(parent_lookup)

    nodes: Dict[str, CPVGraphNode] = {}
    for record in records:
        path_codes = list(reversed(ancestor_codes(record.code, parent_lookup, include_self=True)))
        path_labels = [
            records_by_code[path_code].label
            for path_code in path_codes
            if path_code in records_by_code and records_by_code[path_code].label
        ]
        nodes[record.code] = CPVGraphNode(
            code=record.code,
            label=record.label,
            description=record.description,
            parent_code=parent_lookup.get(record.code, record.parent_code),
            path_codes=path_codes,
            path_labels=path_labels,
            examples=list(record.examples),
        )

    sibling_lookup: Dict[str, List[str]] = {}
    for code in nodes:
        parent = parent_lookup.get(code, "")
        if not parent:
            sibling_lookup[code] = []
            continue
        sibling_lookup[code] = [sibling for sibling in children_lookup.get(parent, []) if sibling != code]

    return CPVKnowledgeGraph(
        nodes=nodes,
        parent_lookup=parent_lookup,
        children_lookup=children_lookup,
        sibling_lookup=sibling_lookup,
    )


def _select_stage_prefixes(
    *,
    codes: Sequence[str],
    combined_scores: Dict[str, float],
    lexical_norm: Dict[str, float],
    width: int,
) -> List[str]:
    groups = _group_codes_by_prefix(list(codes), width)
    if not groups:
        return []
    scored_prefixes: List[tuple[str, float]] = []
    for prefix, members in groups.items():
        member_scores = sorted((combined_scores.get(code, 0.0) for code in members), reverse=True)
        member_lex = sorted((lexical_norm.get(code, 0.0) for code in members), reverse=True)
        score = (
            0.55 * (sum(member_scores[:3]) / max(1, min(3, len(member_scores))))
            + 0.30 * (max(member_scores) if member_scores else 0.0)
            + 0.10 * (max(member_lex) if member_lex else 0.0)
            + min(0.05, 0.01 * len(members))
        )
        scored_prefixes.append((prefix, score))
    scored_prefixes.sort(key=lambda item: item[1], reverse=True)
    if not scored_prefixes:
        return []
    best_score = scored_prefixes[0][1]
    threshold = best_score * HIERARCHY_STAGE_RELATIVE_THRESHOLDS.get(width, 0.9)
    top_count = HIERARCHY_STAGE_TOP_COUNTS.get(width, 2)
    selected = [
        prefix
        for index, (prefix, score) in enumerate(scored_prefixes)
        if index < top_count or score >= threshold
    ]
    return selected


def cpv_path_text(node: CPVGraphNode) -> str:
    parts = list(node.path_labels) + [node.label, node.description]
    if node.examples:
        parts.append(" ".join(node.examples[:3]))
    return "\n".join(part for part in parts if part.strip())


def export_cpv_kg_for_neo4j(
    graph: CPVKnowledgeGraph,
    *,
    out_dir: str,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    nodes_csv = os.path.join(out_dir, "cpv_kg_nodes.csv")
    edges_csv = os.path.join(out_dir, "cpv_kg_edges.csv")
    cypher_path = os.path.join(out_dir, "import_neo4j.cypher")
    readme_path = os.path.join(out_dir, "README.md")

    with open(nodes_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "code:ID(CPV)",
                "label",
                "description",
                "parent_code",
                "code_level:int",
                "path_codes",
                "path_labels",
                "examples",
                ":LABEL",
            ],
        )
        writer.writeheader()
        for code in sorted(graph.nodes.keys()):
            node = graph.nodes[code]
            writer.writerow(
                {
                    "code:ID(CPV)": node.code,
                    "label": node.label,
                    "description": node.description,
                    "parent_code": node.parent_code,
                    "code_level:int": code_level(node.code),
                    "path_codes": " | ".join(node.path_codes),
                    "path_labels": " | ".join(node.path_labels),
                    "examples": " | ".join(node.examples[:12]),
                    ":LABEL": "CPVCode",
                }
            )

    edge_rows: List[Dict[str, str]] = []
    for child, parent in sorted(graph.parent_lookup.items()):
        if not child or not parent:
            continue
        edge_rows.append(
            {
                ":START_ID(CPV)": child,
                ":END_ID(CPV)": parent,
                ":TYPE": "CHILD_OF",
                "relation": "child_of",
            }
        )
        edge_rows.append(
            {
                ":START_ID(CPV)": parent,
                ":END_ID(CPV)": child,
                ":TYPE": "PARENT_OF",
                "relation": "parent_of",
            }
        )
    for code, siblings in sorted(graph.sibling_lookup.items()):
        for sibling in siblings:
            if code < sibling:
                edge_rows.append(
                    {
                        ":START_ID(CPV)": code,
                        ":END_ID(CPV)": sibling,
                        ":TYPE": "SIBLING_OF",
                        "relation": "sibling_of",
                    }
                )

    with open(edges_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                ":START_ID(CPV)",
                ":END_ID(CPV)",
                ":TYPE",
                "relation",
            ],
        )
        writer.writeheader()
        writer.writerows(edge_rows)

    cypher = """// Copy cpv_kg_nodes.csv and cpv_kg_edges.csv into Neo4j's import directory.
// Then run this script in Neo4j Browser.

CREATE CONSTRAINT cpv_code_unique IF NOT EXISTS
FOR (n:CPVCode)
REQUIRE n.code IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///cpv_kg_nodes.csv' AS row
MERGE (n:CPVCode {code: row['code:ID(CPV)']})
SET n.label = row.label,
    n.description = row.description,
    n.parent_code = row.parent_code,
    n.code_level = toInteger(row['code_level:int']),
    n.path_codes = split(COALESCE(row.path_codes, ''), ' | '),
    n.path_labels = split(COALESCE(row.path_labels, ''), ' | '),
    n.examples = split(COALESCE(row.examples, ''), ' | ');

LOAD CSV WITH HEADERS FROM 'file:///cpv_kg_edges.csv' AS row
MATCH (a:CPVCode {code: row[':START_ID(CPV)']})
MATCH (b:CPVCode {code: row[':END_ID(CPV)']})
CALL apoc.merge.relationship(a, row[':TYPE'], {}, {relation: row.relation}, b) YIELD rel
RETURN count(rel);
"""
    with open(cypher_path, "w", encoding="utf-8") as f:
        f.write(cypher)

    readme = """# CPV KG for Neo4j

Files:
- `cpv_kg_nodes.csv`: CPV nodes with labels, descriptions, hierarchy path, and examples
- `cpv_kg_edges.csv`: `CHILD_OF`, `PARENT_OF`, and `SIBLING_OF` relationships
- `import_neo4j.cypher`: Neo4j Browser script for loading the graph

Usage:
1. Copy both CSV files into Neo4j's `import/` directory.
2. Open Neo4j Browser.
3. Run the commands from `import_neo4j.cypher`.

Note:
- The Cypher script uses `apoc.merge.relationship(...)`, so APOC should be available.
- If APOC is unavailable, the CSV can still be imported with separate `LOAD CSV` statements per relation type.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    return {
        "enabled": True,
        "out_dir": out_dir,
        "nodes_csv": nodes_csv,
        "edges_csv": edges_csv,
        "import_cypher": cypher_path,
        "readme_md": readme_path,
        "n_nodes": len(graph.nodes),
        "n_edges": len(edge_rows),
    }


def cpv_kg_profile_settings(profile: str) -> Dict[str, object]:
    profiles = {
        "safe_branch": {
            "algorithm": "hierarchy",
            "max_siblings": 6,
            "max_children": 8,
            "max_seed_codes": 6,
            "max_graph_candidates": 18,
            "proximity_weight": 0.82,
            "strict_branching": True,
            "min_seed_norm": 0.12,
            "min_branch_support": 2,
            "allowed_relations": ["sibling", "child", "ancestor"],
        },
        "conservative": {
            "algorithm": "hierarchy",
            "max_siblings": 4,
            "max_children": 6,
            "max_seed_codes": 8,
            "max_graph_candidates": 35,
            "proximity_weight": 0.78,
            "strict_branching": False,
        },
        "balanced": {
            "algorithm": "hierarchy",
            "max_siblings": 8,
            "max_children": 12,
            "max_seed_codes": 12,
            "max_graph_candidates": 60,
            "proximity_weight": 0.70,
            "strict_branching": False,
        },
        "exploratory": {
            "algorithm": "ppr",
            "max_siblings": 16,
            "max_children": 24,
            "max_seed_codes": 18,
            "max_graph_candidates": 120,
            "proximity_weight": 0.62,
            "strict_branching": False,
        },
        "ppr_only": {
            "algorithm": "ppr",
            "max_siblings": 12,
            "max_children": 18,
            "max_seed_codes": 12,
            "max_graph_candidates": 80,
            "proximity_weight": 0.68,
            "strict_branching": False,
        },
        "direct_only": {
            "algorithm": "hierarchy",
            "max_siblings": 8,
            "max_children": 12,
            "max_seed_codes": 12,
            "max_graph_candidates": 60,
            "proximity_weight": 0.70,
            "strict_branching": False,
        },
        "selection": {
            "algorithm": "ppr",
            "max_siblings": 24,
            "max_children": 32,
            "max_seed_codes": 20,
            "max_graph_candidates": 160,
            "proximity_weight": 0.62,
            "strict_branching": False,
        },
    }
    return dict(profiles.get(profile, profiles["balanced"]))


def _supported_branch_prefixes(
    *,
    seed_codes: Sequence[str],
    base_norm: Dict[str, float],
    min_seed_norm: float,
    min_branch_support: int,
) -> Dict[int, set[str]]:
    strong_codes = [
        normalize_cpv_code(code)
        for code in seed_codes
        if normalize_cpv_code(code) and float(base_norm.get(code, 0.0)) >= min_seed_norm
    ]
    supported: Dict[int, set[str]] = {prefix_len: set() for prefix_len in SAFE_BRANCH_PREFIX_LEVELS}
    for prefix_len in SAFE_BRANCH_PREFIX_LEVELS:
        groups: Dict[str, List[str]] = defaultdict(list)
        for code in strong_codes:
            groups[code[:prefix_len]].append(code)
        for prefix, members in groups.items():
            if len(members) >= min_branch_support:
                supported[prefix_len].add(prefix)
    return supported


def _belongs_to_supported_branch(
    code: str,
    supported_prefixes: Dict[int, set[str]],
) -> bool:
    normalized = normalize_cpv_code(code)
    if not normalized:
        return False
    for prefix_len, prefixes in supported_prefixes.items():
        if normalized[:prefix_len] in prefixes:
            return True
    return False


def graph_neighbor_codes(
    code: str,
    graph: CPVKnowledgeGraph,
    *,
    max_siblings: int = 8,
    max_children: int = 12,
) -> Dict[str, str]:
    neighbors: Dict[str, str] = {}
    normalized = normalize_cpv_code(code)
    if not normalized:
        return neighbors

    for ancestor in ancestor_codes(normalized, graph.parent_lookup, include_self=False):
        if ancestor in graph.nodes:
            neighbors.setdefault(ancestor, "ancestor")

    siblings = graph.sibling_lookup.get(normalized, [])
    for sibling in siblings[:max_siblings]:
        if sibling in graph.nodes:
            neighbors.setdefault(sibling, "sibling")

    for child in graph.children_lookup.get(normalized, [])[:max_children]:
        if child in graph.nodes:
            neighbors.setdefault(child, "child")

    return neighbors


def _candidate_path(
    *,
    seed_code: str,
    candidate_code: str,
    relation_type: str,
    graph: CPVKnowledgeGraph,
) -> str:
    seed_node = graph.nodes.get(seed_code)
    candidate_node = graph.nodes.get(candidate_code)
    if seed_node is None or candidate_node is None:
        return relation_type
    seed_label = seed_node.label or seed_code
    candidate_label = candidate_node.label or candidate_code
    return f"{seed_code} {seed_label} -[{relation_type}]- {candidate_code} {candidate_label}"


def _candidate_rich_text(
    code: str,
    *,
    graph: CPVKnowledgeGraph,
    chunks_by_code: Dict[str, Dict[str, object]],
) -> str:
    node = graph.nodes.get(code)
    chunk = chunks_by_code.get(code)
    parts: List[str] = []
    if node is not None:
        parts.append(cpv_path_text(node))
    if chunk is not None:
        parts.append(str(chunk.get("title") or ""))
        parts.append(str(chunk.get("text") or ""))
    return "\n".join(part for part in parts if part.strip())


def _expand_cpv_candidate_pool(
    *,
    query: str,
    base_rows: Sequence[Dict[str, object]],
    chunks_by_code: Dict[str, Dict[str, object]],
    graph: CPVKnowledgeGraph,
    kg_profile: str,
    graph_algorithm: str | None = None,
    max_seed_codes: int | None = None,
    max_graph_candidates: int | None = None,
) -> tuple[
    List[str],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, str],
    Dict[str, str],
    Dict[str, object],
]:
    settings = cpv_kg_profile_settings(kg_profile)
    resolved_algorithm = graph_algorithm or str(settings["algorithm"])
    resolved_max_seed_codes = int(max_seed_codes if max_seed_codes is not None else settings["max_seed_codes"])
    resolved_max_graph_candidates = int(
        max_graph_candidates if max_graph_candidates is not None else settings["max_graph_candidates"]
    )
    max_siblings = int(settings["max_siblings"])
    max_children = int(settings["max_children"])
    proximity_weight = float(settings["proximity_weight"])
    strict_branching = bool(settings.get("strict_branching", False))
    min_seed_norm = float(settings.get("min_seed_norm", 0.0))
    min_branch_support = int(settings.get("min_branch_support", 2))
    allowed_relations = {
        str(item)
        for item in settings.get("allowed_relations", ["ancestor", "sibling", "child"])
    }

    base_codes = [
        str(row.get("cpv_code") or row.get("chunk_id") or "").strip()
        for row in base_rows
        if str(row.get("cpv_code") or row.get("chunk_id") or "").strip()
    ]
    if not base_codes:
        return [], {}, {}, {}, {}, {}, {
            "kg_profile": kg_profile,
            "graph_algorithm": resolved_algorithm,
            "kg_settings": settings,
        }

    base_scores = [float(row.get("score") or 0.0) for row in base_rows]
    base_norm = dict(zip(base_codes, min_max_normalize(base_scores)))
    seed_codes = list(dict.fromkeys(base_codes[:resolved_max_seed_codes]))
    supported_prefixes = _supported_branch_prefixes(
        seed_codes=seed_codes,
        base_norm=base_norm,
        min_seed_norm=min_seed_norm,
        min_branch_support=min_branch_support,
    ) if strict_branching else {}
    has_supported_branch = any(supported_prefixes.values()) if strict_branching else False

    candidate_reasons: Dict[str, str] = {code: "base" for code in base_codes}
    candidate_paths: Dict[str, str] = {}
    proximity_scores: Dict[str, float] = {}
    for rank, seed_code in enumerate(seed_codes):
        seed_strength = max(base_norm.get(seed_code, 0.0), 1.0 / (rank + 1))
        proximity_scores[seed_code] = max(proximity_scores.get(seed_code, 0.0), seed_strength)
        if strict_branching and not has_supported_branch:
            continue
        if strict_branching and seed_strength < min_seed_norm and not _belongs_to_supported_branch(seed_code, supported_prefixes):
            continue
        for neighbor_code, relation_type in graph_neighbor_codes(
            seed_code,
            graph,
            max_siblings=max_siblings,
            max_children=max_children,
        ).items():
            if neighbor_code not in chunks_by_code:
                continue
            if relation_type not in allowed_relations:
                continue
            if strict_branching and has_supported_branch:
                if not (
                    _belongs_to_supported_branch(seed_code, supported_prefixes)
                    or _belongs_to_supported_branch(neighbor_code, supported_prefixes)
                ):
                    continue
            decay = {
                "ancestor": 0.64,
                "sibling": 0.52,
                "child": 0.58,
            }.get(relation_type, 0.45)
            proximity_scores[neighbor_code] = max(
                proximity_scores.get(neighbor_code, 0.0),
                seed_strength * decay,
            )
            candidate_reasons.setdefault(neighbor_code, relation_type)
            candidate_paths.setdefault(
                neighbor_code,
                _candidate_path(
                    seed_code=seed_code,
                    candidate_code=neighbor_code,
                    relation_type=relation_type,
                    graph=graph,
                ),
            )
            if len(candidate_reasons) >= len(base_codes) + resolved_max_graph_candidates:
                break
    if resolved_algorithm == "ppr" and not strict_branching:
        frontier = {code: proximity_scores.get(code, 0.0) for code in seed_codes}
        for depth in range(3):
            next_frontier: Dict[str, float] = {}
            for seed_code, seed_strength in frontier.items():
                for neighbor_code, relation_type in graph_neighbor_codes(
                    seed_code,
                    graph,
                    max_siblings=max_siblings,
                    max_children=max_children,
                ).items():
                    if neighbor_code not in chunks_by_code:
                        continue
                    decay = {"ancestor": 0.58, "sibling": 0.42, "child": 0.50}.get(relation_type, 0.35)
                    score = seed_strength * decay * (0.72 ** depth)
                    if score > proximity_scores.get(neighbor_code, 0.0):
                        proximity_scores[neighbor_code] = score
                        next_frontier[neighbor_code] = score
                        candidate_reasons.setdefault(neighbor_code, f"ppr_{relation_type}")
                        candidate_paths.setdefault(
                            neighbor_code,
                            _candidate_path(
                                seed_code=seed_code,
                                candidate_code=neighbor_code,
                                relation_type=f"ppr_{relation_type}",
                                graph=graph,
                            ),
                        )
                    if len(candidate_reasons) >= len(base_codes) + resolved_max_graph_candidates:
                        break
            frontier = next_frontier
            if not frontier:
                break

    candidate_codes = list(candidate_reasons.keys())
    lexical_scores: Dict[str, float] = {}
    for code in candidate_codes:
        node = graph.nodes.get(code)
        chunk = chunks_by_code.get(code)
        text = cpv_path_text(node) if node else str(chunk.get("text", "") if chunk else "")
        lexical_scores[code] = lexical_overlap_score(query, text)
    lexical_norm = dict(zip(candidate_codes, min_max_normalize([lexical_scores[code] for code in candidate_codes])))
    graph_raw = [
        proximity_weight * proximity_scores.get(code, 0.0) + (1.0 - proximity_weight) * lexical_norm.get(code, 0.0)
        for code in candidate_codes
    ]
    graph_norm = dict(zip(candidate_codes, min_max_normalize(graph_raw)))

    meta = {
        "kg_profile": kg_profile,
        "graph_algorithm": resolved_algorithm,
        "kg_settings": {
            "max_seed_codes": resolved_max_seed_codes,
            "max_graph_candidates": resolved_max_graph_candidates,
            "max_siblings": max_siblings,
            "max_children": max_children,
            "proximity_weight": proximity_weight,
            "strict_branching": strict_branching,
            "min_seed_norm": min_seed_norm,
            "min_branch_support": min_branch_support,
            "allowed_relations": sorted(allowed_relations),
            "supported_prefixes": {str(k): sorted(v) for k, v in supported_prefixes.items() if v},
        },
    }
    return candidate_codes, base_norm, graph_norm, proximity_scores, candidate_reasons, candidate_paths, meta


def graph_expand_cpv_pool(
    *,
    query: str,
    base_rows: Sequence[Dict[str, object]],
    chunks_by_code: Dict[str, Dict[str, object]],
    graph: CPVKnowledgeGraph,
    kg_profile: str = "balanced",
    graph_algorithm: str | None = None,
    max_seed_codes: int | None = None,
    max_graph_candidates: int | None = None,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    (
        candidate_codes,
        base_norm,
        graph_norm,
        proximity_scores,
        candidate_reasons,
        candidate_paths,
        meta,
    ) = _expand_cpv_candidate_pool(
        query=query,
        base_rows=base_rows,
        chunks_by_code=chunks_by_code,
        graph=graph,
        kg_profile=kg_profile,
        graph_algorithm=graph_algorithm,
        max_seed_codes=max_seed_codes,
        max_graph_candidates=max_graph_candidates,
    )
    base_codes = [
        str(row.get("cpv_code") or row.get("chunk_id") or "").strip()
        for row in base_rows
        if str(row.get("cpv_code") or row.get("chunk_id") or "").strip()
    ]
    if not candidate_codes:
        return list(base_rows), {
            "enabled": True,
            "base_codes": base_codes,
            "candidate_pool_codes": base_codes,
            "added_codes": [],
            "paths": {},
            **meta,
        }

    pool_rows: List[Dict[str, object]] = []
    for code in candidate_codes:
        chunk = chunks_by_code.get(code)
        if chunk is None:
            continue
        row = dict(chunk)
        row["base_retrieval_score"] = float(base_norm.get(code, 0.0))
        row["kg_graph_score"] = float(graph_norm.get(code, 0.0))
        row["kg_path_score"] = float(proximity_scores.get(code, 0.0))
        row["score"] = float(base_norm.get(code, 0.0))
        row["retriever"] = "cpv_kg_augmented" if code not in base_norm else str(
            next((base_row.get("retriever", "") for base_row in base_rows if base_row.get("cpv_code") == code), "")
        )
        row["retrieval_source"] = (
            "vector+graph" if code in base_norm and graph_norm.get(code, 0.0) > 0 else "graph" if code not in base_norm else "vector"
        )
        row["kg_candidate_reason"] = candidate_reasons.get(code, "")
        row["kg_path"] = candidate_paths.get(code, "")
        pool_rows.append(row)

    added_codes = [code for code in candidate_codes if code not in set(base_codes)]
    return pool_rows, {
        "enabled": True,
        "base_codes": base_codes,
        "candidate_pool_codes": candidate_codes,
        "added_codes": added_codes,
        "paths": candidate_paths,
        **meta,
    }


def _group_codes_by_prefix(codes: Sequence[str], prefix_len: int) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for code in codes:
        normalized = normalize_cpv_code(code)
        if len(normalized) >= prefix_len:
            groups[normalized[:prefix_len]].append(normalized)
    return dict(groups)


def _refinement_candidate_codes(
    pool_codes: Sequence[str],
    *,
    graph: CPVKnowledgeGraph,
    base_scores: Dict[str, float],
    rows_by_code: Dict[str, Dict[str, object]],
) -> set[str]:
    base_sorted = sorted(pool_codes, key=lambda code: base_scores.get(code, 0.0), reverse=True)
    refinement = set(base_sorted[:CPV_REFINEMENT_BASE_WINDOW])
    for code in base_sorted[:CPV_REFINEMENT_SEED_WINDOW]:
        for sibling in graph.sibling_lookup.get(code, []):
            if sibling in rows_by_code:
                refinement.add(sibling)
        parent = graph.parent_lookup.get(code, "")
        if parent:
            for child in graph.children_lookup.get(parent, []):
                if child in rows_by_code:
                    refinement.add(child)
    for code in pool_codes:
        row = rows_by_code.get(code, {})
        reason = str(row.get("kg_candidate_reason") or "")
        if reason and reason != "base":
            refinement.add(code)
    return refinement


def hierarchy_select_cpv_candidates(
    *,
    query: str,
    pool_rows: Sequence[Dict[str, object]],
    graph: CPVKnowledgeGraph,
    chunks_by_code: Dict[str, Dict[str, object]],
    top_k: int,
    selection_weights: Dict[str, float] | None = None,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    weights = dict(CPV_SELECTION_WEIGHTS)
    if selection_weights:
        weights.update(selection_weights)

    if top_k <= 0 or not pool_rows:
        return [], {"selection_enabled": True, "pool_size": 0}

    pool_codes = [
        str(row.get("cpv_code") or row.get("chunk_id") or "").strip()
        for row in pool_rows
        if str(row.get("cpv_code") or row.get("chunk_id") or "").strip()
    ]
    if not pool_codes:
        return [], {"selection_enabled": True, "pool_size": 0}

    rows_by_code = {
        str(row.get("cpv_code") or row.get("chunk_id") or "").strip(): dict(row)
        for row in pool_rows
    }

    lexical_scores = {
        code: lexical_overlap_score(
            query,
            _candidate_rich_text(code, graph=graph, chunks_by_code=chunks_by_code),
        )
        for code in pool_codes
    }
    base_scores = {
        code: float(rows_by_code[code].get("base_retrieval_score", rows_by_code[code].get("score", 0.0)))
        for code in pool_codes
    }
    graph_scores = {
        code: float(rows_by_code[code].get("kg_graph_score", 0.0))
        for code in pool_codes
    }

    base_norm = dict(zip(pool_codes, min_max_normalize([base_scores[code] for code in pool_codes])))
    lexical_norm = dict(zip(pool_codes, min_max_normalize([lexical_scores[code] for code in pool_codes])))
    graph_norm = dict(zip(pool_codes, min_max_normalize([graph_scores[code] for code in pool_codes])))

    combined_scores: Dict[str, float] = {}
    sibling_boost_applied: Dict[str, float] = {}
    class_boost_applied: Dict[str, float] = {}
    refinement_codes = _refinement_candidate_codes(
        pool_codes,
        graph=graph,
        base_scores=base_scores,
        rows_by_code=rows_by_code,
    )
    for code in pool_codes:
        combined_scores[code] = float(base_scores.get(code, 0.0))

    for code in refinement_codes:
        combined_scores[code] = (
            weights["base"] * base_norm.get(code, 0.0)
            + weights["lexical"] * lexical_norm.get(code, 0.0)
            + weights["graph"] * graph_norm.get(code, 0.0)
        )

    active_codes = list(refinement_codes)
    selected_division_prefixes = _select_stage_prefixes(
        codes=active_codes,
        combined_scores=combined_scores,
        lexical_norm=lexical_norm,
        width=2,
    )
    if selected_division_prefixes:
        active_codes = [code for code in active_codes if code[:2] in selected_division_prefixes]
    selected_group_prefixes = _select_stage_prefixes(
        codes=active_codes,
        combined_scores=combined_scores,
        lexical_norm=lexical_norm,
        width=4,
    )
    if selected_group_prefixes:
        active_codes = [code for code in active_codes if code[:4] in selected_group_prefixes]
    selected_class_prefixes = _select_stage_prefixes(
        codes=active_codes,
        combined_scores=combined_scores,
        lexical_norm=lexical_norm,
        width=6,
    )
    if selected_class_prefixes:
        active_codes = [code for code in active_codes if code[:6] in selected_class_prefixes]
    if not active_codes:
        active_codes = list(refinement_codes)

    parent_groups: Dict[str, List[str]] = defaultdict(list)
    for code in active_codes:
        parent = graph.parent_lookup.get(code, "")
        if parent:
            parent_groups[parent].append(code)
    for siblings in parent_groups.values():
        if len(siblings) < 2:
            continue
        sibling_lex = [lexical_norm.get(code, 0.0) for code in siblings]
        max_lex = max(sibling_lex) if sibling_lex else 0.0
        if max_lex <= 0.0:
            continue
        for code in siblings:
            boost = weights["sibling_boost"] * (lexical_norm.get(code, 0.0) / max_lex)
            combined_scores[code] += boost
            sibling_boost_applied[code] = boost

    for prefix_len in (6, 4):
        for _, members in _group_codes_by_prefix(list(active_codes), prefix_len).items():
            if len(members) < 2:
                continue
            cluster_lex = [lexical_norm.get(code, 0.0) for code in members]
            max_lex = max(cluster_lex) if cluster_lex else 0.0
            if max_lex <= 0.0:
                continue
            for code in members:
                boost = weights["class_cluster_boost"] * (lexical_norm.get(code, 0.0) / max_lex)
                combined_scores[code] += boost
                class_boost_applied[code] = class_boost_applied.get(code, 0.0) + boost

    base_sorted = sorted(pool_codes, key=lambda code: base_scores.get(code, 0.0), reverse=True)
    if base_sorted:
        top_base = base_sorted[0]
        parent = graph.parent_lookup.get(top_base, "")
        if parent:
            sibling_cluster = [
                code
                for code in refinement_codes
                if graph.parent_lookup.get(code, "") == parent
            ]
            if len(sibling_cluster) >= 2:
                lexical_winner = max(sibling_cluster, key=lambda code: lexical_scores.get(code, 0.0))
                if lexical_winner != top_base:
                    margin = lexical_norm.get(lexical_winner, 0.0) - lexical_norm.get(top_base, 0.0)
                    if margin >= 0.07:
                        combined_scores[lexical_winner] = max(
                            combined_scores.get(lexical_winner, 0.0),
                            combined_scores.get(top_base, 0.0) + margin * weights["sibling_boost"],
                        )
        top_window = base_sorted[: min(CPV_REFINEMENT_BASE_WINDOW, len(base_sorted))]
        if len(top_window) >= 2:
            lexical_winner = max(top_window, key=lambda code: lexical_scores.get(code, 0.0))
            if lexical_winner != top_base:
                margin = lexical_norm.get(lexical_winner, 0.0) - lexical_norm.get(top_base, 0.0)
                if margin >= 0.14:
                    combined_scores[lexical_winner] = max(
                        combined_scores.get(lexical_winner, 0.0),
                        combined_scores.get(top_base, 0.0) + margin * weights["lexical"],
                    )

    refined_sorted = sorted(active_codes, key=lambda code: combined_scores.get(code, 0.0), reverse=True)
    dropped_refinement = sorted(
        [code for code in refinement_codes if code not in set(active_codes)],
        key=lambda code: combined_scores.get(code, 0.0),
        reverse=True,
    )
    tail = [code for code in base_sorted if code not in refinement_codes]
    ranked_codes = list(dict.fromkeys(refined_sorted + dropped_refinement + tail))

    selected: List[Dict[str, object]] = []
    for code in ranked_codes[: min(top_k, len(ranked_codes))]:
        row = rows_by_code.get(code)
        if row is None:
            continue
        row["score"] = float(combined_scores.get(code, 0.0))
        row["hierarchy_selection_score"] = float(combined_scores.get(code, 0.0))
        row["hierarchy_lexical_score"] = float(lexical_scores.get(code, 0.0))
        row["hierarchy_sibling_boost"] = float(sibling_boost_applied.get(code, 0.0))
        row["hierarchy_class_boost"] = float(class_boost_applied.get(code, 0.0))
        row["hierarchy_stage_division_selected"] = 1 if code[:2] in selected_division_prefixes else 0
        row["hierarchy_stage_group_selected"] = 1 if code[:4] in selected_group_prefixes else 0
        row["hierarchy_stage_class_selected"] = 1 if code[:6] in selected_class_prefixes else 0
        row["retriever"] = str(row.get("retriever") or "hierarchy_selection")
        selected.append(row)

    return selected, {
        "selection_enabled": True,
        "pool_size": len(pool_codes),
        "refinement_size": len(active_codes),
        "selected_division_prefixes": selected_division_prefixes,
        "selected_group_prefixes": selected_group_prefixes,
        "selected_class_prefixes": selected_class_prefixes,
        "sibling_groups_applied": sum(1 for siblings in parent_groups.values() if len(siblings) >= 2),
        "class_clusters_applied": sum(
            1 for prefix_len in (6, 4) for members in _group_codes_by_prefix(list(active_codes), prefix_len).values() if len(members) >= 2
        ),
    }


def expand_and_select_cpv(
    *,
    query: str,
    base_rows: Sequence[Dict[str, object]],
    chunks_by_code: Dict[str, Dict[str, object]],
    graph: CPVKnowledgeGraph,
    top_k: int,
    kg_enabled: bool,
    kg_profile: str = "selection",
    graph_algorithm: str | None = None,
    graph_weight: float = 0.35,
    max_seed_codes: int | None = None,
    max_graph_candidates: int | None = None,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    del graph_weight  # kept for API compatibility; hierarchy_select uses CPV_SELECTION_WEIGHTS
    profile = kg_profile if kg_enabled else "balanced"
    if kg_enabled:
        pool_rows, kg_meta = graph_expand_cpv_pool(
            query=query,
            base_rows=base_rows,
            chunks_by_code=chunks_by_code,
            graph=graph,
            kg_profile=profile,
            graph_algorithm=graph_algorithm,
            max_seed_codes=max_seed_codes,
            max_graph_candidates=max_graph_candidates,
        )
    else:
        pool_rows = [dict(row) for row in base_rows]
        base_codes = [
            str(row.get("cpv_code") or row.get("chunk_id") or "").strip()
            for row in base_rows
            if str(row.get("cpv_code") or row.get("chunk_id") or "").strip()
        ]
        for row in pool_rows:
            row["base_retrieval_score"] = float(row.get("score") or 0.0)
            row.setdefault("kg_graph_score", 0.0)
        kg_meta = {
            "enabled": False,
            "base_codes": base_codes,
            "candidate_pool_codes": base_codes,
            "added_codes": [],
            "paths": {},
        }

    selected, selection_meta = hierarchy_select_cpv_candidates(
        query=query,
        pool_rows=pool_rows,
        graph=graph,
        chunks_by_code=chunks_by_code,
        top_k=top_k,
    )
    final_codes = [str(row.get("cpv_code") or "") for row in selected]
    base_codes = [str(code) for code in kg_meta.get("base_codes", [])]
    kg_meta["added_codes"] = [code for code in final_codes if code not in set(base_codes[:top_k])]
    kg_meta["paths"] = {
        str(row.get("cpv_code") or ""): str(row.get("kg_path") or "")
        for row in selected
        if str(row.get("kg_path") or "").strip()
    }
    kg_meta.update(selection_meta)
    return selected, kg_meta


def graph_expand_and_rerank_cpv(
    *,
    query: str,
    base_rows: Sequence[Dict[str, object]],
    chunks_by_code: Dict[str, Dict[str, object]],
    graph: CPVKnowledgeGraph,
    top_k: int,
    graph_weight: float = 0.35,
    kg_profile: str = "balanced",
    graph_algorithm: str | None = None,
    max_seed_codes: int | None = None,
    max_graph_candidates: int | None = None,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    (
        candidate_codes,
        base_norm,
        graph_norm,
        proximity_scores,
        candidate_reasons,
        candidate_paths,
        meta,
    ) = _expand_cpv_candidate_pool(
        query=query,
        base_rows=base_rows,
        chunks_by_code=chunks_by_code,
        graph=graph,
        kg_profile=kg_profile,
        graph_algorithm=graph_algorithm,
        max_seed_codes=max_seed_codes,
        max_graph_candidates=max_graph_candidates,
    )
    base_codes = [
        str(row.get("cpv_code") or row.get("chunk_id") or "").strip()
        for row in base_rows
        if str(row.get("cpv_code") or row.get("chunk_id") or "").strip()
    ]
    if top_k <= 0:
        return [], {
            "enabled": True,
            "base_codes": [],
            "candidate_pool_codes": [],
            "added_codes": [],
            "paths": {},
            **meta,
        }
    if not candidate_codes:
        return list(base_rows[:top_k]), {
            "enabled": True,
            "base_codes": base_codes,
            "candidate_pool_codes": base_codes,
            "added_codes": [],
            "paths": {},
            **meta,
        }

    lexical_scores: Dict[str, float] = {}
    for code in candidate_codes:
        lexical_scores[code] = lexical_overlap_score(
            query,
            _candidate_rich_text(code, graph=graph, chunks_by_code=chunks_by_code),
        )
    lexical_norm = dict(zip(candidate_codes, min_max_normalize([lexical_scores[code] for code in candidate_codes])))

    reranked: List[Dict[str, object]] = []
    for code in candidate_codes:
        chunk = chunks_by_code.get(code)
        if chunk is None:
            continue
        base_score = base_norm.get(code, 0.0)
        graph_score = graph_norm.get(code, 0.0)
        final_score = (1.0 - graph_weight) * base_score + graph_weight * graph_score
        row = dict(chunk)
        row["score"] = float(final_score)
        row["base_retrieval_score"] = float(base_score)
        row["kg_graph_score"] = float(graph_score)
        row["kg_path_score"] = float(lexical_norm.get(code, 0.0))
        row["retriever"] = "cpv_kg_augmented" if code not in base_norm else str(
            next((base_row.get("retriever", "") for base_row in base_rows if base_row.get("cpv_code") == code), "")
        )
        row["retrieval_source"] = (
            "vector+graph" if code in base_norm and graph_score > 0 else "graph" if code not in base_norm else "vector"
        )
        row["kg_candidate_reason"] = candidate_reasons.get(code, "")
        row["kg_path"] = candidate_paths.get(code, "")
        reranked.append(row)

    reranked.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    final_rows = reranked[: min(top_k, len(reranked))]
    final_codes = [str(row.get("cpv_code") or "") for row in final_rows]
    added_codes = [code for code in final_codes if code not in set(base_codes[:top_k])]
    return final_rows, {
        "enabled": True,
        "base_codes": base_codes[:top_k],
        "candidate_pool_codes": candidate_codes,
        "added_codes": added_codes,
        "paths": {code: candidate_paths.get(code, "") for code in final_codes if candidate_paths.get(code)},
        "pool_rows": reranked,
        **meta,
    }


def _is_direct_sibling(code_a: str, code_b: str, graph: CPVKnowledgeGraph) -> bool:
    if code_a == code_b:
        return False
    parent_a = graph.parent_lookup.get(code_a, "")
    parent_b = graph.parent_lookup.get(code_b, "")
    return bool(parent_a and parent_a == parent_b)


def _is_strict_ancestor(candidate: str, reference: str, graph: CPVKnowledgeGraph) -> bool:
    return candidate in set(
        ancestor_codes(reference, graph.parent_lookup, include_self=False)
    )


def post_refine_cpv_ranking(
    *,
    query: str,
    ranked_rows: Sequence[Dict[str, object]],
    pool_rows: Sequence[Dict[str, object]],
    graph: CPVKnowledgeGraph,
    chunks_by_code: Dict[str, Dict[str, object]],
    top_k: int,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    del query
    del pool_rows
    if top_k <= 0 or not ranked_rows:
        return [], {"post_refine_applied": False}

    current = [dict(row) for row in ranked_rows[:top_k]]
    ranked_codes = [str(row.get("cpv_code") or row.get("chunk_id") or "").strip() for row in current if str(row.get("cpv_code") or row.get("chunk_id") or "").strip()]
    if len(ranked_codes) < 2:
        return current, {"post_refine_applied": False}

    rows_by_code = {code: dict(row) for code, row in zip(ranked_codes, current)}
    branch_support: Dict[str, float] = {}
    branch_neighbor_counts: Dict[str, int] = {}
    for code in ranked_codes:
        row = rows_by_code[code]
        neighbors: List[float] = []
        for other in ranked_codes:
            if other == code:
                continue
            if (
                _is_direct_sibling(code, other, graph)
                or cpv_common_prefix_length(code, other) >= 4
                or _is_strict_ancestor(code, other, graph)
                or _is_strict_ancestor(other, code, graph)
            ):
                neighbors.append(float(rows_by_code[other].get("learned_label_score", rows_by_code[other].get("score", 0.0)) or 0.0))
        branch_neighbor_counts[code] = len(neighbors)
        branch_support[code] = sum(neighbors) / len(neighbors) if neighbors else 0.0
        learned_score = float(row.get("learned_label_score", row.get("score", 0.0)) or 0.0)
        row["post_refine_branch_support"] = branch_support[code]
        row["post_refine_neighbor_count"] = branch_neighbor_counts[code]
        row["post_refine_bonus"] = 0.10 * branch_support[code] + min(0.03, 0.01 * branch_neighbor_counts[code])
        row["post_refine_score"] = learned_score + float(row.get("post_refine_bonus") or 0.0)
        row["post_refine_promoted"] = False
        row["post_refine_reason"] = ""
        row["post_refine_lexical_score"] = 0.0
        rows_by_code[code] = row

    refined = sorted(
        [rows_by_code[code] for code in ranked_codes],
        key=lambda row: float(row.get("post_refine_score", row.get("score", 0.0)) or 0.0),
        reverse=True,
    )
    original_top = ranked_codes[0]
    refined_top = str(refined[0].get("cpv_code") or refined[0].get("chunk_id") or "").strip()
    if refined_top and refined_top != original_top:
        refined[0]["post_refine_promoted"] = True
        refined[0]["post_refine_reason"] = "branch_graph_support"
    for row in refined:
        row["score"] = float(row.get("post_refine_score", row.get("score", 0.0)) or 0.0)

    return refined, {
        "post_refine_applied": True,
        "post_refine_reason": "branch_graph_support",
        "post_refine_promoted_code": refined_top,
    }


def cpv_kg_metrics(
    *,
    gold_code: str,
    base_codes: Sequence[str],
    final_codes: Sequence[str],
    candidate_pool_codes: Sequence[str],
    added_codes: Sequence[str],
    paths: Dict[str, str],
    top_k: int,
) -> Dict[str, object]:
    gold = str(gold_code).strip()
    base_top = [str(code).strip() for code in base_codes[:top_k] if str(code).strip()]
    final_top = [str(code).strip() for code in final_codes[:top_k] if str(code).strip()]
    pool = [str(code).strip() for code in candidate_pool_codes if str(code).strip()]
    added = [str(code).strip() for code in added_codes if str(code).strip()]

    base_hit = gold in base_top
    final_hit = gold in final_top
    pool_hit = gold in pool
    top1_exact = bool(final_top and final_top[0] == gold)
    base_top1_match = best_cpv_hierarchy_match(base_top[0], [gold]) if base_top else None
    base_top1_close_wrong = bool(
        base_top and base_top[0] != gold and base_top1_match and float(base_top1_match["score"]) >= 0.25
    )
    best_final_match = max(
        [
            float(match["score"])
            for code in final_top
            for match in [best_cpv_hierarchy_match(code, [gold])]
            if match is not None
        ]
        or [0.0]
    )
    added_noise = [
        code
        for code in added
        for match in [best_cpv_hierarchy_match(code, [gold])]
        if match is None or float(match["score"]) <= 0.0
    ]
    path_values = [path for code, path in paths.items() if code in final_top and path]
    ppr_paths = [path for path in path_values if "ppr_" in path]
    sibling_paths = [path for path in path_values if "sibling" in path]
    ancestor_child_paths = [path for path in path_values if "ancestor" in path or "child" in path]
    added_useful = [
        code
        for code in added
        for match in [best_cpv_hierarchy_match(code, [gold])]
        if match is not None and float(match["score"]) > 0.0
    ]
    branch_competition_codes = [
        code
        for code in final_top
        for match in [best_cpv_hierarchy_match(code, [gold])]
        if match is not None and float(match["score"]) >= 0.25
    ]
    sibling_disambiguation_success = bool(
        top1_exact
        and len(branch_competition_codes) >= 2
    )
    return {
        "kg_enabled": True,
        "kg_candidate_pool_size": len(pool),
        "kg_added_candidate_count": len(added),
        "kg_expansion_gold_added": bool(not base_hit and final_hit),
        "kg_expansion_gold_available": bool(not base_hit and pool_hit),
        "kg_expansion_noise_rate": (len(added_noise) / len(added)) if added else 0.0,
        "kg_useful_added_candidate_count": len(added_useful),
        "kg_strict_gold_delta": 1.0 if (not base_hit and final_hit) else 0.0,
        "branch_recall_at_k": bool(best_final_match >= 0.25),
        "class_recall_at_k": bool(best_final_match >= 0.75),
        "sibling_disambiguation_success": sibling_disambiguation_success,
        "oracle_rerank_ceiling": bool(pool_hit),
        "path_explanation_coverage": 1.0 if (final_top and paths.get(final_top[0])) else 0.0,
        "kg_path_coverage_at_k": len(path_values) / len(final_top) if final_top else 0.0,
        "kg_ppr_path_share": len(ppr_paths) / len(path_values) if path_values else 0.0,
        "kg_sibling_path_share": len(sibling_paths) / len(path_values) if path_values else 0.0,
        "kg_hierarchy_path_share": len(ancestor_child_paths) / len(path_values) if path_values else 0.0,
        "kg_oracle_pool_gap": 1.0 if (pool_hit and not final_hit) else 0.0,
        "kg_candidate_added_codes": added,
        "kg_candidate_pool_codes": pool,
        "kg_top1_path": paths.get(final_top[0], "") if final_top else "",
    }
