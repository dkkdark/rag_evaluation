#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse
import urllib.error
import urllib.request

from rag_eval.classifiers.cpv_baseline import load_cpv_catalog_from_db
from rag_eval.classifiers.cpv_kg import build_cpv_knowledge_graph, code_level
from rag_eval.data.ted_notice_store import load_cpv_profiles


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload the CPV knowledge graph for a completed TED/CPV run directly into Neo4j.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to an existing TED/CPV run directory containing run_summary.json.",
    )
    parser.add_argument(
        "--db-path",
        default="",
        help="Optional override for ted_notices.sqlite. Defaults to <index_dir>/ted_notices.sqlite from run_summary.",
    )
    parser.add_argument(
        "--neo4j-uri",
        required=True,
        help="Neo4j base URI, e.g. http://localhost:7474",
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username. Default: neo4j",
    )
    parser.add_argument(
        "--neo4j-password-env",
        default="NEO4J_PASSWORD",
        help="Environment variable containing the Neo4j password. Default: NEO4J_PASSWORD",
    )
    parser.add_argument(
        "--neo4j-database",
        default="neo4j",
        help="Neo4j database name. Default: neo4j",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for node and relationship uploads.",
    )
    parser.add_argument(
        "--clear-existing-cpv-graph",
        action="store_true",
        help="Delete existing :CPVCode nodes before uploading the new graph.",
    )
    parser.add_argument(
        "--clear-existing-run-evidence",
        action="store_true",
        help="Delete existing run-specific evidence nodes for this run before uploading them again.",
    )
    return parser


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _extract_first_cpv_code(value: object) -> str:
    match = re.search(r"\b\d{8}\b", str(value or ""))
    return match.group(0) if match else ""


def _build_run_evidence_payload(
    *,
    run_dir: Path,
    run_summary: dict,
    ted_notice_db_path: Path,
) -> dict[str, object]:
    rag_results_path = run_dir / "rag_results.csv"
    retrieved_chunks_path = run_dir / "retrieved_chunks.csv"
    diagnostics_path = run_dir / "diagnostics.csv"
    if not rag_results_path.exists() or not retrieved_chunks_path.exists():
        raise FileNotFoundError("rag_results.csv and retrieved_chunks.csv are required for run-specific evidence export.")

    rag_rows = _read_csv_rows(rag_results_path)
    retrieved_rows = _read_csv_rows(retrieved_chunks_path)
    diagnostics_rows = _read_csv_rows(diagnostics_path) if diagnostics_path.exists() else []
    diagnostics_by_qid = {
        str(row.get("question_id") or "").strip(): row
        for row in diagnostics_rows
        if str(row.get("question_id") or "").strip()
    }

    run_name = run_dir.name
    classifier_summary = run_summary.get("classifier_summary") or {}
    top_k = int(classifier_summary.get("top_k") or run_summary.get("top_k") or 0)

    query_nodes: list[dict[str, object]] = []
    prediction_nodes: list[dict[str, object]] = []
    query_retrieval_edges: list[dict[str, object]] = []
    prediction_edges: list[dict[str, object]] = []
    relevant_codes: set[str] = set()

    for row in rag_rows:
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            continue
        query_node_id = f"{run_name}::query::{question_id}"
        prediction_node_id = f"{run_name}::prediction::{question_id}"
        predicted_code = _extract_first_cpv_code(row.get("top1_predicted_cpv"))
        gold_code = _extract_first_cpv_code(row.get("expected_cpv_codes"))
        diagnostics = diagnostics_by_qid.get(question_id, {})

        query_nodes.append(
            {
                "id": query_node_id,
                "run_id": run_name,
                "question_id": question_id,
                "text": str(row.get("question") or ""),
                "retrieval_query": str(row.get("retrieval_query") or ""),
                "object_query": str(row.get("object_query") or ""),
                "auto_flag": str(row.get("auto_flag") or ""),
                "primary_error_reason": str(row.get("primary_error_reason") or diagnostics.get("primary_error_reason") or ""),
                "secondary_error_reason": str(row.get("secondary_error_reason") or diagnostics.get("secondary_error_reason") or ""),
            }
        )
        prediction_nodes.append(
            {
                "id": prediction_node_id,
                "run_id": run_name,
                "question_id": question_id,
                "predicted_code": predicted_code,
                "gold_code": gold_code,
                "exact_top1_match": str(row.get("exact_top1_match") or ""),
                "auto_flag": str(row.get("auto_flag") or ""),
                "score_margin_top1_top2": str(row.get("score_margin_top1_top2") or ""),
                "best_hierarchy_score_at_k": str(row.get("best_hierarchy_score_at_k") or ""),
                "gold_present_at_k": str(row.get("gold_present_at_k") or ""),
            }
        )
        prediction_edges.append(
            {
                "query_id": query_node_id,
                "prediction_id": prediction_node_id,
                "predicted_code": predicted_code,
                "gold_code": gold_code,
            }
        )
        if predicted_code:
            relevant_codes.add(predicted_code)
        if gold_code:
            relevant_codes.add(gold_code)

    for row in retrieved_rows:
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            continue
        query_node_id = f"{run_name}::query::{question_id}"
        code = _extract_first_cpv_code(row.get("cpv_code") or row.get("chunk_id") or row.get("title"))
        if not code:
            continue
        relevant_codes.add(code)
        query_retrieval_edges.append(
            {
                "query_id": query_node_id,
                "cpv_code": code,
                "rank": int(float(row.get("rank") or 0)),
                "score": float(row.get("score") or 0.0),
                "source_type": str(row.get("source_type") or ""),
                "retrieval_source": str(row.get("retrieval_source") or ""),
                "relevance_grade": str(row.get("relevance_grade") or ""),
                "title": str(row.get("title") or ""),
                "text": str(row.get("text") or "")[:2000],
            }
        )

    profiles = {
        str(profile.get("code") or "").strip(): profile
        for profile in load_cpv_profiles(str(ted_notice_db_path))
    }
    notice_example_nodes: list[dict[str, object]] = []
    notice_example_edges: list[dict[str, object]] = []
    for code in sorted(relevant_codes):
        profile = profiles.get(code) or {}
        examples = [str(item).strip() for item in profile.get("notice_examples", []) if str(item).strip()]
        for index, example_text in enumerate(examples[:3], start=1):
            example_id = f"{code}::notice_example::{index}"
            notice_example_nodes.append(
                {
                    "id": example_id,
                    "cpv_code": code,
                    "example_index": index,
                    "text": example_text,
                    "run_id": run_name,
                }
            )
            notice_example_edges.append(
                {
                    "cpv_code": code,
                    "example_id": example_id,
                    "example_index": index,
                }
            )

    return {
        "run_node": {
            "id": run_name,
            "run_dir": str(run_dir),
            "classifier_type": str(run_summary.get("classifier_type") or ""),
            "top_k": top_k,
            "retriever": str(classifier_summary.get("retriever") or ""),
            "kg_enabled": bool((run_summary.get("kg") or {}).get("enabled")),
        },
        "query_nodes": query_nodes,
        "prediction_nodes": prediction_nodes,
        "query_retrieval_edges": query_retrieval_edges,
        "prediction_edges": prediction_edges,
        "notice_example_nodes": notice_example_nodes,
        "notice_example_edges": notice_example_edges,
        "relevant_codes": sorted(relevant_codes),
    }


def _resolve_ted_notice_db_path(run_summary: dict, override: str) -> str:
    if override:
        return override
    classifier_summary = run_summary.get("classifier_summary") or {}
    search_backend = classifier_summary.get("search_backend") or {}
    index_dir = str(search_backend.get("index_dir") or ".rag_eval_indices").strip()
    return os.path.join(index_dir, "ted_notices.sqlite")


def _neo4j_tx_url(base_uri: str, database: str) -> str:
    return base_uri.rstrip("/") + f"/db/{database}/tx/commit"


def _neo4j_headers(user: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _post_neo4j(base_uri: str, database: str, headers: dict[str, str], statement: str, parameters: dict) -> dict:
    payload = {
        "statements": [
            {
                "statement": statement,
                "parameters": parameters,
            }
        ]
    }
    request = urllib.request.Request(
        _neo4j_tx_url(base_uri, database),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Neo4j HTTP error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach Neo4j at {base_uri}: {exc}") from exc
    data = json.loads(body)
    errors = data.get("errors") or []
    if errors:
        raise RuntimeError(f"Neo4j returned errors: {errors}")
    return data


def _batched(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _upload_graph_to_neo4j(
    *,
    graph,
    evidence_payload: dict[str, object],
    base_uri: str,
    database: str,
    user: str,
    password: str,
    batch_size: int,
    clear_existing: bool,
    clear_existing_run_evidence: bool,
) -> dict[str, object]:
    headers = _neo4j_headers(user, password)

    _post_neo4j(
        base_uri,
        database,
        headers,
        """
        CREATE CONSTRAINT cpv_code_unique IF NOT EXISTS
        FOR (n:CPVCode)
        REQUIRE n.code IS UNIQUE
        """,
        {},
    )
    _post_neo4j(
        base_uri,
        database,
        headers,
        """
        CREATE CONSTRAINT cpv_run_unique IF NOT EXISTS
        FOR (n:CPVRun)
        REQUIRE n.id IS UNIQUE
        """,
        {},
    )
    _post_neo4j(
        base_uri,
        database,
        headers,
        """
        CREATE CONSTRAINT cpv_query_unique IF NOT EXISTS
        FOR (n:CPVQuery)
        REQUIRE n.id IS UNIQUE
        """,
        {},
    )
    _post_neo4j(
        base_uri,
        database,
        headers,
        """
        CREATE CONSTRAINT cpv_prediction_unique IF NOT EXISTS
        FOR (n:CPVPrediction)
        REQUIRE n.id IS UNIQUE
        """,
        {},
    )
    _post_neo4j(
        base_uri,
        database,
        headers,
        """
        CREATE CONSTRAINT cpv_notice_example_unique IF NOT EXISTS
        FOR (n:NoticeExample)
        REQUIRE n.id IS UNIQUE
        """,
        {},
    )

    if clear_existing:
        _post_neo4j(
            base_uri,
            database,
            headers,
            "MATCH (n:CPVCode) DETACH DELETE n",
            {},
        )
    if clear_existing_run_evidence:
        _post_neo4j(
            base_uri,
            database,
            headers,
            "MATCH (n) WHERE n.run_id = $run_id OR (n:CPVRun AND n.id = $run_id) DETACH DELETE n",
            {"run_id": str(evidence_payload.get("run_node", {}).get("id") or "")},
        )

    nodes = []
    for code in sorted(graph.nodes.keys()):
        node = graph.nodes[code]
        nodes.append(
            {
                "code": node.code,
                "label": node.label,
                "description": node.description,
                "parent_code": node.parent_code,
                "code_level": code_level(node.code),
                "path_codes": node.path_codes,
                "path_labels": node.path_labels,
                "examples": node.examples[:12],
            }
        )

    node_statement = """
    UNWIND $rows AS row
    MERGE (n:CPVCode {code: row.code})
    SET n.label = row.label,
        n.description = row.description,
        n.parent_code = row.parent_code,
        n.code_level = row.code_level,
        n.path_codes = row.path_codes,
        n.path_labels = row.path_labels,
        n.examples = row.examples
    """
    for batch in _batched(nodes, batch_size):
        _post_neo4j(base_uri, database, headers, node_statement, {"rows": batch})

    child_edges = []
    parent_edges = []
    sibling_edges = []
    for child, parent in sorted(graph.parent_lookup.items()):
        if child and parent:
            child_edges.append({"source": child, "target": parent})
            parent_edges.append({"source": parent, "target": child})
    for code, siblings in sorted(graph.sibling_lookup.items()):
        for sibling in siblings:
            if code < sibling:
                sibling_edges.append({"source": code, "target": sibling})

    rel_templates = [
        (
            child_edges,
            "CHILD_OF",
            """
            UNWIND $rows AS row
            MATCH (a:CPVCode {code: row.source})
            MATCH (b:CPVCode {code: row.target})
            MERGE (a)-[:CHILD_OF]->(b)
            """,
        ),
        (
            parent_edges,
            "PARENT_OF",
            """
            UNWIND $rows AS row
            MATCH (a:CPVCode {code: row.source})
            MATCH (b:CPVCode {code: row.target})
            MERGE (a)-[:PARENT_OF]->(b)
            """,
        ),
        (
            sibling_edges,
            "SIBLING_OF",
            """
            UNWIND $rows AS row
            MATCH (a:CPVCode {code: row.source})
            MATCH (b:CPVCode {code: row.target})
            MERGE (a)-[:SIBLING_OF]->(b)
            MERGE (b)-[:SIBLING_OF]->(a)
            """,
        ),
    ]

    relationship_counts = {}
    for rows, rel_name, statement in rel_templates:
        for batch in _batched(rows, batch_size):
            _post_neo4j(base_uri, database, headers, statement, {"rows": batch})
        relationship_counts[rel_name] = len(rows)

    run_node = dict(evidence_payload.get("run_node") or {})
    _post_neo4j(
        base_uri,
        database,
        headers,
        """
        MERGE (r:CPVRun {id: $row.id})
        SET r.run_dir = $row.run_dir,
            r.classifier_type = $row.classifier_type,
            r.top_k = $row.top_k,
            r.retriever = $row.retriever,
            r.kg_enabled = $row.kg_enabled
        """,
        {"row": run_node},
    )
    for batch in _batched(list(evidence_payload.get("query_nodes") or []), batch_size):
        _post_neo4j(
            base_uri,
            database,
            headers,
            """
            UNWIND $rows AS row
            MERGE (q:CPVQuery {id: row.id})
            SET q.run_id = row.run_id,
                q.question_id = row.question_id,
                q.text = row.text,
                q.retrieval_query = row.retrieval_query,
                q.object_query = row.object_query,
                q.auto_flag = row.auto_flag,
                q.primary_error_reason = row.primary_error_reason,
                q.secondary_error_reason = row.secondary_error_reason
            WITH q, row
            MATCH (r:CPVRun {id: row.run_id})
            MERGE (r)-[:HAS_QUERY]->(q)
            """,
            {"rows": batch},
        )
    for batch in _batched(list(evidence_payload.get("prediction_nodes") or []), batch_size):
        _post_neo4j(
            base_uri,
            database,
            headers,
            """
            UNWIND $rows AS row
            MERGE (p:CPVPrediction {id: row.id})
            SET p.run_id = row.run_id,
                p.question_id = row.question_id,
                p.predicted_code = row.predicted_code,
                p.gold_code = row.gold_code,
                p.exact_top1_match = row.exact_top1_match,
                p.auto_flag = row.auto_flag,
                p.score_margin_top1_top2 = row.score_margin_top1_top2,
                p.best_hierarchy_score_at_k = row.best_hierarchy_score_at_k,
                p.gold_present_at_k = row.gold_present_at_k
            """,
            {"rows": batch},
        )
    for batch in _batched(list(evidence_payload.get("prediction_edges") or []), batch_size):
        _post_neo4j(
            base_uri,
            database,
            headers,
            """
            UNWIND $rows AS row
            MATCH (q:CPVQuery {id: row.query_id})
            MATCH (p:CPVPrediction {id: row.prediction_id})
            MERGE (q)-[:HAS_PREDICTION]->(p)
            WITH row, p
            FOREACH (_ IN CASE WHEN row.predicted_code <> '' THEN [1] ELSE [] END |
              MERGE (c:CPVCode {code: row.predicted_code})
              MERGE (p)-[:PREDICTED_CODE]->(c)
            )
            FOREACH (_ IN CASE WHEN row.gold_code <> '' THEN [1] ELSE [] END |
              MERGE (g:CPVCode {code: row.gold_code})
              MERGE (p)-[:GOLD_CODE]->(g)
            )
            """,
            {"rows": batch},
        )
    for batch in _batched(list(evidence_payload.get("query_retrieval_edges") or []), batch_size):
        _post_neo4j(
            base_uri,
            database,
            headers,
            """
            UNWIND $rows AS row
            MATCH (q:CPVQuery {id: row.query_id})
            MATCH (c:CPVCode {code: row.cpv_code})
            MERGE (q)-[rel:RETRIEVED]->(c)
            SET rel.rank = row.rank,
                rel.score = row.score,
                rel.source_type = row.source_type,
                rel.retrieval_source = row.retrieval_source,
                rel.relevance_grade = row.relevance_grade,
                rel.title = row.title,
                rel.text = row.text
            """,
            {"rows": batch},
        )
    for batch in _batched(list(evidence_payload.get("notice_example_nodes") or []), batch_size):
        _post_neo4j(
            base_uri,
            database,
            headers,
            """
            UNWIND $rows AS row
            MERGE (n:NoticeExample {id: row.id})
            SET n.run_id = row.run_id,
                n.cpv_code = row.cpv_code,
                n.example_index = row.example_index,
                n.text = row.text
            """,
            {"rows": batch},
        )
    for batch in _batched(list(evidence_payload.get("notice_example_edges") or []), batch_size):
        _post_neo4j(
            base_uri,
            database,
            headers,
            """
            UNWIND $rows AS row
            MATCH (c:CPVCode {code: row.cpv_code})
            MATCH (n:NoticeExample {id: row.example_id})
            MERGE (c)-[:HAS_NOTICE_EXAMPLE {example_index: row.example_index}]->(n)
            """,
            {"rows": batch},
        )

    return {
        "enabled": True,
        "transport": "http",
        "neo4j_uri": base_uri,
        "neo4j_database": database,
        "n_nodes": len(nodes),
        "relationship_counts": relationship_counts,
        "n_edges_total": sum(relationship_counts.values()),
        "cleared_existing": clear_existing,
        "run_evidence": {
            "run_id": run_node.get("id"),
            "n_queries": len(evidence_payload.get("query_nodes") or []),
            "n_predictions": len(evidence_payload.get("prediction_nodes") or []),
            "n_retrieval_edges": len(evidence_payload.get("query_retrieval_edges") or []),
            "n_notice_examples": len(evidence_payload.get("notice_example_nodes") or []),
            "cleared_existing_run_evidence": clear_existing_run_evidence,
        },
    }


def _upload_graph_to_neo4j_bolt(
    *,
    graph,
    evidence_payload: dict[str, object],
    uri: str,
    database: str,
    user: str,
    password: str,
    batch_size: int,
    clear_existing: bool,
    clear_existing_run_evidence: bool,
) -> dict[str, object]:
    try:
        from neo4j import GraphDatabase
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The neo4j Python driver is required for Aura/cloud connections. "
            "Install it with `pip install neo4j` or add it to your environment first."
        ) from exc

    driver = GraphDatabase.driver(uri, auth=(user, password))

    nodes = []
    for code in sorted(graph.nodes.keys()):
        node = graph.nodes[code]
        nodes.append(
            {
                "code": node.code,
                "label": node.label,
                "description": node.description,
                "parent_code": node.parent_code,
                "code_level": code_level(node.code),
                "path_codes": node.path_codes,
                "path_labels": node.path_labels,
                "examples": node.examples[:12],
            }
        )

    child_edges = []
    parent_edges = []
    sibling_edges = []
    for child, parent in sorted(graph.parent_lookup.items()):
        if child and parent:
            child_edges.append({"source": child, "target": parent})
            parent_edges.append({"source": parent, "target": child})
    for code, siblings in sorted(graph.sibling_lookup.items()):
        for sibling in siblings:
            if code < sibling:
                sibling_edges.append({"source": code, "target": sibling})

    with driver.session(database=database) as session:
        session.run(
            """
            CREATE CONSTRAINT cpv_code_unique IF NOT EXISTS
            FOR (n:CPVCode)
            REQUIRE n.code IS UNIQUE
            """
        ).consume()
        session.run(
            """
            CREATE CONSTRAINT cpv_run_unique IF NOT EXISTS
            FOR (n:CPVRun)
            REQUIRE n.id IS UNIQUE
            """
        ).consume()
        session.run(
            """
            CREATE CONSTRAINT cpv_query_unique IF NOT EXISTS
            FOR (n:CPVQuery)
            REQUIRE n.id IS UNIQUE
            """
        ).consume()
        session.run(
            """
            CREATE CONSTRAINT cpv_prediction_unique IF NOT EXISTS
            FOR (n:CPVPrediction)
            REQUIRE n.id IS UNIQUE
            """
        ).consume()
        session.run(
            """
            CREATE CONSTRAINT cpv_notice_example_unique IF NOT EXISTS
            FOR (n:NoticeExample)
            REQUIRE n.id IS UNIQUE
            """
        ).consume()

        if clear_existing:
            session.run("MATCH (n:CPVCode) DETACH DELETE n").consume()
        if clear_existing_run_evidence:
            session.run(
                "MATCH (n) WHERE n.run_id = $run_id OR (n:CPVRun AND n.id = $run_id) DETACH DELETE n",
                run_id=str(evidence_payload.get("run_node", {}).get("id") or ""),
            ).consume()

        node_statement = """
        UNWIND $rows AS row
        MERGE (n:CPVCode {code: row.code})
        SET n.label = row.label,
            n.description = row.description,
            n.parent_code = row.parent_code,
            n.code_level = row.code_level,
            n.path_codes = row.path_codes,
            n.path_labels = row.path_labels,
            n.examples = row.examples
        """
        for batch in _batched(nodes, batch_size):
            session.run(node_statement, rows=batch).consume()

        rel_statements = [
            (
                child_edges,
                "CHILD_OF",
                """
                UNWIND $rows AS row
                MATCH (a:CPVCode {code: row.source})
                MATCH (b:CPVCode {code: row.target})
                MERGE (a)-[:CHILD_OF]->(b)
                """,
            ),
            (
                parent_edges,
                "PARENT_OF",
                """
                UNWIND $rows AS row
                MATCH (a:CPVCode {code: row.source})
                MATCH (b:CPVCode {code: row.target})
                MERGE (a)-[:PARENT_OF]->(b)
                """,
            ),
            (
                sibling_edges,
                "SIBLING_OF",
                """
                UNWIND $rows AS row
                MATCH (a:CPVCode {code: row.source})
                MATCH (b:CPVCode {code: row.target})
                MERGE (a)-[:SIBLING_OF]->(b)
                MERGE (b)-[:SIBLING_OF]->(a)
                """,
            ),
        ]
        relationship_counts = {}
        for rows, rel_name, statement in rel_statements:
            for batch in _batched(rows, batch_size):
                session.run(statement, rows=batch).consume()
            relationship_counts[rel_name] = len(rows)

        run_node = dict(evidence_payload.get("run_node") or {})
        session.run(
            """
            MERGE (r:CPVRun {id: $row.id})
            SET r.run_dir = $row.run_dir,
                r.classifier_type = $row.classifier_type,
                r.top_k = $row.top_k,
                r.retriever = $row.retriever,
                r.kg_enabled = $row.kg_enabled
            """,
            row=run_node,
        ).consume()
        for batch in _batched(list(evidence_payload.get("query_nodes") or []), batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MERGE (q:CPVQuery {id: row.id})
                SET q.run_id = row.run_id,
                    q.question_id = row.question_id,
                    q.text = row.text,
                    q.retrieval_query = row.retrieval_query,
                    q.object_query = row.object_query,
                    q.auto_flag = row.auto_flag,
                    q.primary_error_reason = row.primary_error_reason,
                    q.secondary_error_reason = row.secondary_error_reason
                WITH q, row
                MATCH (r:CPVRun {id: row.run_id})
                MERGE (r)-[:HAS_QUERY]->(q)
                """,
                rows=batch,
            ).consume()
        for batch in _batched(list(evidence_payload.get("prediction_nodes") or []), batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MERGE (p:CPVPrediction {id: row.id})
                SET p.run_id = row.run_id,
                    p.question_id = row.question_id,
                    p.predicted_code = row.predicted_code,
                    p.gold_code = row.gold_code,
                    p.exact_top1_match = row.exact_top1_match,
                    p.auto_flag = row.auto_flag,
                    p.score_margin_top1_top2 = row.score_margin_top1_top2,
                    p.best_hierarchy_score_at_k = row.best_hierarchy_score_at_k,
                    p.gold_present_at_k = row.gold_present_at_k
                """,
                rows=batch,
            ).consume()
        for batch in _batched(list(evidence_payload.get("prediction_edges") or []), batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MATCH (q:CPVQuery {id: row.query_id})
                MATCH (p:CPVPrediction {id: row.prediction_id})
                MERGE (q)-[:HAS_PREDICTION]->(p)
                WITH row, p
                FOREACH (_ IN CASE WHEN row.predicted_code <> '' THEN [1] ELSE [] END |
                  MERGE (c:CPVCode {code: row.predicted_code})
                  MERGE (p)-[:PREDICTED_CODE]->(c)
                )
                FOREACH (_ IN CASE WHEN row.gold_code <> '' THEN [1] ELSE [] END |
                  MERGE (g:CPVCode {code: row.gold_code})
                  MERGE (p)-[:GOLD_CODE]->(g)
                )
                """,
                rows=batch,
            ).consume()
        for batch in _batched(list(evidence_payload.get("query_retrieval_edges") or []), batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MATCH (q:CPVQuery {id: row.query_id})
                MATCH (c:CPVCode {code: row.cpv_code})
                MERGE (q)-[rel:RETRIEVED]->(c)
                SET rel.rank = row.rank,
                    rel.score = row.score,
                    rel.source_type = row.source_type,
                    rel.retrieval_source = row.retrieval_source,
                    rel.relevance_grade = row.relevance_grade,
                    rel.title = row.title,
                    rel.text = row.text
                """,
                rows=batch,
            ).consume()
        for batch in _batched(list(evidence_payload.get("notice_example_nodes") or []), batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:NoticeExample {id: row.id})
                SET n.run_id = row.run_id,
                    n.cpv_code = row.cpv_code,
                    n.example_index = row.example_index,
                    n.text = row.text
                """,
                rows=batch,
            ).consume()
        for batch in _batched(list(evidence_payload.get("notice_example_edges") or []), batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MATCH (c:CPVCode {code: row.cpv_code})
                MATCH (n:NoticeExample {id: row.example_id})
                MERGE (c)-[:HAS_NOTICE_EXAMPLE {example_index: row.example_index}]->(n)
                """,
                rows=batch,
            ).consume()

    driver.close()
    return {
        "enabled": True,
        "transport": "bolt",
        "neo4j_uri": uri,
        "neo4j_database": database,
        "n_nodes": len(nodes),
        "relationship_counts": relationship_counts,
        "n_edges_total": sum(relationship_counts.values()),
        "cleared_existing": clear_existing,
        "run_evidence": {
            "run_id": run_node.get("id"),
            "n_queries": len(evidence_payload.get("query_nodes") or []),
            "n_predictions": len(evidence_payload.get("prediction_nodes") or []),
            "n_retrieval_edges": len(evidence_payload.get("query_retrieval_edges") or []),
            "n_notice_examples": len(evidence_payload.get("notice_example_nodes") or []),
            "cleared_existing_run_evidence": clear_existing_run_evidence,
        },
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_summary_path = run_dir / "run_summary.json"
    if not run_summary_path.exists():
        raise FileNotFoundError(f"run_summary.json not found in {run_dir}")

    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    if str(run_summary.get("classifier_type") or "") != "ted_cpv":
        raise ValueError("This exporter currently supports only classifier_type=ted_cpv runs.")

    ted_notice_db_path = Path(_resolve_ted_notice_db_path(run_summary, args.db_path)).expanduser()
    if not ted_notice_db_path.is_absolute():
        ted_notice_db_path = (Path.cwd() / ted_notice_db_path).resolve()
    if not ted_notice_db_path.exists():
        raise FileNotFoundError(f"TED notice DB not found: {ted_notice_db_path}")
    neo4j_password = os.environ.get(args.neo4j_password_env, "").strip()
    if not neo4j_password:
        raise ValueError(f"Environment variable {args.neo4j_password_env} is empty or not set.")

    catalog = load_cpv_catalog_from_db(str(ted_notice_db_path))
    if not catalog:
        raise ValueError(f"No CPV catalog records found in {ted_notice_db_path}")

    graph = build_cpv_knowledge_graph(catalog)
    evidence_payload = _build_run_evidence_payload(
        run_dir=run_dir,
        run_summary=run_summary,
        ted_notice_db_path=ted_notice_db_path,
    )
    parsed = urlparse(args.neo4j_uri)
    scheme = parsed.scheme.lower()
    if scheme in {"neo4j", "neo4j+s", "neo4j+ssc", "bolt", "bolt+s", "bolt+ssc"}:
        upload_summary = _upload_graph_to_neo4j_bolt(
            graph=graph,
            evidence_payload=evidence_payload,
            uri=args.neo4j_uri,
            database=args.neo4j_database,
            user=args.neo4j_user,
            password=neo4j_password,
            batch_size=max(1, int(args.batch_size)),
            clear_existing=bool(args.clear_existing_cpv_graph),
            clear_existing_run_evidence=bool(args.clear_existing_run_evidence),
        )
    else:
        upload_summary = _upload_graph_to_neo4j(
            graph=graph,
            evidence_payload=evidence_payload,
            base_uri=args.neo4j_uri,
            database=args.neo4j_database,
            user=args.neo4j_user,
            password=neo4j_password,
            batch_size=max(1, int(args.batch_size)),
            clear_existing=bool(args.clear_existing_cpv_graph),
            clear_existing_run_evidence=bool(args.clear_existing_run_evidence),
        )

    print(json.dumps(
        {
            "run_dir": str(run_dir),
            "ted_notice_db_path": str(ted_notice_db_path),
            **upload_summary,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
