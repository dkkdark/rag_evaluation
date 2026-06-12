from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Sequence


def local_db_backend_enabled(config: Dict[str, object] | None) -> bool:
    if not config:
        return False
    return str(config.get("backend") or "").strip().lower() == "sqlite"


def _sanitize_index_name(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9_-]+", "-", str(value or "").strip().lower()).strip("-_")
    return sanitized or "rag-eval-index"


def build_local_index_name(
    *,
    prefix: str,
    namespace: str,
    fingerprint_source: str,
    explicit_name: str | None = None,
) -> str:
    if explicit_name and str(explicit_name).strip():
        return _sanitize_index_name(str(explicit_name))
    digest = hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()[:12]
    return _sanitize_index_name(f"{prefix}-{namespace}-{digest}")


def chunk_fingerprint(chunks: Sequence[Dict[str, object]], *, fields: Sequence[str], model_name: str) -> str:
    digest = hashlib.sha1()
    digest.update(str(model_name).encode("utf-8"))
    for chunk in chunks:
        for field in fields:
            digest.update(str(chunk.get(field) or "").encode("utf-8"))
            digest.update(b"\x1f")
        digest.update(b"\x1e")
    return digest.hexdigest()


def _index_paths(config: Dict[str, object], index_name: str) -> Dict[str, str]:
    root = Path(str(config.get("index_dir") or ".rag_eval_indices")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    base = root / index_name
    return {
        "root": str(root),
        "sqlite_path": str(base.with_suffix(".sqlite")),
        "faiss_path": str(base.with_suffix(".faiss")),
        "manifest_path": str(base.with_suffix(".manifest.json")),
    }


def _strip_boost(field: str) -> str:
    return str(field or "").split("^", 1)[0].strip()


def _target_column(fields: Sequence[str]) -> str:
    cleaned = [_strip_boost(field) for field in fields if _strip_boost(field)]
    for preferred in ["search_text_en", "search_text_multilingual", "search_text", "description_en", "title", "text"]:
        if preferred in cleaned:
            return preferred
    return cleaned[0] if cleaned else "search_text"


def _fts_query(text: str) -> str:
    tokens = re.findall(r"[\w\u00C0-\u024F]+", str(text or "").casefold())
    if not tokens:
        return ""
    return " OR ".join(dict.fromkeys(tokens))


def _open_db(sqlite_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_local_index(
    *,
    config: Dict[str, object],
    index_name: str,
    documents: Sequence[Dict[str, object]],
    embeddings=None,
    manifest_payload: Dict[str, object],
) -> Dict[str, object]:
    paths = _index_paths(config, index_name)
    manifest_path = Path(paths["manifest_path"])
    sqlite_path = Path(paths["sqlite_path"])
    faiss_path = Path(paths["faiss_path"])

    requires_faiss = embeddings is not None
    if manifest_path.exists() and sqlite_path.exists() and (faiss_path.exists() or not requires_faiss):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("fingerprint") == manifest_payload.get("fingerprint")
                and int(manifest.get("count") or 0) == len(documents)
                and str(manifest.get("model_name") or "") == str(manifest_payload.get("model_name") or "")
                and bool(manifest.get("has_faiss")) == bool(requires_faiss)
            ):
                return {**paths, "index_name": index_name}
        except Exception:
            pass

    if sqlite_path.exists():
        sqlite_path.unlink()
    if faiss_path.exists():
        faiss_path.unlink()

    conn = _open_db(str(sqlite_path))
    try:
        conn.executescript(
            """
            CREATE TABLE docs (
              chunk_id TEXT PRIMARY KEY,
              vector_position INTEGER NOT NULL,
              payload_json TEXT NOT NULL,
              title TEXT,
              text TEXT,
              search_text TEXT,
              search_text_en TEXT,
              search_text_multilingual TEXT,
              description_en TEXT,
              generated_keywords_en TEXT,
              generated_synonyms_en TEXT,
              doc_id TEXT,
              section_id TEXT,
              program_id TEXT,
              program_name TEXT,
              cpv_code TEXT,
              code TEXT
            );
            CREATE VIRTUAL TABLE docs_fts USING fts5(
              chunk_id UNINDEXED,
              title,
              text,
              search_text,
              search_text_en,
              search_text_multilingual,
              description_en,
              generated_keywords_en,
              generated_synonyms_en,
              content=''
            );
            """
        )
        insert_docs = """
            INSERT INTO docs (
              chunk_id, vector_position, payload_json, title, text, search_text, search_text_en,
              search_text_multilingual, description_en, generated_keywords_en, generated_synonyms_en,
              doc_id, section_id, program_id, program_name, cpv_code, code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        insert_fts = """
            INSERT INTO docs_fts (
              chunk_id, title, text, search_text, search_text_en, search_text_multilingual,
              description_en, generated_keywords_en, generated_synonyms_en
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for position, document in enumerate(documents):
            payload_json = json.dumps(document, ensure_ascii=False)
            row = (
                str(document.get("chunk_id") or ""),
                int(position),
                payload_json,
                str(document.get("title") or ""),
                str(document.get("text") or ""),
                str(document.get("search_text") or ""),
                str(document.get("search_text_en") or ""),
                str(document.get("search_text_multilingual") or ""),
                str(document.get("description_en") or ""),
                str(document.get("generated_keywords_en") or ""),
                str(document.get("generated_synonyms_en") or ""),
                str(document.get("doc_id") or ""),
                str(document.get("section_id") or ""),
                str(document.get("program_id") or ""),
                str(document.get("program_name") or ""),
                str(document.get("cpv_code") or ""),
                str(document.get("code") or ""),
            )
            conn.execute(insert_docs, row)
            conn.execute(
                insert_fts,
                (
                    str(document.get("chunk_id") or ""),
                    str(document.get("title") or ""),
                    str(document.get("text") or ""),
                    str(document.get("search_text") or ""),
                    str(document.get("search_text_en") or ""),
                    str(document.get("search_text_multilingual") or ""),
                    str(document.get("description_en") or ""),
                    str(document.get("generated_keywords_en") or ""),
                    str(document.get("generated_synonyms_en") or ""),
                ),
            )
        conn.commit()
    finally:
        conn.close()

    if embeddings is not None:
        import faiss

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(faiss_path))

    manifest_payload = dict(manifest_payload)
    manifest_payload["count"] = len(documents)
    manifest_payload["has_faiss"] = bool(embeddings is not None)
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**paths, "index_name": index_name}


def load_faiss_index(faiss_path: str):
    if not faiss_path or not Path(faiss_path).exists():
        return None
    import faiss

    return faiss.read_index(faiss_path)


def load_vector_ids(sqlite_path: str) -> List[str]:
    conn = _open_db(sqlite_path)
    try:
        rows = conn.execute("SELECT chunk_id FROM docs ORDER BY vector_position ASC").fetchall()
        return [str(row["chunk_id"]) for row in rows]
    finally:
        conn.close()


def _metadata_sql(metadata_filter: Dict[str, object] | None) -> tuple[str, List[object]]:
    if not metadata_filter:
        return "", []
    clauses: List[str] = []
    params: List[object] = []
    for key, expected in metadata_filter.items():
        if expected is None or expected == "" or expected == []:
            continue
        values = expected if isinstance(expected, (list, tuple, set)) else [expected]
        values = [str(value) for value in values if str(value).strip()]
        if not values:
            continue
        placeholders = ",".join("?" for _ in values)
        clauses.append(f"d.{key} IN ({placeholders})")
        params.extend(values)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def lexical_search(
    *,
    sqlite_path: str,
    query: str,
    k: int,
    fields: Sequence[str],
    metadata_filter: Dict[str, object] | None = None,
) -> List[Dict[str, object]]:
    fts_query = _fts_query(query)
    if not fts_query:
        return []
    target_column = _target_column(fields)
    metadata_sql, metadata_params = _metadata_sql(metadata_filter)
    # docs_fts is a contentless FTS5 table, so auxiliary UNINDEXED columns like
    # chunk_id read back as NULL. Join via rowid instead; both docs and docs_fts
    # are rebuilt together in insertion order inside ensure_local_index().
    sql = f"""
        SELECT d.payload_json AS payload_json, -bm25(docs_fts) AS score
        FROM docs_fts
        JOIN docs d ON d.rowid = docs_fts.rowid
        WHERE docs_fts MATCH ?
          AND docs_fts.{target_column} MATCH ?
          {metadata_sql}
        ORDER BY score DESC
        LIMIT ?
    """
    conn = _open_db(sqlite_path)
    try:
        rows = conn.execute(sql, [fts_query, fts_query, *metadata_params, int(k)]).fetchall()
    except sqlite3.OperationalError:
        fallback_sql = f"""
            SELECT d.payload_json AS payload_json, -bm25(docs_fts) AS score
            FROM docs_fts
            JOIN docs d ON d.rowid = docs_fts.rowid
            WHERE docs_fts MATCH ?
              {metadata_sql}
            ORDER BY score DESC
            LIMIT ?
        """
        rows = conn.execute(fallback_sql, [fts_query, *metadata_params, int(k)]).fetchall()
    finally:
        conn.close()
    out: List[Dict[str, object]] = []
    for row in rows:
        payload = json.loads(str(row["payload_json"]))
        payload["score"] = float(row["score"] or 0.0)
        out.append(payload)
    return out


def fetch_documents_by_ids(sqlite_path: str, chunk_ids: Sequence[str]) -> Dict[str, Dict[str, object]]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    conn = _open_db(sqlite_path)
    try:
        rows = conn.execute(
            f"SELECT chunk_id, payload_json FROM docs WHERE chunk_id IN ({placeholders})",
            [str(chunk_id) for chunk_id in chunk_ids],
        ).fetchall()
    finally:
        conn.close()
    return {
        str(row["chunk_id"]): json.loads(str(row["payload_json"]))
        for row in rows
    }


def summarize_local_backend(config: Dict[str, object] | None, *, index_name: str | None = None, sqlite_path: str | None = None, faiss_path: str | None = None) -> Dict[str, object]:
    if not local_db_backend_enabled(config):
        return {"backend": "local", "index_name": None}
    return {
        "backend": "sqlite",
        "index_name": index_name,
        "index_dir": str((config or {}).get("index_dir") or ".rag_eval_indices"),
        "sqlite_path": sqlite_path,
        "faiss_path": faiss_path,
    }
