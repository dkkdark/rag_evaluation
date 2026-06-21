from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

from rag_eval.core.text_utils import metadata_value_matches
from rag_eval.retrieval.query_enrichment import procurement_type_bias
from rag_eval.retrieval.local_search import (
    build_local_index_name,
    chunk_fingerprint as local_chunk_fingerprint,
    ensure_local_index,
    fetch_documents_by_ids,
    lexical_search as local_lexical_search,
    load_faiss_index,
    load_vector_ids,
    local_db_backend_enabled,
    summarize_local_backend,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_SENTENCE_TRANSFORMER_CACHE: Dict[str, object] = {}


# Put different score ranges on the same 0..1 scale before mixing retrievers.
def min_max_normalize(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if abs(high - low) < 1e-12:
        return [1.0 if high > 0 else 0.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


# BM25 works best on normalized word tokens instead of raw text strings.
def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


def lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = set(tokenize_for_bm25(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize_for_bm25(text))
    if not text_tokens:
        return 0.0
    return len(query_tokens.intersection(text_tokens)) / len(query_tokens)


def format_dense_documents(texts: Sequence[str], model_name: str) -> List[str]:
    if "e5" in str(model_name).lower():
        return [f"passage: {text}" for text in texts]
    return list(texts)


def format_dense_query(query: str, model_name: str) -> str:
    if "e5" in str(model_name).lower():
        return f"query: {query}"
    return query


def chunk_search_text(chunk: Dict) -> str:
    title = str(chunk.get("title", "")) if int(chunk.get("chunk_index", 0)) == 0 else ""
    parts = [
        str(chunk.get("section_id", "")),
        title,
        str(chunk.get("text", "")),
    ]
    return "\n".join(part for part in parts if part.strip())


def get_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    cached_model = _SENTENCE_TRANSFORMER_CACHE.get(model_name)
    if cached_model is not None:
        return cached_model

    model = SentenceTransformer(model_name)
    _SENTENCE_TRANSFORMER_CACHE[model_name] = model
    return model


def get_dense_model(model_name: str):
    normalized_name = str(model_name or "").strip()
    if normalized_name.lower().startswith("text-embedding-"):
        raise ValueError(
            "OpenAI embedding models are not supported for retrieval. "
            "Use a sentence-transformers model such as "
            "'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'."
        )
    return {
        "backend": "sentence_transformer",
        "model_name": normalized_name or DEFAULT_EMBEDDING_MODEL,
        "client": get_sentence_transformer(normalized_name or DEFAULT_EMBEDDING_MODEL),
    }


def encode_dense_texts(model_state: Dict[str, object], texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
    prepared = [
        format_dense_query(text, str(model_state.get("model_name") or "")) if is_query
        else text
        for text in texts
    ]
    embeddings = model_state["client"].encode(
        prepared,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


def chunk_matches_filter(chunk: Dict, metadata_filter: Dict[str, object] | None) -> bool:
    if not metadata_filter:
        return True
    for key, expected in metadata_filter.items():
        if expected is None or expected == "" or expected == []:
            continue
        if isinstance(expected, dict):
            excluded = expected.get("exclude")
            if excluded is not None and excluded != "" and excluded != []:
                excluded_values = excluded if isinstance(excluded, (list, tuple, set)) else [excluded]
                if any(metadata_value_matches(chunk.get(key, ""), value) for value in excluded_values):
                    return False
            required = expected.get("include")
            if required is None or required == "" or required == []:
                continue
            values = required if isinstance(required, (list, tuple, set)) else [required]
            if not any(metadata_value_matches(chunk.get(key, ""), value) for value in values):
                return False
            continue
        values = expected if isinstance(expected, (list, tuple, set)) else [expected]
        if not any(metadata_value_matches(chunk.get(key, ""), value) for value in values):
            return False
    return True


# TF-IDF is a simple lexical retriever: it ranks chunks higher when they share
# the same important words or short phrases with the query. It is fast and easy
# to reason about, but it does not understand paraphrases very well.
def build_tfidf_retriever(chunks: Sequence[Dict]) -> Dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [chunk_search_text(chunk) for chunk in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50000,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    return {"backend": "tfidf", "vectorizer": vectorizer, "matrix": matrix}


# BM25 is also lexical, but usually stronger than plain TF-IDF for search. It
# still relies on word overlap, but it scores rare and informative terms more
# carefully, so it often works well for legal, academic, and technical text.
def build_bm25_retriever(chunks: Sequence[Dict]) -> Dict:
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize_for_bm25(chunk_search_text(chunk)) for chunk in chunks]
    return {
        "backend": "bm25",
        "bm25": BM25Okapi(tokenized_corpus),
        "tokenized_corpus": tokenized_corpus,
    }


# Dense retrieval converts the query and chunks into embeddings and compares
# them in vector space. This helps when the question and the document express
# the same idea with different words, but it can be less precise for exact
# wording matches.
def build_dense_retriever(chunks: Sequence[Dict], model_name: str) -> Dict:
    texts = format_dense_documents([chunk_search_text(chunk) for chunk in chunks], model_name)
    model = get_dense_model(model_name)
    embeddings = encode_dense_texts(model, texts)
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return {
        "backend": "dense",
        "model_name": model_name,
        "model": model,
        "index": index,
        "embeddings": embeddings,
    }


def _ensure_local_document_index(
    *,
    chunks: Sequence[Dict],
    model_name: str,
    search_backend_config: Dict[str, object],
    index_name: str | None,
    profile: str,
) -> Dict[str, object]:
    texts = [chunk_search_text(chunk) for chunk in chunks]
    model = None
    embeddings = None
    uses_dense = profile in {"dense", "hybrid"}
    if uses_dense:
        model = get_dense_model(model_name)
        embeddings = encode_dense_texts(model, format_dense_documents(texts, model_name))
    fingerprint = local_chunk_fingerprint(
        chunks,
        fields=["chunk_id", "doc_id", "section_id", "title", "text"],
        model_name=model_name if uses_dense else f"lexical-only:{profile}",
    )
    resolved_index_name = build_local_index_name(
        prefix=str(search_backend_config.get("index_prefix") or "rag-eval"),
        namespace="document-qa",
        fingerprint_source=fingerprint,
        explicit_name=index_name,
    )
    paths = ensure_local_index(
        config=search_backend_config,
        index_name=resolved_index_name,
        documents=[
            {
                **dict(chunk),
                "search_text": str(chunk.get("text") or ""),
                **(
                    {"dense_vector": [float(value) for value in embedding]}
                    if embeddings is not None
                    else {}
                ),
            }
            for chunk, embedding in zip(chunks, embeddings)
        ] if embeddings is not None else [
            {
                **dict(chunk),
                "search_text": str(chunk.get("text") or ""),
            }
            for chunk in chunks
        ],
        embeddings=embeddings,
        manifest_payload={
            "fingerprint": fingerprint,
            "model_name": model_name if uses_dense else f"lexical-only:{profile}",
            "namespace": "document-qa",
        },
    )
    return {
        "backend": "sqlite_document",
        "profile": None,
        "model_name": model_name,
        "model": model,
        "sqlite_path": paths["sqlite_path"],
        "faiss_path": paths["faiss_path"] if uses_dense else None,
        "vector_ids": load_vector_ids(paths["sqlite_path"]) if uses_dense else [],
        "index": load_faiss_index(paths["faiss_path"]) if uses_dense else None,
        "index_name": resolved_index_name,
        "search_backend_config": dict(search_backend_config),
        "search_backend": summarize_local_backend(
            search_backend_config,
            index_name=resolved_index_name,
            sqlite_path=paths["sqlite_path"],
            faiss_path=paths["faiss_path"] if uses_dense else None,
        ),
    }


def _build_dense_retriever_for_texts(texts: Sequence[str], model_name: str, search_field: str) -> Dict:
    formatted_texts = format_dense_documents(texts, model_name)
    model = get_dense_model(model_name)
    embeddings = encode_dense_texts(model, formatted_texts)
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return {
        "backend": "dense",
        "model_name": model_name,
        "model": model,
        "index": index,
        "embeddings": embeddings,
        "search_field": search_field,
    }


def _build_bm25_retriever_for_texts(texts: Sequence[str], search_field: str) -> Dict:
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize_for_bm25(text) for text in texts]
    return {
        "backend": "bm25",
        "bm25": BM25Okapi(tokenized_corpus),
        "tokenized_corpus": tokenized_corpus,
        "search_field": search_field,
    }


def _merge_hybrid_rows(
    dense_rows: Sequence[Dict],
    lexical_rows: Sequence[Dict],
    *,
    top_k: int,
    retriever_name: str,
) -> List[Dict]:
    if top_k <= 0:
        return []
    dense_scores = {str(row.get("chunk_id") or ""): float(row.get("score") or 0.0) for row in dense_rows}
    lexical_scores = {str(row.get("chunk_id") or ""): float(row.get("score") or 0.0) for row in lexical_rows}
    dense_norm = dict(zip(dense_scores.keys(), min_max_normalize(list(dense_scores.values()))))
    lexical_norm = dict(zip(lexical_scores.keys(), min_max_normalize(list(lexical_scores.values()))))
    all_ids = [chunk_id for chunk_id in dict.fromkeys(list(dense_scores.keys()) + list(lexical_scores.keys())) if chunk_id]
    row_lookup: Dict[str, Dict] = {}
    for row in list(dense_rows) + list(lexical_rows):
        chunk_id = str(row.get("chunk_id") or "")
        if chunk_id and chunk_id not in row_lookup:
            row_lookup[chunk_id] = dict(row)

    merged: List[Dict] = []
    for chunk_id in all_ids:
        row = dict(row_lookup[chunk_id])
        dense_score = dense_norm.get(chunk_id, 0.0)
        lexical_score = lexical_norm.get(chunk_id, 0.0)
        row["dense_score"] = dense_score
        row["bm25_score"] = lexical_score
        row["score"] = (0.5 * dense_score) + (0.5 * lexical_score)
        row["retriever"] = retriever_name
        merged.append(row)
    merged.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return merged[: min(top_k, len(merged))]


def build_cpv_multi_retriever(
    chunks: Sequence[Dict],
    model_name: str,
    profile: str,
    *,
    search_backend_config: Dict[str, object] | None = None,
    index_name: str | None = None,
) -> Dict:
    english_texts = [str(chunk.get("search_text_en") or chunk_search_text(chunk)) for chunk in chunks]
    alias_texts = [
        "\n".join(
            part for part in [
                str(chunk.get("cpv_code") or ""),
                str(chunk.get("cpv_label") or ""),
                str(chunk.get("description_multilingual_aliases") or ""),
                str(chunk.get("keywords_en") or ""),
            ] if part.strip()
        )
        for chunk in chunks
    ]
    multilingual_texts = [str(chunk.get("search_text_multilingual") or chunk_search_text(chunk)) for chunk in chunks]

    if local_db_backend_enabled(search_backend_config):
        uses_dense = profile in {"dense", "hybrid"}
        model = get_dense_model(model_name) if uses_dense else None
        embeddings = None
        if uses_dense:
            embeddings = encode_dense_texts(model, format_dense_documents(multilingual_texts, model_name))
        fingerprint = local_chunk_fingerprint(
            chunks,
            fields=["cpv_code", "cpv_label", "description_en", "search_text_en", "search_text_multilingual"],
            model_name=model_name if uses_dense else f"lexical-only:{profile}",
        )
        resolved_index_name = build_local_index_name(
            prefix=str((search_backend_config or {}).get("index_prefix") or "rag-eval"),
            namespace="ted-cpv",
            fingerprint_source=fingerprint,
            explicit_name=index_name,
        )
        paths = ensure_local_index(
            config=search_backend_config or {},
            index_name=resolved_index_name,
            documents=[
                {
                    **dict(chunk),
                    "code": str(chunk.get("cpv_code") or chunk.get("chunk_id") or ""),
                    "description_en": str(chunk.get("description_en") or chunk.get("text") or ""),
                    "generated_keywords_en": str(chunk.get("keywords_en") or ""),
                    "generated_synonyms_en": str(chunk.get("description_multilingual_aliases") or ""),
                    "search_text_en": str(chunk.get("search_text_en") or chunk.get("text") or ""),
                    "search_text_multilingual": str(
                        chunk.get("search_text_multilingual")
                        or chunk.get("search_text_en")
                        or chunk.get("text")
                        or ""
                    ),
                    **(
                        {"dense_vector": [float(value) for value in embedding]}
                        if embeddings is not None
                        else {}
                    ),
                }
                for chunk, embedding in zip(chunks, embeddings)
            ] if embeddings is not None else [
                {
                    **dict(chunk),
                    "code": str(chunk.get("cpv_code") or chunk.get("chunk_id") or ""),
                    "description_en": str(chunk.get("description_en") or chunk.get("text") or ""),
                    "generated_keywords_en": str(chunk.get("keywords_en") or ""),
                    "generated_synonyms_en": str(chunk.get("description_multilingual_aliases") or ""),
                    "search_text_en": str(chunk.get("search_text_en") or chunk.get("text") or ""),
                    "search_text_multilingual": str(
                        chunk.get("search_text_multilingual")
                        or chunk.get("search_text_en")
                        or chunk.get("text")
                        or ""
                    ),
                }
                for chunk in chunks
            ],
            embeddings=embeddings,
            manifest_payload={
                "fingerprint": fingerprint,
                "model_name": model_name if uses_dense else f"lexical-only:{profile}",
                "namespace": "ted-cpv",
            },
        )
        return {
            "backend": "sqlite_cpv_multi",
            "profile": profile,
            "model_name": model_name,
            "model": model if uses_dense else None,
            "sqlite_path": paths["sqlite_path"],
            "faiss_path": paths["faiss_path"] if uses_dense else None,
            "vector_ids": load_vector_ids(paths["sqlite_path"]) if uses_dense else [],
            "index": load_faiss_index(paths["faiss_path"]) if uses_dense else None,
            "index_name": resolved_index_name,
            "rrf_k": 60.0,
            "search_backend_config": dict(search_backend_config or {}),
            "search_backend": summarize_local_backend(
                search_backend_config,
                index_name=resolved_index_name,
                sqlite_path=paths["sqlite_path"],
                faiss_path=paths["faiss_path"] if uses_dense else None,
            ),
        }

    indexes: List[Tuple[str, Dict, float, str]] = []
    if profile == "hybrid":
        indexes.append(("dense_multilingual", _build_dense_retriever_for_texts(multilingual_texts, model_name, "search_text_multilingual"), 1.0, "original"))
        indexes.append(("bm25_multilingual", _build_bm25_retriever_for_texts(multilingual_texts, "search_text_multilingual"), 1.0, "original"))
        return {
            "backend": "cpv_multi",
            "profile": profile,
            "indexes": indexes,
            "rrf_k": 60.0,
        }
    if profile in {"dense", "hybrid"}:
        indexes.append(("dense_multilingual", _build_dense_retriever_for_texts(multilingual_texts, model_name, "search_text_multilingual"), 1.0, "original"))
        indexes.append(("dense_english", _build_dense_retriever_for_texts(english_texts, model_name, "search_text_en"), 0.9, "translated"))
    if profile in {"bm25", "tfidf", "dense"}:
        indexes.append(("bm25_english", _build_bm25_retriever_for_texts(english_texts, "search_text_en"), 0.85, "translated"))
        indexes.append(("bm25_aliases", _build_bm25_retriever_for_texts(alias_texts, "description_multilingual_aliases"), 0.7, "original"))

    return {
        "backend": "cpv_multi",
        "profile": profile,
        "indexes": indexes,
        "rrf_k": 60.0,
    }


def build_retriever_with_backend(
    chunks: Sequence[Dict],
    retriever_type: str,
    model_name: str,
    *,
    search_backend_config: Dict[str, object] | None = None,
    index_name: str | None = None,
) -> Dict:
    if local_db_backend_enabled(search_backend_config):
        state = _ensure_local_document_index(
            chunks=chunks,
            model_name=model_name,
            search_backend_config=search_backend_config or {},
            index_name=index_name,
            profile=retriever_type,
        )
        state["profile"] = retriever_type
        return state
    if retriever_type == "tfidf":
        return build_tfidf_retriever(chunks)
    if retriever_type == "bm25":
        return build_bm25_retriever(chunks)
    if retriever_type == "dense":
        return build_dense_retriever(chunks, model_name)
    if retriever_type == "hybrid":
        return {
            "backend": "hybrid",
            "dense": build_dense_retriever(chunks, model_name),
            "bm25": build_bm25_retriever(chunks),
        }
    raise ValueError(f"Unsupported retriever: {retriever_type}")


def _retrieve_top_k_from_single_state(
    *,
    query: str,
    retriever_state: Dict,
    chunks: Sequence[Dict],
    k: int,
    metadata_filter: Dict[str, object] | None = None,
) -> List[Dict]:
    if k <= 0:
        return []
    if retriever_state["backend"] == "dense":
        q = encode_dense_texts(retriever_state["model"], [query], is_query=True)
        search_k = len(chunks) if metadata_filter else min(k, len(chunks))
        scores, indices = retriever_state["index"].search(q, search_k)
        out: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            row = dict(chunks[int(idx)])
            if not chunk_matches_filter(row, metadata_filter):
                continue
            row["score"] = float(score)
            row["retriever"] = "dense"
            out.append(row)
            if len(out) >= k:
                break
        return out
    if retriever_state["backend"] == "sqlite_document":
        profile = str(retriever_state.get("profile") or "hybrid")
        if profile in {"tfidf", "bm25"}:
            rows = local_lexical_search(
                sqlite_path=str(retriever_state["sqlite_path"]),
                query=query,
                k=k,
                fields=["search_text"],
                metadata_filter=metadata_filter,
            )
            for row in rows:
                row["retriever"] = "bm25"
            return rows
        q = encode_dense_texts(retriever_state["model"], [query], is_query=True)
        if profile == "dense":
            search_k = len(retriever_state["vector_ids"]) if metadata_filter else min(k, len(retriever_state["vector_ids"]))
            scores, indices = retriever_state["index"].search(q, search_k)
            candidate_ids = [retriever_state["vector_ids"][int(idx)] for idx in indices[0] if int(idx) >= 0]
            row_lookup = fetch_documents_by_ids(str(retriever_state["sqlite_path"]), candidate_ids)
            out: List[Dict] = []
            for score, chunk_id in zip(scores[0], candidate_ids):
                row = dict(row_lookup.get(chunk_id) or {})
                if not row or not chunk_matches_filter(row, metadata_filter):
                    continue
                row["score"] = float(score)
                row["retriever"] = "dense"
                out.append(row)
                if len(out) >= k:
                    break
            return out
        if profile == "hybrid":
            dense_rows = _retrieve_top_k_from_single_state(
                query=query,
                retriever_state={**retriever_state, "profile": "dense"},
                chunks=chunks,
                k=max(k, 100),
                metadata_filter=metadata_filter,
            )
            lexical_rows = local_lexical_search(
                sqlite_path=str(retriever_state["sqlite_path"]),
                query=query,
                k=max(k, 100),
                fields=["search_text"],
                metadata_filter=metadata_filter,
            )
            dense_scores = {row["chunk_id"]: float(row.get("score") or 0.0) for row in dense_rows}
            lexical_scores = {row["chunk_id"]: float(row.get("score") or 0.0) for row in lexical_rows}
            dense_norm = dict(zip(dense_scores.keys(), min_max_normalize(list(dense_scores.values()))))
            lexical_norm = dict(zip(lexical_scores.keys(), min_max_normalize(list(lexical_scores.values()))))
            merged: List[Dict] = []
            all_ids = set(dense_scores) | set(lexical_scores)
            row_lookup = {row["chunk_id"]: row for row in dense_rows + lexical_rows}
            for chunk_id in all_ids:
                row = dict(row_lookup[chunk_id])
                row["dense_score"] = dense_norm.get(chunk_id, 0.0)
                row["bm25_score"] = lexical_norm.get(chunk_id, 0.0)
                row["score"] = (0.5 * row["dense_score"]) + (0.5 * row["bm25_score"])
                row["retriever"] = "hybrid"
                merged.append(row)
            merged.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            return merged[:k]
    if retriever_state["backend"] == "bm25":
        tokenized_query = tokenize_for_bm25(query)
        scores = np.asarray(retriever_state["bm25"].get_scores(tokenized_query), dtype=float)
        order = np.argsort(-scores)
        out: List[Dict] = []
        for idx in order:
            row = dict(chunks[int(idx)])
            if not chunk_matches_filter(row, metadata_filter):
                continue
            row["score"] = float(scores[int(idx)])
            row["retriever"] = "bm25"
            out.append(row)
            if len(out) >= k:
                break
        return out
    raise ValueError(f"Unsupported single-state backend: {retriever_state['backend']}")


def _build_cpv_search_channel_specs(
    *,
    query_original: str,
    query_translated_to_en: str,
    query_object: str,
    query_object_translated: str,
    search_channels: Sequence[Dict[str, object]] | None,
) -> List[tuple[str, str, float, str, List[str]]]:
    if search_channels:
        specs: List[tuple[str, str, float, str, List[str]]] = []
        for channel in search_channels:
            query_text = str(channel.get("query") or "").strip()
            if not query_text:
                continue
            channel_name = str(channel.get("name") or "channel")
            weight = float(channel.get("weight") or 1.0)
            kind = str(channel.get("kind") or "both")
            fields_en = [str(field) for field in channel.get("fields_en", ["search_text_en"]) if str(field).strip()]
            fields_multilingual = [
                str(field) for field in channel.get("fields_multilingual", ["search_text_multilingual"]) if str(field).strip()
            ]
            if kind in {"both", "vector"}:
                specs.append((f"dense_{channel_name}", "vector", weight, query_text, fields_multilingual))
            if kind in {"both", "lexical"}:
                specs.append((f"bm25_{channel_name}", "lexical", weight, query_text, fields_en or fields_multilingual))
        return specs

    query_source_original = query_original
    query_source_translated = query_translated_to_en or query_original
    object_query = (query_object or query_original or "").strip()
    object_query_translated = (query_object_translated or query_translated_to_en or object_query).strip()
    return [
        ("dense_object_multilingual", "vector", 1.25, object_query, ["search_text_multilingual"]),
        ("dense_multilingual", "vector", 1.0, query_source_original, ["search_text_multilingual"]),
        ("dense_object_english", "vector", 1.15, object_query_translated, ["search_text_en"]),
        ("dense_english", "vector", 0.95, query_source_translated, ["search_text_en"]),
        ("bm25_object_english", "lexical", 1.2, object_query_translated, ["search_text_en"]),
        ("bm25_english", "lexical", 0.9, query_source_translated, ["search_text_en"]),
        ("bm25_object_aliases", "lexical", 1.05, object_query, ["search_text_multilingual"]),
        ("bm25_aliases", "lexical", 0.75, query_source_original, ["search_text_multilingual"]),
    ]


def retrieve_top_k_cpv_multi(
    *,
    query_original: str,
    query_translated_to_en: str,
    query_object: str = "",
    query_object_translated: str = "",
    procurement_action: str = "",
    secondary_context: str = "",
    exclude_as_primary: Sequence[str] | None = None,
    search_channels: Sequence[Dict[str, object]] | None = None,
    primary_selection_bias: str = "",
    service_component: bool = False,
    retriever_state: Dict,
    chunks: Sequence[Dict],
    k: int,
    metadata_filter: Dict[str, object] | None = None,
) -> List[Dict]:
    if k <= 0:
        return []
    if retriever_state.get("backend") != "cpv_multi":
        if retriever_state.get("backend") == "sqlite_cpv_multi":
            pass
        else:
            raise ValueError("retrieve_top_k_cpv_multi requires a cpv_multi retriever state.")

    if retriever_state.get("backend") == "sqlite_cpv_multi":
        if retriever_state.get("profile") == "hybrid":
            query_text = query_original or query_translated_to_en
            if not str(query_text or "").strip():
                return []
            per_index_depth = max(k, 200)
            model = retriever_state["model"]
            q = encode_dense_texts(model, [query_text], is_query=True)
            scores, indices = retriever_state["index"].search(q, min(per_index_depth, len(retriever_state["vector_ids"])))
            candidate_ids = [retriever_state["vector_ids"][int(idx)] for idx in indices[0] if int(idx) >= 0]
            row_lookup = fetch_documents_by_ids(str(retriever_state["sqlite_path"]), candidate_ids)
            dense_rows = []
            for score, chunk_id in zip(scores[0], candidate_ids):
                row = dict(row_lookup.get(chunk_id) or {})
                if not row or not chunk_matches_filter(row, metadata_filter):
                    continue
                row["score"] = float(score)
                dense_rows.append(row)
            lexical_rows = local_lexical_search(
                sqlite_path=str(retriever_state["sqlite_path"]),
                query=query_text,
                k=per_index_depth,
                fields=["search_text_multilingual"],
                metadata_filter=metadata_filter,
            )
            return _merge_hybrid_rows(
                dense_rows,
                lexical_rows,
                top_k=k,
                retriever_name="hybrid",
            )
        fused_scores: Dict[str, float] = {}
        fused_rows: Dict[str, Dict] = {}
        rrf_k = float(retriever_state.get("rrf_k") or 60.0)
        per_index_depth = max(k, 500)
        object_query = (query_object or query_original or "").strip()
        index_specs = _build_cpv_search_channel_specs(
            query_original=query_original,
            query_translated_to_en=query_translated_to_en,
            query_object=object_query,
            query_object_translated=(query_object_translated or query_translated_to_en or object_query).strip(),
            search_channels=search_channels,
        )
        model = retriever_state["model"]
        for index_name, kind, weight, query_text, lexical_fields in index_specs:
            if not str(query_text or "").strip():
                continue
            if retriever_state.get("profile") in {"bm25", "tfidf"} and kind == "vector":
                continue
            if retriever_state.get("profile") == "dense" and kind == "lexical":
                continue
            if kind == "vector":
                q = encode_dense_texts(model, [query_text], is_query=True)
                scores, indices = retriever_state["index"].search(q, min(per_index_depth, len(retriever_state["vector_ids"])))
                candidate_ids = [retriever_state["vector_ids"][int(idx)] for idx in indices[0] if int(idx) >= 0]
                row_lookup = fetch_documents_by_ids(str(retriever_state["sqlite_path"]), candidate_ids)
                rows = []
                for score, chunk_id in zip(scores[0], candidate_ids):
                    row = dict(row_lookup.get(chunk_id) or {})
                    if not row or not chunk_matches_filter(row, metadata_filter):
                        continue
                    row["score"] = float(score)
                    rows.append(row)
            else:
                rows = local_lexical_search(
                    sqlite_path=str(retriever_state["sqlite_path"]),
                    query=query_text,
                    k=per_index_depth,
                    fields=lexical_fields,
                    metadata_filter=metadata_filter,
                )
            for rank, row in enumerate(rows, start=1):
                chunk_id = str(row.get("chunk_id") or "")
                score = float(weight) / (rrf_k + rank)
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + score
                fused_row = fused_rows.setdefault(chunk_id, dict(row))
                fused_row.setdefault("rrf_sources", [])
                fused_row["rrf_sources"].append(index_name)
                fused_row[f"{index_name}_rank"] = rank
                fused_row[f"{index_name}_score"] = float(row.get("score") or 0.0)
        ranked = []
        exclusion_text = " ".join(str(value).strip() for value in (exclude_as_primary or []) if str(value).strip())
        for chunk_id, score in fused_scores.items():
            row = fused_rows[chunk_id]
            english_text = " ".join(
                str(row.get(key) or "")
                for key in ["cpv_label", "description_en", "use_when_text", "do_not_use_when_text", "children_labels", "sibling_labels", "search_text_en"]
            )
            multilingual_text = str(row.get("search_text_multilingual") or english_text)
            object_overlap = max(
                lexical_overlap_score(object_query, english_text),
                lexical_overlap_score(object_query, multilingual_text),
            )
            action_overlap = lexical_overlap_score(procurement_action, english_text)
            secondary_overlap = lexical_overlap_score(secondary_context, english_text)
            exclusion_overlap = lexical_overlap_score(exclusion_text, english_text)
            focus_bonus = (0.14 * object_overlap) + (0.05 * action_overlap) + (0.02 * secondary_overlap)
            if exclusion_overlap > object_overlap:
                focus_bonus -= 0.08 * exclusion_overlap
            type_bonus = procurement_type_bias(
                primary_selection_bias=primary_selection_bias,
                service_component=service_component,
                candidate_procurement_type=str(row.get("procurement_type") or ""),
            )
            row["object_overlap_score"] = object_overlap
            row["procurement_action_overlap_score"] = action_overlap
            row["secondary_context_overlap_score"] = secondary_overlap
            row["exclude_overlap_score"] = exclusion_overlap
            row["procurement_type_bias"] = type_bonus
            row["score"] = float(score + focus_bonus + type_bonus)
            row["retriever"] = f"{retriever_state.get('profile', 'hybrid')}_rrf"
            row["rrf_source_count"] = len(row.get("rrf_sources") or [])
            ranked.append(row)
        ranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return ranked[: min(k, len(ranked))]

    if retriever_state.get("profile") == "hybrid":
        dense_state = next((state for index_name, state, _, _ in retriever_state.get("indexes", []) if index_name == "dense_multilingual"), None)
        lexical_state = next((state for index_name, state, _, _ in retriever_state.get("indexes", []) if index_name == "bm25_multilingual"), None)
        query_text = query_original or query_translated_to_en
        if dense_state is None or lexical_state is None or not str(query_text or "").strip():
            return []
        dense_rows = _retrieve_top_k_from_single_state(
            query=query_text,
            retriever_state=dense_state,
            chunks=chunks,
            k=min(max(k, 200), len(chunks)),
            metadata_filter=metadata_filter,
        )
        lexical_rows = _retrieve_top_k_from_single_state(
            query=query_text,
            retriever_state=lexical_state,
            chunks=chunks,
            k=min(max(k, 200), len(chunks)),
            metadata_filter=metadata_filter,
        )
        return _merge_hybrid_rows(
            dense_rows,
            lexical_rows,
            top_k=k,
            retriever_name="hybrid",
        )

    fused_scores: Dict[str, float] = {}
    fused_rows: Dict[str, Dict] = {}
    rrf_k = float(retriever_state.get("rrf_k") or 60.0)
    per_index_depth = min(max(k, 120), len(chunks))

    for index_name, state, weight, query_source in retriever_state.get("indexes", []):
        query_text = query_original if query_source == "original" else (query_translated_to_en or query_original)
        rows = _retrieve_top_k_from_single_state(
            query=query_text,
            retriever_state=state,
            chunks=chunks,
            k=per_index_depth,
            metadata_filter=metadata_filter,
        )
        for rank, row in enumerate(rows, start=1):
            chunk_id = str(row.get("chunk_id") or "")
            score = float(weight) / (rrf_k + rank)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + score
            fused_row = fused_rows.setdefault(chunk_id, dict(row))
            fused_row.setdefault("rrf_sources", [])
            fused_row["rrf_sources"].append(index_name)
            fused_row[f"{index_name}_rank"] = rank
            fused_row[f"{index_name}_score"] = float(row.get("score") or 0.0)

    ranked = []
    for chunk_id, score in fused_scores.items():
        row = fused_rows[chunk_id]
        row["score"] = float(score)
        row["retriever"] = f"{retriever_state.get('profile', 'hybrid')}_rrf"
        row["rrf_source_count"] = len(row.get("rrf_sources") or [])
        ranked.append(row)
    ranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return ranked[: min(k, len(ranked))]


# Retrieve the top-k chunks with the selected strategy and attach scores so we
# can inspect why a chunk was returned.
def retrieve_top_k(
    query: str,
    retriever_state: Dict,
    chunks: Sequence[Dict],
    k: int,
    hybrid_alpha: float = 0.5,
    metadata_filter: Dict[str, object] | None = None,
) -> List[Dict]:
    import numpy as np

    if k <= 0:
        return []

    if retriever_state["backend"] == "sqlite_document":
        return _retrieve_top_k_from_single_state(
            query=query,
            retriever_state=retriever_state,
            chunks=chunks,
            k=k,
            metadata_filter=metadata_filter,
        )

    if retriever_state["backend"] == "dense":
        q = encode_dense_texts(retriever_state["model"], [query], is_query=True)
        search_k = len(chunks) if metadata_filter else min(k, len(chunks))
        scores, indices = retriever_state["index"].search(q, search_k)
        out: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            row = dict(chunks[int(idx)])
            if not chunk_matches_filter(row, metadata_filter):
                continue
            row["score"] = float(score)
            row["retriever"] = "dense"
            out.append(row)
            if len(out) >= k:
                break
        return out

    if retriever_state["backend"] == "tfidf":
        q = retriever_state["vectorizer"].transform([query])
        scores = (retriever_state["matrix"] @ q.T).toarray().ravel()
        order = np.argsort(-scores)
        out = []
        for idx in order:
            row = dict(chunks[int(idx)])
            if not chunk_matches_filter(row, metadata_filter):
                continue
            row["score"] = float(scores[int(idx)])
            row["retriever"] = "tfidf"
            out.append(row)
            if len(out) >= k:
                break
        return out

    if retriever_state["backend"] == "bm25":
        tokenized_query = tokenize_for_bm25(query)
        scores = np.asarray(retriever_state["bm25"].get_scores(tokenized_query), dtype=float)
        order = np.argsort(-scores)
        out = []
        for idx in order:
            row = dict(chunks[int(idx)])
            if not chunk_matches_filter(row, metadata_filter):
                continue
            row["score"] = float(scores[int(idx)])
            row["retriever"] = "bm25"
            out.append(row)
            if len(out) >= k:
                break
        return out

    if retriever_state["backend"] == "hybrid":
        dense_rows = retrieve_top_k(
            query=query,
            retriever_state=retriever_state["dense"],
            chunks=chunks,
            k=len(chunks),
            hybrid_alpha=hybrid_alpha,
            metadata_filter=metadata_filter,
        )
        bm25_rows = retrieve_top_k(
            query=query,
            retriever_state=retriever_state["bm25"],
            chunks=chunks,
            k=len(chunks),
            hybrid_alpha=hybrid_alpha,
            metadata_filter=metadata_filter,
        )
        dense_scores = {row["chunk_id"]: row["score"] for row in dense_rows}
        bm25_scores = {row["chunk_id"]: row["score"] for row in bm25_rows}
        dense_norm = dict(zip(dense_scores.keys(), min_max_normalize(list(dense_scores.values()))))
        bm25_norm = dict(zip(bm25_scores.keys(), min_max_normalize(list(bm25_scores.values()))))

        # Hybrid retrieval mixes dense semantic matching with BM25 word-based
        # matching. In practice this is often the safest default because it can
        # reward both exact terminology and meaning-level similarity.
        combined: List[Dict] = []
        for chunk in chunks:
            if not chunk_matches_filter(chunk, metadata_filter):
                continue
            chunk_id = chunk["chunk_id"]
            dense_score = dense_norm.get(chunk_id, 0.0)
            bm25_score = bm25_norm.get(chunk_id, 0.0)
            score = hybrid_alpha * dense_score + (1.0 - hybrid_alpha) * bm25_score
            row = dict(chunk)
            row["score"] = float(score)
            row["dense_score"] = float(dense_score)
            row["bm25_score"] = float(bm25_score)
            row["retriever"] = "hybrid"
            combined.append(row)

        combined.sort(key=lambda item: item["score"], reverse=True)
        return combined[: min(k, len(combined))]

    raise ValueError(f"Unsupported retriever backend: {retriever_state['backend']}")


def rerank_with_lexical_signal(
    *,
    query: str,
    rows: Sequence[Dict],
    top_k: int,
    rerank_top_n: int,
    rerank_weight: float,
) -> List[Dict]:
    if top_k <= 0:
        return []
    if rerank_top_n <= 1 or rerank_weight <= 0.0 or not rows:
        return list(rows[:top_k])

    limited_top_n = min(len(rows), max(top_k, rerank_top_n))
    head = [dict(row) for row in rows[:limited_top_n]]
    tail = [dict(row) for row in rows[limited_top_n:]]

    base_scores = [float(row.get("score") or 0.0) for row in head]
    base_norm = min_max_normalize(base_scores)
    lexical_scores = [
        lexical_overlap_score(query, f"{row.get('title', '')}\n{row.get('text', '')}")
        for row in head
    ]

    reranked: List[Dict] = []
    for row, base_score, lexical_score in zip(head, base_norm, lexical_scores):
        final_score = (1.0 - rerank_weight) * base_score + rerank_weight * lexical_score
        row["base_score_before_rerank"] = float(row.get("score") or 0.0)
        row["lexical_rerank_score"] = lexical_score
        row["score"] = float(final_score)
        row["reranked"] = True
        reranked.append(row)

    reranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    combined = reranked + tail
    return combined[:top_k]
