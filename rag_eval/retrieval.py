from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

from rag_eval.chunking import chunk_by_section
from rag_eval.models import Section

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
    return re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", text.lower())


def is_e5_model(model_name: str) -> bool:
    return "e5" in model_name.casefold()


def is_e5_mistral_model(model_name: str) -> bool:
    lowered = model_name.casefold()
    return "e5" in lowered and "mistral" in lowered


def format_dense_documents(texts: Sequence[str], model_name: str) -> List[str]:
    # We keep raw chunk text for all dense models because prefix-based formatting
    # degraded retrieval quality on this document collection.
    return list(texts)


def format_dense_query(query: str, model_name: str) -> str:
    # Keep the raw query as well so query/document formatting stays symmetric
    # with historical runs and remains easy to compare across embedding models.
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


def normalize_metadata_value(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("&", "and").replace("/", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    aliases = {
        "ba": "bachelor",
        "bpo": "bachelor",
        "ma": "master",
        "mpo": "master",
        "sciences": "science",
    }
    return " ".join(aliases.get(token, token) for token in normalized.split())


def metadata_value_matches(actual: object, expected: object) -> bool:
    actual_norm = normalize_metadata_value(actual)
    expected_norm = normalize_metadata_value(expected)
    return (
        actual_norm == expected_norm
        or actual_norm in expected_norm
        or expected_norm in actual_norm
    )


def chunk_matches_filter(chunk: Dict, metadata_filter: Dict[str, object] | None) -> bool:
    if not metadata_filter:
        return True
    for key, expected in metadata_filter.items():
        if expected is None or expected == "" or expected == []:
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
    import faiss

    texts = format_dense_documents([chunk_search_text(chunk) for chunk in chunks], model_name)
    model = get_sentence_transformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return {
        "backend": "dense",
        "model_name": model_name,
        "model": model,
        "index": index,
        "embeddings": embeddings,
    }


# Build one retrieval backend behind a shared interface so the rest of the
# pipeline can switch strategies without changing the experiment logic.
def build_retriever(chunks: Sequence[Dict], retriever_type: str, model_name: str) -> Dict:
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

    if retriever_state["backend"] == "dense":
        q = retriever_state["model"].encode(
            [format_dense_query(query, retriever_state["model_name"])],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
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


# Build a lightweight section-level retriever when coarse document navigation is enough.
def build_section_retriever_for_enrichment(sections: Sequence[Section]) -> Tuple[List[Dict], Dict]:
    section_chunks = chunk_by_section(sections)
    retriever_state = build_retriever(section_chunks, "tfidf", "")
    return section_chunks, retriever_state
