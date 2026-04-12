#!/usr/bin/env python3
import glob
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

import faiss
import numpy as np
import pandas as pd
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer


SECTION_RE = re.compile(r"^§\s*(\d+[a-zA-Z]?)\s+(.+)$")
EMPTY_TABLE_CELL = "[EMPTY]"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_SENTENCE_TRANSFORMER_CACHE: Dict[str, object] = {}


@dataclass
class Section:
    doc_id: str
    section_id: str
    title: str
    text: str


def clean_text(text: str) -> str:
    # Minimal PDF cleanup: unwrap soft hyphenation, normalize spaces/newlines.
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-zäöüß])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def normalize_table_cell(cell: str | None) -> str:
    if cell is None:
        return EMPTY_TABLE_CELL
    normalized = clean_text(cell)
    normalized = normalized.replace("\n", " / ").strip()
    return normalized or EMPTY_TABLE_CELL


def serialize_table_rows(rows: List[List[str | None]], table_index: int) -> str:
    non_empty_rows = [list(row) for row in rows if any((cell or "").strip() for cell in row)]
    if not non_empty_rows:
        return ""

    max_cols = max(len(row) for row in non_empty_rows)
    lines = [f"[TABLE {table_index}]"]
    for row in non_empty_rows:
        padded = list(row) + [None] * (max_cols - len(row))
        lines.append(" | ".join(normalize_table_cell(cell) for cell in padded))
    return "\n".join(lines)


def extract_page_text_outside_tables(page, table_bboxes: List[Tuple[float, float, float, float]]) -> str:
    if not table_bboxes:
        return page.extract_text() or ""

    filtered_page = page
    try:
        for bbox in table_bboxes:
            filtered_page = filtered_page.outside_bbox(bbox)
        return filtered_page.extract_text() or ""
    except Exception:
        return page.extract_text() or ""


def extract_pdf_page_content(page) -> str:
    table_blocks: List[str] = []
    table_bboxes: List[Tuple[float, float, float, float]] = []

    try:
        tables = page.find_tables()
    except Exception:
        tables = []

    for table_index, table in enumerate(tables, start=1):
        serialized = serialize_table_rows(table.extract(), table_index)
        if serialized:
            table_blocks.append(serialized)
            bbox = getattr(table, "bbox", None)
            if bbox:
                table_bboxes.append(bbox)

    page_text = extract_page_text_outside_tables(page, table_bboxes)
    parts = [part.strip() for part in [page_text, *table_blocks] if part and part.strip()]
    return "\n\n".join(parts)


def parse_pdf_sections(pdf_path: str) -> Tuple[str, List[Section]]:
    doc_id = os.path.basename(pdf_path)
    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(extract_pdf_page_content(page))
    full_text = clean_text("\n".join(pages))

    sections: List[Section] = []
    cur_id = "PREAMBLE"
    cur_title = "Preamble"
    buf: List[str] = []

    # Split document by headings like "§ 12 Title".
    for line in full_text.splitlines():
        match = SECTION_RE.match(line)
        if match:
            if buf:
                sections.append(
                    Section(
                        doc_id=doc_id,
                        section_id=cur_id,
                        title=cur_title,
                        text="\n".join(buf).strip(),
                    )
                )
            cur_id = f"§ {match.group(1)}"
            cur_title = match.group(2).strip()
            buf = []
        else:
            buf.append(line)

    if buf:
        sections.append(
            Section(
                doc_id=doc_id,
                section_id=cur_id,
                title=cur_title,
                text="\n".join(buf).strip(),
            )
        )

    sections = [s for s in sections if s.text]
    return full_text, sections


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def chunk_sections(
    sections: List[Section], chunk_size: int = 450, chunk_overlap: int = 60
) -> List[Dict]:
    # Fixed-size word chunks with overlap as a simple baseline strategy.
    chunks: List[Dict] = []
    for sec in sections:
        words = tokenize_words(sec.text)
        if not words:
            continue
        start = 0
        idx = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{sec.doc_id}|{sec.section_id}|{idx}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": sec.doc_id,
                    "section_id": sec.section_id,
                    "title": sec.title,
                    "text": chunk_text,
                    "n_words": len(chunk_words),
                }
            )
            if end == len(words):
                break
            start = max(end - chunk_overlap, start + 1)
            idx += 1
    return chunks


def load_questions(qa_path: str) -> List[Dict]:
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        items = data["questions"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("qa_seed.json should be a list or {'questions': [...]} object.")
    for i, item in enumerate(items):
        if "id" not in item:
            item["id"] = f"q{i+1}"
    return items


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return x / norms


def is_e5_model(model_name: str) -> bool:
    return "e5" in model_name.lower()


def is_e5_mistral_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return "e5" in lowered and "mistral" in lowered


def format_dense_documents(texts: List[str], model_name: str) -> List[str]:
    # Raw chunk text worked better for this corpus than prefix-based E5 formatting.
    return texts


def format_dense_query(query: str, model_name: str) -> str:
    # Keep the original raw query to match earlier retrieval behavior.
    return query


def get_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    cached_model = _SENTENCE_TRANSFORMER_CACHE.get(model_name)
    if cached_model is not None:
        return cached_model

    model = SentenceTransformer(model_name)
    _SENTENCE_TRANSFORMER_CACHE[model_name] = model
    return model


def build_index(
    chunks: List[Dict], embedding_backend: str, model_name: str
) -> Tuple[Dict, faiss.Index, np.ndarray]:
    texts = [c["text"] for c in chunks]
    if embedding_backend == "sentence-transformers":
        model = get_sentence_transformer(model_name)
        emb = model.encode(
            format_dense_documents(texts, model_name),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return {"backend": embedding_backend, "model": model, "model_name": model_name}, index, emb

    # Default/offline path: no network/model download required.
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), max_features=50000, min_df=1
    )
    matrix = vectorizer.fit_transform(texts)
    emb = matrix.toarray().astype("float32")
    emb = l2_normalize(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return {"backend": "tfidf", "vectorizer": vectorizer}, index, emb


def retrieve_top_k(
    query: str,
    embedding_state: Dict,
    index: faiss.Index,
    chunks: List[Dict],
    k: int = 5,
) -> List[Dict]:
    # Always retrieve by cosine-like similarity
    if embedding_state["backend"] == "sentence-transformers":
        model = embedding_state["model"]
        q = model.encode(
            [format_dense_query(query, embedding_state["model_name"])],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
    else:
        vectorizer = embedding_state["vectorizer"]
        q = vectorizer.transform([query]).toarray().astype("float32")
        q = l2_normalize(q)
    scores, idx = index.search(q, k)
    out: List[Dict] = []
    for score, i in zip(scores[0], idx[0]):
        row = dict(chunks[int(i)])
        row["score"] = float(score)
        out.append(row)
    return out


def keyword_extractive_answer(question: str, retrieved: List[Dict]) -> str:
    # Lightweight answer generator: pick sentence with max lexical overlap.
    q_words = set(
        w.lower()
        for w in re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", question)
        if len(w) > 3
    )
    best_sentence = ""
    best_score = -1
    for r in retrieved:
        sentences = re.split(r"(?<=[.!?])\s+", r["text"])
        for s in sentences:
            s_words = set(w.lower() for w in re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", s))
            overlap = len(q_words.intersection(s_words))
            if overlap > best_score:
                best_score = overlap
                best_sentence = s.strip()
    if best_sentence:
        return best_sentence
    return retrieved[0]["text"][:400] if retrieved else "No context."


def evaluate_answer(answer: str, item: Dict) -> str:
    # Simple automatic check for seed questions with expected keywords.
    answer_low = answer.lower()
    if item.get("expected_keywords"):
        kws = [str(k).lower() for k in item["expected_keywords"]]
        ok = all(kw in answer_low for kw in kws)
        return "correct" if ok else "incorrect"
    if item.get("gold_answer"):
        gold = str(item["gold_answer"]).lower()
        return "correct" if gold in answer_low else "incorrect"
    return "needs_manual_review"


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    # Minimal fixed config for baseline (no CLI args).
    pdf_glob = "*.pdf*"
    qa_file = "qa_seed.json"
    out_dir = "outputs"
    top_k = 5
    chunk_size = 450
    chunk_overlap = 60
    embedding_backend = "tfidf"
    embedding_model = DEFAULT_EMBEDDING_MODEL

    ensure_out_dir(out_dir)
    pdf_paths = sorted(glob.glob(pdf_glob))

    all_raw_parts: List[str] = []
    all_sections: List[Section] = []
    for p in pdf_paths:
        raw_text, sections = parse_pdf_sections(p)
        all_raw_parts.append(f"===== {os.path.basename(p)} =====\n{raw_text}")
        all_sections.extend(sections)

    raw_path = os.path.join(out_dir, "regulations_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_raw_parts))

    sec_df = pd.DataFrame(
        [
            {
                "doc_id": s.doc_id,
                "section_id": s.section_id,
                "title": s.title,
                "text": s.text,
            }
            for s in all_sections
        ]
    )
    sec_csv = os.path.join(out_dir, "regulations_sections.csv")
    sec_df.to_csv(sec_csv, index=False)

    chunks = chunk_sections(all_sections, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks_df = pd.DataFrame(chunks)
    chunks_csv = os.path.join(out_dir, "regulation_chunks.csv")
    chunks_df.to_csv(chunks_csv, index=False)

    embedding_state, index, _ = build_index(chunks, embedding_backend, embedding_model)
    faiss.write_index(index, os.path.join(out_dir, "regulations_faiss.index"))

    questions = load_questions(qa_file)
    results: List[Dict] = []
    retrieved_rows: List[Dict] = []
    for item in questions:
        question = item["question"]
        retrieved = retrieve_top_k(
            query=question,
            embedding_state=embedding_state,
            index=index,
            chunks=chunks,
            k=top_k,
        )
        answer = keyword_extractive_answer(question, retrieved)
        answer_mode = "extractive_fallback"
        auto_flag = evaluate_answer(answer, item)

        for rank, r in enumerate(retrieved, start=1):
            retrieved_rows.append(
                {
                    "question_id": item["id"],
                    "question": question,
                    "auto_flag": auto_flag,
                    "rank": rank,
                    "chunk_id": r["chunk_id"],
                    "score": r["score"],
                    "section_id": r["section_id"],
                    "title": r["title"],
                    "text": r["text"],
                }
            )

        results.append(
            {
                "question_id": item["id"],
                "question": question,
                "retrieved_chunk_ids": json.dumps([r["chunk_id"] for r in retrieved], ensure_ascii=False),
                "answer": answer,
                "answer_mode": answer_mode,
                "auto_flag": auto_flag,
                "manual_flag": "",
                "manual_comment": "",
            }
        )

    res_df = pd.DataFrame(results)
    n_correct = int((res_df["auto_flag"] == "correct").sum())
    n_incorrect = int((res_df["auto_flag"] == "incorrect").sum())

    # Keep a small manual-review queue directly in the output CSV.
    max_manual = min(10, len(res_df))
    res_df.loc[: max_manual - 1, "manual_flag"] = "reviewed"
    res_df.loc[: max_manual - 1, "manual_comment"] = "fill_correct_or_incorrect"

    out_csv = os.path.join(out_dir, "rag_results.csv")
    res_df.to_csv(out_csv, index=False)

    retrieved_df = pd.DataFrame(retrieved_rows)
    retrieved_csv = os.path.join(out_dir, "retrieved_chunks_by_question.csv")
    retrieved_df.to_csv(retrieved_csv, index=False)

    summary = {
        "pdf_files": pdf_paths,
        "n_sections": len(all_sections),
        "n_chunks": len(chunks),
        "n_questions": len(questions),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "outputs": {
            "regulations_raw_txt": raw_path,
            "regulations_sections_csv": sec_csv,
            "chunks_csv": chunks_csv,
            "faiss_index": os.path.join(out_dir, "regulations_faiss.index"),
            "rag_results_csv": out_csv,
            "retrieved_chunks_csv": retrieved_csv,
        },
        "notes": "Baseline uses extractive fallback for answers.",
        "embedding_backend": embedding_backend,
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"auto_flag correct: {n_correct}, incorrect: {n_incorrect}")


if __name__ == "__main__":
    main()
