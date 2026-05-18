from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

from rag_eval.core.models import Paragraph, Section


# Use a simple whitespace tokenizer so chunking stays offline and predictable.
def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\S+", text)


# Split long texts into overlapping windows to balance recall and context continuity.
def make_fixed_chunks(
    texts: Iterable[Dict[str, str]],
    chunk_size: int,
    chunk_overlap: int,
    strategy_name: str,
) -> List[Dict]:
    chunks: List[Dict] = []
    for item in texts:
        words = tokenize_words(item["text"])
        if not words:
            continue

        start = 0
        chunk_index = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk_words = words[start:end]
            chunks.append(
                {
                    "chunk_id": f"{item['source_id']}|c{chunk_index}",
                    "doc_id": item["doc_id"],
                    "doc_path": item["doc_path"],
                    "program_id": item["program_id"],
                    "program_name": item["program_name"],
                    "section_id": item["section_id"],
                    "title": item["title"],
                    "text": " ".join(chunk_words),
                    "n_words": len(chunk_words),
                    "chunk_index": chunk_index,
                    "chunking_strategy": strategy_name,
                    "source_type": item["source_type"],
                    "source_id": item["source_id"],
                    "start_word": start,
                    "end_word": end,
                }
            )
            if end >= len(words):
                break
            start = max(end - chunk_overlap, start + 1)
            chunk_index += 1
    return chunks


# Keep each structural section as one chunk when section boundaries matter most.
def chunk_by_section(sections: Sequence[Section]) -> List[Dict]:
    chunks: List[Dict] = []
    for section in sections:
        words = tokenize_words(section.text)
        chunks.append(
            {
                "chunk_id": f"{section.doc_id}|{section.section_id}|section",
                "doc_id": section.doc_id,
                "doc_path": section.doc_path,
                "program_id": section.program_id,
                "program_name": section.program_name,
                "section_id": section.section_id,
                "title": section.title,
                "text": section.text,
                "n_words": len(words),
                "chunk_index": 0,
                "chunking_strategy": "by_section",
                "source_type": "section",
                "source_id": f"{section.doc_id}|{section.section_id}",
                "start_word": 0,
                "end_word": len(words),
            }
        )
    return chunks


# Keep paragraph boundaries when we want smaller and more focused retrieval units.
def chunk_by_paragraph(paragraphs: Sequence[Paragraph]) -> List[Dict]:
    chunks: List[Dict] = []
    for paragraph in paragraphs:
        words = tokenize_words(paragraph.text)
        chunks.append(
            {
                "chunk_id": f"{paragraph.paragraph_id}|paragraph",
                "doc_id": paragraph.doc_id,
                "doc_path": paragraph.doc_path,
                "program_id": paragraph.program_id,
                "program_name": paragraph.program_name,
                "section_id": paragraph.section_id,
                "title": paragraph.title,
                "text": paragraph.text,
                "n_words": len(words),
                "chunk_index": paragraph.paragraph_index,
                "chunking_strategy": "by_paragraph",
                "source_type": "paragraph",
                "source_id": paragraph.paragraph_id,
                "start_word": 0,
                "end_word": len(words),
            }
        )
    return chunks


def build_chunks(
    sections: Sequence[Section],
    paragraphs: Sequence[Paragraph],
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict]:
    # The document is cut into pieces of equal size by words.
    if strategy == "fixed_words":
        source_rows = [
            {
                "source_id": f"{section.doc_id}|{section.section_id}",
                "doc_id": section.doc_id,
                "doc_path": section.doc_path,
                "program_id": section.program_id,
                "program_name": section.program_name,
                "section_id": section.section_id,
                "title": section.title,
                "text": section.text,
                "source_type": "section",
            }
            for section in sections
        ]
        return make_fixed_chunks(source_rows, chunk_size, chunk_overlap, strategy)

    # The document is cut into pieces from paragraphs
    if strategy == "fixed_tokens":
        source_rows = [
            {
                "source_id": paragraph.paragraph_id,
                "doc_id": paragraph.doc_id,
                "doc_path": paragraph.doc_path,
                "program_id": paragraph.program_id,
                "program_name": paragraph.program_name,
                "section_id": paragraph.section_id,
                "title": paragraph.title,
                "text": paragraph.text,
                "source_type": "paragraph",
            }
            for paragraph in paragraphs
        ]
        return make_fixed_chunks(source_rows, chunk_size, chunk_overlap, strategy)

    # The document is cut into pieces from sections
    if strategy == "by_section":
        return chunk_by_section(sections)

    # The document is cut into pieces from paragraphs
    if strategy == "by_paragraph":
        return chunk_by_paragraph(paragraphs)

    raise ValueError(f"Unsupported chunking strategy: {strategy}")
