from __future__ import annotations

import glob
import json
import os
import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

from rag_eval.models import Paragraph, SECTION_RE, Section

EMPTY_TABLE_CELL = "[EMPTY]"


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").lower()
    return normalized or "unknown"


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-zäöüß])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_toc_line(line: str) -> bool:
    if re.search(r"\.{5,}\s*\d*\s*$", line):
        return True
    if re.match(r"^[IVX]+\.\s+\S+", line) and len(line.split()) <= 6:
        return True
    if re.match(r"^Anlage(?:\(n\))?:", line) and re.search(r"\b\d+\b", line):
        return True
    return False


def is_valid_section_heading(
    line: str,
    match: re.Match[str],
    seen_first_section: bool,
    next_line: str = "",
) -> bool:
    section_number = match.group(1)
    title = match.group(2).strip()
    if is_toc_line(line):
        return False
    if not seen_first_section:
        if section_number != "1":
            return False
        if "(1)" not in f"{line} {next_line}":
            return False
    if re.match(r"^(Abs\.|Absatz|Satz)(?:\s|$)", title):
        return False
    if title and title[0].islower():
        return False
    return True


def split_paragraphs(text: str) -> List[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if parts:
        return parts
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines


def normalize_table_cell(cell: str | None) -> str:
    if cell is None:
        return EMPTY_TABLE_CELL
    normalized = clean_text(cell)
    normalized = normalized.replace("\n", " / ").strip()
    return normalized or EMPTY_TABLE_CELL


def serialize_table_rows(rows: Sequence[Sequence[str | None]], table_index: int) -> str:
    non_empty_rows = [list(row) for row in rows if any((cell or "").strip() for cell in row)]
    if not non_empty_rows:
        return ""

    max_cols = max(len(row) for row in non_empty_rows)
    lines = [f"[TABLE {table_index}]"]
    for row in non_empty_rows:
        padded = list(row) + [None] * (max_cols - len(row))
        lines.append(" | ".join(normalize_table_cell(cell) for cell in padded))
    return "\n".join(lines)


def extract_page_text_outside_tables(page, table_bboxes: Sequence[Tuple[float, float, float, float]]) -> str:
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


def infer_document_metadata(pdf_path: str, docs_root: str | None = None) -> Dict[str, str]:
    normalized_path = os.path.normpath(pdf_path)
    doc_path = os.path.relpath(normalized_path, docs_root) if docs_root else normalized_path
    doc_path = doc_path.replace(os.sep, "/")
    parent = os.path.dirname(doc_path)
    program_name = parent.split("/")[0] if parent else "root"
    return {
        "doc_id": doc_path,
        "doc_path": doc_path,
        "program_id": slugify(program_name),
        "program_name": program_name,
    }


def parse_pdf_sections(pdf_path: str, docs_root: str | None = None) -> Tuple[str, List[Section]]:
    import pdfplumber

    metadata = infer_document_metadata(pdf_path, docs_root)
    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(extract_pdf_page_content(page))

    full_text = clean_text("\n\n".join(pages))
    sections: List[Section] = []
    cur_id = "PREAMBLE"
    cur_title = "Preamble"
    buf: List[str] = []
    seen_first_section = False

    lines = full_text.splitlines()
    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            buf.append("")
            continue
        next_lines: List[str] = []
        for candidate in lines[line_index + 1 :]:
            if candidate.strip():
                next_lines.append(candidate.strip())
                if len(next_lines) >= 3:
                    break
        next_line = " ".join(next_lines)

        match = SECTION_RE.match(line)
        if match:
            if not is_valid_section_heading(line, match, seen_first_section, next_line):
                if not seen_first_section:
                    continue
                buf.append(line)
                continue
            if any(item.strip() for item in buf):
                sections.append(
                    Section(
                        doc_id=metadata["doc_id"],
                        doc_path=metadata["doc_path"],
                        program_id=metadata["program_id"],
                        program_name=metadata["program_name"],
                        section_id=cur_id,
                        title=cur_title,
                        text="\n".join(buf).strip(),
                    )
                )
            cur_id = f"§ {match.group(1)}"
            cur_title = match.group(2).strip()
            buf = []
            seen_first_section = True
        else:
            if not seen_first_section and is_toc_line(line):
                continue
            buf.append(line)

    if any(item.strip() for item in buf):
        sections.append(
            Section(
                doc_id=metadata["doc_id"],
                doc_path=metadata["doc_path"],
                program_id=metadata["program_id"],
                program_name=metadata["program_name"],
                section_id=cur_id,
                title=cur_title,
                text="\n".join(buf).strip(),
            )
        )

    return full_text, [section for section in sections if section.text]


def extract_paragraphs(sections: Sequence[Section]) -> List[Paragraph]:
    paragraphs: List[Paragraph] = []
    for section in sections:
        for idx, paragraph_text in enumerate(split_paragraphs(section.text)):
            paragraphs.append(
                Paragraph(
                    paragraph_id=f"{section.doc_id}|{section.section_id}|p{idx}",
                    doc_id=section.doc_id,
                    doc_path=section.doc_path,
                    program_id=section.program_id,
                    program_name=section.program_name,
                    section_id=section.section_id,
                    title=section.title,
                    paragraph_index=idx,
                    text=paragraph_text,
                )
            )
    return paragraphs


def load_questions(qa_path: str) -> List[Dict]:
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        items = data["questions"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Questions file should be a list or {'questions': [...]} object.")

    normalized: List[Dict] = []
    for idx, item in enumerate(items):
        if "question" not in item:
            raise ValueError(f"Question item at index {idx} is missing 'question'.")

        row = dict(item)
        row.setdefault("id", f"q{idx + 1}")
        row.setdefault("gold_answer", "")
        row.setdefault("expected_keywords", [])
        row.setdefault("program_id", "")
        row.setdefault("program_name", "")
        row.setdefault("doc_id", "")
        normalized.append(row)
    return normalized


def resolve_doc_paths(patterns: str) -> List[str]:
    matches: List[str] = []
    for raw_pattern in patterns.split(","):
        pattern = raw_pattern.strip()
        if not pattern:
            continue
        if os.path.isdir(pattern):
            found = sorted(glob.glob(os.path.join(pattern, "**", "*.pdf*"), recursive=True))
        else:
            found = sorted(glob.glob(pattern, recursive=True))
        if not found and os.path.exists(pattern):
            found = [pattern]
        matches.extend(found)

    unique_paths = sorted(dict.fromkeys(matches))
    if not unique_paths:
        raise FileNotFoundError(f"No documents matched: {patterns}")
    return unique_paths
