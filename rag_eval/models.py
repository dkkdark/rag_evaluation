from __future__ import annotations

import re
from dataclasses import dataclass


SECTION_RE = re.compile(r"^§\s*(\d+[a-zA-Z]?)\s+(.+)$")


@dataclass
class Section:
    doc_id: str
    section_id: str
    title: str
    text: str


@dataclass
class Paragraph:
    paragraph_id: str
    doc_id: str
    section_id: str
    title: str
    paragraph_index: int
    text: str

@dataclass
class AnswerMetricResult:
    answer_accuracy_label: str
    answer_keyword_coverage: float | None
    gold_answer_overlap: float | None
    proxy_faithfulness: float | None
    proxy_context_relevance: float | None
    answer_has_gold_substring: bool | None


@dataclass
class DiagnosticResult:
    primary_error_reason: str
    secondary_error_reason: str
    explanation: str


@dataclass
class LLMConfig:
    enabled: bool
    model: str
    api_key_env: str


@dataclass
class LLMCallResult:
    answer: str | None
    used: bool
    status: str
    error: str | None
