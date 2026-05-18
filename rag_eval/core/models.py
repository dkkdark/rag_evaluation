from __future__ import annotations

import re
from dataclasses import dataclass, field


SECTION_RE = re.compile(r"^§\s*(\d+[a-zA-Z]?)\s+(.+)$")


@dataclass
class Section:
    doc_id: str
    doc_path: str
    program_id: str
    program_name: str
    section_id: str
    title: str
    text: str


@dataclass
class Paragraph:
    paragraph_id: str
    doc_id: str
    doc_path: str
    program_id: str
    program_name: str
    section_id: str
    title: str
    paragraph_index: int
    text: str


@dataclass
class AnswerMetricResult:
    answer_accuracy_label: str
    gold_answer_overlap: float | None
    answer_gold_support: float | None
    proxy_faithfulness: float | None
    proxy_context_relevance: float | None
    answer_has_gold_substring: bool | None
    expected_answerable: bool = True
    abstained: bool = False
    abstention_correct: bool | None = None
    over_answered: bool = False
    false_refusal: bool = False
    answerability_confidence: float | None = None
    runtime_retrieval_status: str = "not_evaluated"
    runtime_retrieval_action: str = "answer"
    runtime_retrieval_reason: str = ""
    runtime_retrieval_score: float | None = None
    gold_claim_count: int = 0
    answer_claim_count: int = 0
    context_claim_recall: float | None = None
    answer_claim_recall: float | None = None
    answer_claim_precision: float | None = None
    answer_claim_f1: float | None = None
    factual_correctness_precision: float | None = None
    factual_correctness_recall: float | None = None
    factual_correctness_f1: float | None = None
    grounded_claim_ratio: float | None = None
    hallucinated_claim_ratio: float | None = None
    noise_sensitivity_relevant: float | None = None
    noise_sensitivity_irrelevant: float | None = None
    context_utilization: float | None = None
    context_entities_recall: float | None = None
    answer_entity_precision: float | None = None
    evidence_attribution_precision: float | None = None
    evidence_attribution_recall: float | None = None
    evidence_attribution_f1: float | None = None
    evidence_coverage: float | None = None
    attributed_answer_claim_count: int = 0
    attributed_gold_claim_count: int = 0
    invalid_attribution_count: int = 0
    claim_evidence_map: list[dict] = field(default_factory=list)
    unsupported_claim_count: int = 0
    missing_gold_claim_count: int = 0
    contradicted_claim_count: int = 0
    claim_diagnostic: str = "not_evaluated"
    claim_judge_used: bool = False
    claim_judge_status: str = "disabled"
    claim_judge_error: str | None = None
    claim_judge_model: str | None = None


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
    temperature: float


@dataclass
class ClaimJudgeResult:
    used: bool
    status: str
    error: str | None
    model: str | None
    metrics: dict | None = None
    raw_response: str | None = None


@dataclass
class LLMCallResult:
    answer: str | None
    used: bool
    status: str
    error: str | None
