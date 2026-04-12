from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from math import log2
from typing import Dict, List, Optional, Sequence

from rag_eval.models import AnswerMetricResult, DiagnosticResult


# Fallback generation strategy: pick the sentence with the strongest lexical overlap
# with the question.
def keyword_extractive_answer(question: str, retrieved: Sequence[Dict]) -> str:
    q_words = set(token for token in matching_tokens(question) if len(token) > 2)

    best_sentence = ""
    best_score = -1
    for row in retrieved:
        for sentence in re.split(r"(?<=[.!?])\s+", row["text"]):
            sentence_words = set(token for token in matching_tokens(sentence) if len(token) > 2)
            overlap = len(q_words.intersection(sentence_words))
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence.strip()

    if best_sentence:
        return best_sentence
    if retrieved:
        return retrieved[0]["text"][:400]
    return "No context."


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


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


# Remove punctuation noise so comparisons depend more on content than formatting.
def normalize_for_matching(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[_/|]+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def get_spacy_lemmatizer():
    try:
        import spacy

        return spacy.load("de_core_news_sm", disable=["parser", "ner"])
    except Exception:
        return None


def fallback_german_lemma(token: str) -> str:
    if len(token) <= 4 or any(char.isdigit() for char in token):
        return token
    irregular = {
        "prüfungen": "prüfung",
        "leistungen": "leistung",
        "module": "modul",
        "modulen": "modul",
        "studiengänge": "studiengang",
        "studiengängen": "studiengang",
        "voraussetzungen": "voraussetzung",
        "kenntnisse": "kenntnis",
        "punkte": "punkt",
        "punkten": "punkt",
        "semester": "semester",
    }
    if token in irregular:
        return irregular[token]
    for suffix in ["innen", "ungen", "heiten", "keiten", "ischen", "lichem", "licher", "liche", "ungen", "ern", "en", "er", "es", "e", "n", "s"]:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def lemmatize_text(text: str) -> str:
    normalized = normalize_for_matching(text)
    if not normalized:
        return ""

    nlp = get_spacy_lemmatizer()
    if nlp is not None:
        return " ".join(
            token.lemma_.casefold()
            for token in nlp(normalized)
            if token.lemma_.strip()
        )

    return " ".join(fallback_german_lemma(token) for token in normalized.split())


# Produce reusable normalized tokens for overlap checks and keyword matching.
def matching_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", lemmatize_text(text), flags=re.UNICODE)


# Match either by normalized phrase or by token inclusion to allow mild paraphrasing.
def text_matches_keyword(text: str, keyword: str) -> bool:
    normalized_text = lemmatize_text(text)
    normalized_keyword = lemmatize_text(keyword)
    if not normalized_keyword:
        return False
    if normalized_keyword in normalized_text:
        return True

    keyword_tokens = set(matching_tokens(keyword))
    if not keyword_tokens:
        return False
    text_tokens = set(matching_tokens(text))
    return keyword_tokens.issubset(text_tokens)


# Keep only content-like tokens so overlap metrics are less sensitive to filler words.
def informative_tokens(text: str) -> List[str]:
    return [token for token in matching_tokens(text) if len(token) > 2]


# Measure what share of expected items is covered by the answer or context.
def fraction_present(items: Sequence[str], text: str) -> Optional[float]:
    if not items:
        return None
    matches = sum(1 for item in items if text_matches_keyword(text, str(item)))
    return matches / len(items)


# Use simple token overlap as a lightweight proxy for similarity and grounding.
def token_overlap_fraction(source_text: str, reference_text: str) -> Optional[float]:
    source_tokens = set(informative_tokens(source_text))
    if not source_tokens:
        return None
    reference_tokens = set(informative_tokens(reference_text))
    if not reference_tokens:
        return 0.0
    return len(source_tokens.intersection(reference_tokens)) / len(source_tokens)


def answer_new_information_support(answer: str, question: str, gold_answer: str) -> Optional[float]:
    answer_tokens = set(informative_tokens(answer))
    if not answer_tokens:
        return None
    question_tokens = set(informative_tokens(question))
    new_answer_tokens = answer_tokens.difference(question_tokens)
    if not new_answer_tokens:
        new_answer_tokens = answer_tokens
    gold_tokens = set(informative_tokens(gold_answer))
    if not gold_tokens:
        return 0.0
    return len(new_answer_tokens.intersection(gold_tokens)) / len(new_answer_tokens)


# Combine coverage and grounding signals into one practical label for evaluation.
def classify_answer(
    *,
    gold_answer_overlap: Optional[float],
    answer_gold_support: Optional[float],
    answer_has_gold_substring: Optional[bool],
    faithfulness: Optional[float],
) -> str:
    supported = faithfulness is None or faithfulness >= 0.45
    clearly_unsupported = faithfulness is not None and faithfulness < 0.35

    if answer_has_gold_substring and supported:
        return "correct"

    strong_signal = False
    partial_signal = False

    if gold_answer_overlap is not None:
        strong_signal = strong_signal or gold_answer_overlap >= 0.55
        partial_signal = partial_signal or gold_answer_overlap >= 0.35
    if answer_gold_support is not None:
        strong_signal = strong_signal or answer_gold_support >= 0.7
        partial_signal = partial_signal or answer_gold_support >= 0.55

    if clearly_unsupported and (strong_signal or partial_signal):
        return "unsupported"
    if strong_signal and supported:
        return "correct"
    if partial_signal and supported:
        return "partially_correct"
    if strong_signal or partial_signal:
        return "unsupported"
    return "incorrect"


# Compute all answer-level heuristic signals in one place so reporting stays
# consistent across experiments.
def evaluate_answer_metrics(
    item: Dict,
    answer: str,
    retrieved: Sequence[Dict],
) -> AnswerMetricResult:
    context_text = "\n".join(row["text"] for row in retrieved) # retrieved chuncks in one text
    context_relevance = token_overlap_fraction(item["question"], context_text) if context_text.strip() else None
    faithfulness = token_overlap_fraction(answer, context_text) if context_text.strip() else None
    gold_answer = str(item.get("gold_answer", "")).strip()
    answer_has_gold_substring = None
    gold_answer_overlap = None
    answer_gold_support = None
    if gold_answer:
        answer_has_gold_substring = normalize_text(gold_answer) in normalize_text(answer)
        gold_answer_overlap = token_overlap_fraction(gold_answer, answer)
        answer_gold_support = answer_new_information_support(
            answer=answer,
            question=item["question"],
            gold_answer=gold_answer,
        )

    final_label = classify_answer(
        gold_answer_overlap=gold_answer_overlap,
        answer_gold_support=answer_gold_support,
        answer_has_gold_substring=answer_has_gold_substring,
        faithfulness=faithfulness,
    )

    return AnswerMetricResult(
        answer_accuracy_label=final_label,
        gold_answer_overlap=gold_answer_overlap, # the proportion of gold answer to the actual answer
        answer_gold_support=answer_gold_support, # the proportion of the actual answer that is covered by the gold answer
        proxy_faithfulness=faithfulness, # the proportion of answer to the retrieved context
        proxy_context_relevance=context_relevance, # the proportion of question to the retrieved context
        answer_has_gold_substring=answer_has_gold_substring, # whether the actual answer fully contains the gold answer
    )


# Turn metric patterns into a simple human-readable failure reason.
def diagnose_failure(
    answer_metrics: AnswerMetricResult,
    retrieval_metrics: Dict,
    llm_status: str | None = None,
    answer_mode: str | None = None,
) -> DiagnosticResult:
    mrr_at_k = retrieval_metrics.get("mrr_at_k")
    ndcg_at_k = retrieval_metrics.get("ndcg_at_k")
    recall_at_k = retrieval_metrics.get("recall_at_k")
    ragas_recall_at_k = retrieval_metrics.get("ragas_recall_at_k")
    first_relevant_rank = retrieval_metrics.get("first_relevant_rank")

    strong_retrieval = (
        (mrr_at_k is not None and mrr_at_k >= 1.0)
        or (ndcg_at_k is not None and ndcg_at_k >= 0.75)
        or (ragas_recall_at_k is not None and ragas_recall_at_k >= 0.7)
    )
    weak_retrieval = (
        (mrr_at_k is not None and mrr_at_k == 0.0)
        or (ndcg_at_k is not None and ndcg_at_k < 0.35)
        or (ragas_recall_at_k is not None and ragas_recall_at_k < 0.35)
    )
    partial_retrieval = (
        not strong_retrieval
        and not weak_retrieval
        and (
            (recall_at_k is not None and recall_at_k < 0.2)
            or (ragas_recall_at_k is not None and ragas_recall_at_k < 0.6)
        )
    )
    llm_failed = llm_status not in {None, "success", "disabled"} and answer_mode == "extractive_fallback"

    if answer_metrics.answer_accuracy_label == "unsupported":
        if weak_retrieval:
            return DiagnosticResult(
                primary_error_reason="retrieval_and_grounding_failure",
                secondary_error_reason="generation_unsupported",
                explanation="The retrieved context was weak and the answer also contains claims that are not well supported by it.",
            )
        if llm_failed:
            return DiagnosticResult(
                primary_error_reason="extractive_fallback_unsupported",
                secondary_error_reason="llm_call_failed",
                explanation="The system fell back from the LLM to extractive answering and produced unsupported claims from partially relevant context.",
            )
        return DiagnosticResult(
            primary_error_reason="answer_unsupported",
            secondary_error_reason="none",
            explanation="The answer contains claims that are not well supported by the retrieved context.",
        )

    if answer_metrics.answer_accuracy_label == "correct":
        return DiagnosticResult(
            primary_error_reason="ok",
            secondary_error_reason="none",
            explanation="No obvious failure detected for this question.",
        )

    if (
        answer_metrics.proxy_faithfulness is not None
        and answer_metrics.proxy_faithfulness < 0.5
    ):
        if weak_retrieval:
            return DiagnosticResult(
                primary_error_reason="retrieval_miss",
                secondary_error_reason="generation_hallucination",
                explanation="The system retrieved weak context and the answer added content that is poorly grounded in it.",
            )
        if llm_failed:
            return DiagnosticResult(
                primary_error_reason="extractive_fallback_failure",
                secondary_error_reason="llm_call_failed",
                explanation="The LLM call failed, the system used extractive fallback, and the resulting answer is weakly grounded in the retrieved context.",
            )
        return DiagnosticResult(
            primary_error_reason="generation_hallucination",
            secondary_error_reason="none",
            explanation="The answer uses content that is weakly grounded in the retrieved context.",
        )

    if answer_metrics.answer_accuracy_label == "incorrect":
        if weak_retrieval:
            if first_relevant_rank is None:
                return DiagnosticResult(
                    primary_error_reason="retrieval_miss",
                    secondary_error_reason="none",
                    explanation="The system did not retrieve any clearly relevant chunk in the top-k results.",
                )
            return DiagnosticResult(
                primary_error_reason="wrong_chunks_ranked_high",
                secondary_error_reason="partial_retrieval",
                explanation="Some relevant evidence exists, but stronger or more complete chunks were not ranked high enough to support a correct answer.",
            )
        if partial_retrieval:
            return DiagnosticResult(
                primary_error_reason="partial_retrieval",
                secondary_error_reason="answer_incomplete_from_partial_context",
                explanation="The retrieved context covered only part of the needed evidence, so the final answer missed important details.",
            )
        if strong_retrieval and llm_failed:
            return DiagnosticResult(
                primary_error_reason="llm_call_failed",
                secondary_error_reason="extractive_fallback_failure",
                explanation="Relevant context was retrieved, but the LLM did not produce an answer and the extractive fallback selected an incomplete or misleading sentence.",
            )
        if strong_retrieval:
            return DiagnosticResult(
                primary_error_reason="answer_synthesis_failure",
                secondary_error_reason="none",
                explanation="Relevant context was retrieved, but the answer generator failed to combine the evidence into a correct final answer.",
            )
        return DiagnosticResult(
            primary_error_reason="answer_incorrect",
            secondary_error_reason="none",
            explanation="The answer remains incorrect even though retrieval evidence was partially available.",
        )

    if answer_metrics.answer_accuracy_label == "partially_correct":
        if partial_retrieval or weak_retrieval:
            return DiagnosticResult(
                primary_error_reason="answer_incomplete",
                secondary_error_reason="partial_retrieval",
                explanation="The answer is partly correct, but the retrieved context did not cover enough of the needed evidence.",
            )
        if llm_failed:
            return DiagnosticResult(
                primary_error_reason="answer_incomplete",
                secondary_error_reason="llm_call_failed",
                explanation="The answer is partly correct, but the LLM failed and the extractive fallback did not cover all important facts.",
            )
        return DiagnosticResult(
            primary_error_reason="answer_incomplete_from_good_context",
            secondary_error_reason="none",
            explanation="The answer is mostly grounded, but it does not cover all important facts even though the retrieved context was good enough.",
        )

    return DiagnosticResult(
        primary_error_reason="ok",
        secondary_error_reason="none",
        explanation="No obvious failure detected for this question.",
    )


# Average one metric over rows while safely skipping missing values.
def summarize_average(metric_rows: Sequence[Dict], key: str) -> Optional[float]:
    values = [float(row[key]) for row in metric_rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


# Aggregate answer metrics into a compact experiment-level summary.
def summarize_answer_metrics(metric_rows: Sequence[Dict]) -> Dict:
    return {
        "mean_gold_answer_overlap": summarize_average(metric_rows, "gold_answer_overlap"),
        "mean_answer_gold_support": summarize_average(metric_rows, "answer_gold_support"),
        "mean_proxy_faithfulness": summarize_average(metric_rows, "proxy_faithfulness"),
        "mean_proxy_context_relevance": summarize_average(metric_rows, "proxy_context_relevance"),
        "n_correct": sum(1 for row in metric_rows if row["answer_accuracy_label"] == "correct"),
        "n_partially_correct": sum(
            1 for row in metric_rows if row["answer_accuracy_label"] == "partially_correct"
        ),
        "n_incorrect": sum(1 for row in metric_rows if row["answer_accuracy_label"] == "incorrect"),
        "n_unsupported": sum(1 for row in metric_rows if row["answer_accuracy_label"] == "unsupported"),
        "n_needs_manual_review": sum(
            1 for row in metric_rows if row["answer_accuracy_label"] == "needs_manual_review"
        ),
    }


def split_reference_units(text: str) -> List[str]:
    if not text.strip():
        return []
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+|[-•]\s+)", text)
    units: List[str] = []
    for part in parts:
        normalized = normalize_text(part)
        if len(normalized) >= 8:
            units.append(part.strip())
    return list(dict.fromkeys(unit for unit in units if unit))


def retrieval_keyword_match_count(expected_keywords: Sequence[str], text: str) -> int:
    return sum(1 for keyword in expected_keywords if text_matches_keyword(text, str(keyword)))


def weak_chunk_relevance_score(item: Dict, chunk_text: str) -> float:
    expected_keywords = item.get("expected_keywords", [])
    keyword_coverage = fraction_present(expected_keywords, chunk_text) or 0.0
    gold_answer = str(item.get("gold_answer", "")).strip()
    gold_overlap = token_overlap_fraction(gold_answer, chunk_text) or 0.0 if gold_answer else 0.0
    return max(keyword_coverage, gold_overlap)


def weak_chunk_relevance_grade(item: Dict, chunk_text: str) -> int:
    expected_keywords = item.get("expected_keywords", [])
    keyword_matches = retrieval_keyword_match_count(expected_keywords, chunk_text)
    keyword_coverage = fraction_present(expected_keywords, chunk_text) or 0.0
    gold_answer = str(item.get("gold_answer", "")).strip()
    gold_overlap = token_overlap_fraction(gold_answer, chunk_text) or 0.0 if gold_answer else 0.0

    if keyword_coverage >= 0.6 or gold_overlap >= 0.6:
        return 3
    if keyword_coverage >= 0.35 or gold_overlap >= 0.35 or keyword_matches >= 3:
        return 2
    if keyword_coverage >= 0.15 or gold_overlap >= 0.15 or keyword_matches >= 1:
        return 1
    return 0


def dcg_at_k(grades: Sequence[int], k: int) -> float:
    total = 0.0
    for rank, grade in enumerate(grades[:k], start=1):
        total += (2**grade - 1) / log2(rank + 1)
    return total


def summarize_retrieval_metrics(metric_rows: Sequence[Dict]) -> Dict:
    return {
        "mean_mrr_at_k": summarize_average(metric_rows, "mrr_at_k"),
        "mean_ndcg_at_k": summarize_average(metric_rows, "ndcg_at_k"),
        "mean_recall_at_k": summarize_average(metric_rows, "recall_at_k"),
        "mean_ragas_recall_at_k": summarize_average(metric_rows, "ragas_recall_at_k"),
        "questions_with_relevant_chunk": sum(
            1 for row in metric_rows if row.get("first_relevant_rank") is not None
        ),
        "questions_with_target_doc_at_k": sum(
            1 for row in metric_rows if row.get("target_doc_retrieved_at_k") is True
        ),
    }


def evaluate_retrieval_metrics(
    item: Dict,
    retrieved: Sequence[Dict],
    candidate_chunks: Sequence[Dict],
    k: int,
) -> Dict:
    reference_units = list(item.get("expected_keywords", []))
    reference_units.extend(split_reference_units(str(item.get("gold_answer", ""))))
    reference_units = list(dict.fromkeys(unit for unit in reference_units if str(unit).strip()))

    retrieved_context = "\n".join(row["text"] for row in retrieved)
    ragas_recall_at_k = fraction_present(reference_units, retrieved_context)

    def scoped_grade(row: Dict) -> int:
        target_doc_id = item.get("doc_id")
        if target_doc_id and not metadata_value_matches(row.get("doc_id", ""), target_doc_id):
            return 0
        return weak_chunk_relevance_grade(item, row["text"])

    retrieved_grades = [scoped_grade(row) for row in retrieved[:k]]
    all_candidate_grades = [scoped_grade(row) for row in candidate_chunks]
    target_doc_id = item.get("doc_id")
    target_doc_ranks = [
        rank
        for rank, row in enumerate(retrieved[:k], start=1)
        if target_doc_id and metadata_value_matches(row.get("doc_id", ""), target_doc_id)
    ]

    first_relevant_rank = next(
        (rank for rank, grade in enumerate(retrieved_grades, start=1) if grade > 0),
        None,
    )
    mrr_at_k = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0

    ideal_grades = sorted(all_candidate_grades, reverse=True)
    actual_dcg = dcg_at_k(retrieved_grades, k)
    ideal_dcg = dcg_at_k(ideal_grades, k)
    ndcg_at_k = (actual_dcg / ideal_dcg) if ideal_dcg > 0 else None

    relevant_chunk_count = sum(1 for grade in all_candidate_grades if grade > 0)
    retrieved_relevant_count = sum(1 for grade in retrieved_grades if grade > 0)
    recall_at_k = (
        retrieved_relevant_count / relevant_chunk_count if relevant_chunk_count > 0 else None
    )

    return {
        "mrr_at_k": mrr_at_k,  # reciprocal rank of the first relevant retrieved chunk within top-k. Relevant chunk in 1st place → 1.0
        "ndcg_at_k": ndcg_at_k,  # ranking quality at top-k, rewarding highly relevant chunks near the top
        "recall_at_k": recall_at_k,  # relevant retrieved chunks / all relevant chunks in the candidate pool
        "ragas_recall_at_k": ragas_recall_at_k,  # reference facts from gold_answer+keywords covered by retrieved top-k context
        "first_relevant_rank": first_relevant_rank,  # 1-based position of the first relevant retrieved chunk
        "n_relevant_chunks": relevant_chunk_count,  # how many chunks in the full candidate pool are treated as relevant
        "n_retrieved_relevant_chunks": retrieved_relevant_count,  # how many relevant chunks appear inside retrieved top-k
        "target_doc_retrieved_at_k": bool(target_doc_ranks) if target_doc_id else None,  # whether top-k contains any chunk from the hidden target doc_id
        "first_target_doc_rank": target_doc_ranks[0] if target_doc_ranks else None,  # first top-k rank from the hidden target doc_id
        "n_retrieved_target_doc_chunks": len(target_doc_ranks) if target_doc_id else None,  # top-k chunks from the hidden target doc_id
    }


# Count failure categories to show which error mode dominates an experiment.
def summarize_diagnostics(metric_rows: Sequence[Dict]) -> Dict:
    counts: Dict[str, int] = {}
    for row in metric_rows:
        counts[row["primary_error_reason"]] = counts.get(row["primary_error_reason"], 0) + 1
    return {
        "counts_by_primary_reason": counts,
        "most_common_reason": max(counts, key=counts.get) if counts else None,
    }


# Replace missing metrics with zero so ranking code stays simple and stable.
def safe_metric(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(value)


# Convert summary metrics into one ranking score for experiment comparison.
def score_experiment(summary: Dict, weights: Dict[str, float]) -> Dict:
    answer_score = safe_metric(summary["answer_metrics"].get("mean_proxy_faithfulness"))
    correctness_score = 0.0
    if summary["n_questions"] > 0:
        correctness_score = (
            summary["n_correct"] + 0.5 * summary["answer_metrics"].get("n_partially_correct", 0)
        ) / summary["n_questions"]
    retrieval_score = (
        0.4 * safe_metric(summary.get("retrieval_metrics", {}).get("mean_ndcg_at_k"))
        + 0.3 * safe_metric(summary.get("retrieval_metrics", {}).get("mean_mrr_at_k"))
        + 0.3 * safe_metric(summary.get("retrieval_metrics", {}).get("mean_ragas_recall_at_k"))
    )

    weighted_total = (
        answer_score * weights["answer"]
        + correctness_score * weights["correctness"]
        + retrieval_score * weights["retrieval"]
    )

    return {
        "answer_score": answer_score,
        "correctness_score": correctness_score,
        "retrieval_score": retrieval_score,
        "weighted_total": weighted_total,
    }


# Create a short text recommendation for the best-performing experiment setup.
def build_recommendation(best_summary: Dict, best_score: Dict) -> str:
    return (
        f"Best config: chunking={best_summary['chunking_strategy']}, "
        f"retriever={best_summary['retriever']}, "
        f"chunk_size={best_summary['chunk_size']}, "
        f"overlap={best_summary['chunk_overlap']}, "
        f"top_k={best_summary['top_k']}. "
        f"It leads on answer={best_score['answer_score']:.3f}, "
        f"retrieval={best_score['retrieval_score']:.3f} "
        f"with total={best_score['weighted_total']:.3f}."
    )


# Sort experiments by the combined score so we can pick one default configuration.
def rank_experiments(experiment_summaries: Sequence[Dict], weights: Dict[str, float]) -> List[Dict]:
    ranked: List[Dict] = []
    for summary in experiment_summaries:
        score = score_experiment(summary, weights)
        ranked.append(
            {
                "experiment": summary["experiment"],
                "chunking_strategy": summary["chunking_strategy"],
                "retriever": summary["retriever"],
                "chunk_size": summary["chunk_size"],
                "chunk_overlap": summary["chunk_overlap"],
                "top_k": summary["top_k"],
                "n_correct": summary["n_correct"],
                "n_questions": summary["n_questions"],
                "mean_proxy_faithfulness": summary["answer_metrics"].get("mean_proxy_faithfulness"),
                "mean_proxy_context_relevance": summary["answer_metrics"].get(
                    "mean_proxy_context_relevance"
                ),
                "mean_mrr_at_k": summary.get("retrieval_metrics", {}).get("mean_mrr_at_k"),
                "mean_ndcg_at_k": summary.get("retrieval_metrics", {}).get("mean_ndcg_at_k"),
                "mean_recall_at_k": summary.get("retrieval_metrics", {}).get("mean_recall_at_k"),
                "mean_ragas_recall_at_k": summary.get("retrieval_metrics", {}).get(
                    "mean_ragas_recall_at_k"
                ),
                "most_common_error_reason": summary["diagnostics"].get("most_common_reason"),
                "score_answer": score["answer_score"],
                "score_correctness": score["correctness_score"],
                "score_retrieval": score["retrieval_score"],
                "score_total": score["weighted_total"],
            }
        )

    ranked.sort(
        key=lambda row: (
            row["score_total"],
            row["n_correct"],
        ),
        reverse=True,
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked
