from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from math import log2
from typing import Dict, List, Optional, Sequence

from rag_eval.core.models import AnswerMetricResult, ClaimJudgeResult, DiagnosticResult
from rag_eval.core.text_utils import metadata_value_matches


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
        "semestern": "semester",
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


REFUSAL_PATTERNS = [
    r"\bnot enough (?:information|context|evidence)\b",
    r"\binsufficient (?:information|context|evidence)\b",
    r"\bcannot (?:answer|determine|infer)\b",
    r"\bcan't (?:answer|determine|infer)\b",
    r"\bno (?:relevant )?(?:information|evidence)\b",
    r"\bnot provided\b",
    r"\bnot explicitly mentioned\b",
    r"\bnot explicitly specified\b",
    r"\bnot explicitly stated\b",
    r"\bno explicit(?:ly)? (?:information|indication|answer|mention|specification|duration)\b",
    r"\bnot indicated\b",
    r"\bnot given\b",
    r"\bnot answered\b",
    r"\bcannot be answered\b",
    r"\bnicht genug\b",
    r"\bkeine ausreichenden\b",
    r"\bkeine explizite angabe\b",
    r"\bkeine angabe(?:n)?\b",
    r"\bkeine information(?:en)?\b",
    r"\bnicht (?:aus dem|im) kontext\b",
    r"\baus dem kontext (?:geht|ergibt) .* nicht\b",
    r"\bnicht explizit genannt\b",
    r"\bnicht explizit angegeben\b",
    r"\bnicht angegeben\b",
    r"\bnicht beantwortet werden\b",
    r"\bgenaue .* nicht genannt\b",
]

MIN_RELEVANT_GRADE = 2

RUNTIME_STOPWORDS = {
    "the", "and", "for", "with", "from", "what", "when", "where", "which", "who",
    "how", "many", "much", "next", "after", "before", "this", "that", "was", "were",
    "ist", "sind", "der", "die", "das", "den", "dem", "des", "und", "oder", "mit",
    "von", "vom", "zur", "zum", "welche", "welcher", "welches", "wann", "wo", "wie",
    "nach", "vor", "für", "eine", "einer", "eines", "ein", "im", "in", "am", "an",
}


def parse_bool(value: object, *, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y", "ja", "answerable"}:
        return True
    if normalized in {"false", "0", "no", "n", "nein", "unanswerable"}:
        return False
    return default


def expected_answerable(item: Dict) -> bool:
    if "answerable" in item:
        return parse_bool(item.get("answerable"), default=True)
    if "is_answerable" in item:
        return parse_bool(item.get("is_answerable"), default=True)
    if "unanswerable" in item:
        return not parse_bool(item.get("unanswerable"), default=False)
    return True


def answer_is_refusal(answer: str) -> bool:
    normalized = normalize_text(answer)
    return any(re.search(pattern, normalized) for pattern in REFUSAL_PATTERNS)


def runtime_content_tokens(text: str) -> set[str]:
    return {
        token
        for token in informative_tokens(text)
        if token not in RUNTIME_STOPWORDS and len(token) > 3
    }


def runtime_query_context_relevance(question: str, context_text: str) -> float:
    query_tokens = runtime_content_tokens(question)
    if not query_tokens:
        return 0.0
    context_tokens = runtime_content_tokens(context_text)
    if not context_tokens:
        return 0.0
    return len(query_tokens.intersection(context_tokens)) / len(query_tokens)


# If the evidence is good, you can respond.
# If the evidence is weak, it's better not to respond right away.
# If there's almost no evidence, it's better to abstain or retrieve more.
def runtime_retrieval_evaluation(
    *,
    question: str,
    retrieved: Sequence[Dict],
    min_context_relevance: float = 0.18,
    min_top_score: float = 0.08,
) -> Dict[str, object]:
    if not retrieved:
        return {
            "status": "missing_evidence",
            "action": "abstain",
            "reason": "No chunks were retrieved.",
            "score": 0.0,
            "context_relevance": None,
            "top_score": None,
        }

    context_text = "\n".join(str(row.get("text", "")) for row in retrieved)
    context_relevance = runtime_query_context_relevance(question, context_text)
    top_score = max(float(row.get("score") or 0.0) for row in retrieved)
    score = max(context_relevance, min(top_score / max(min_top_score, 1e-9), 1.0) * 0.5)

    if context_relevance >= min_context_relevance or (
        top_score >= min_top_score and context_relevance >= min_context_relevance * 0.6
    ):
        return {
            "status": "good_evidence",
            "action": "answer",
            "reason": "Retrieved context has enough lexical/query overlap or retrieval score for answer generation.",
            "score": min(max(score, 0.0), 1.0),
            "context_relevance": context_relevance,
            "top_score": top_score,
        }
    if context_relevance >= min_context_relevance * 0.6 or (
        top_score >= min_top_score * 0.6 and context_relevance > 0.0
    ):
        return {
            "status": "weak_evidence",
            "action": "retrieve_more_or_rewrite",
            "reason": "Retrieved context is weak; a production pipeline should rewrite the query or retrieve more before answering.",
            "score": min(max(score, 0.0), 1.0),
            "context_relevance": context_relevance,
            "top_score": top_score,
        }
    return {
        "status": "missing_evidence",
        "action": "abstain",
        "reason": "Retrieved context appears insufficient for grounded answer generation.",
        "score": min(max(score, 0.0), 1.0),
        "context_relevance": context_relevance,
        "top_score": top_score,
    }


def split_claims(text: str) -> List[str]:
    if not text.strip():
        return []
    normalized = text.replace(";", ". ").replace("\n", ". ")
    parts = re.split(r"(?<=[.!?])\s+|(?:\s+-\s+)|(?:\s+•\s+)", normalized)
    claims: List[str] = []
    for part in parts:
        part = part.strip(" -•\t\r\n")
        if not part:
            continue
        for subpart in re.split(r"\s+(?:und|oder|and|or)\s+(?=[A-ZÄÖÜ0-9])", part):
            subpart = subpart.strip(" ,")
            if len(informative_tokens(subpart)) >= 2:
                claims.append(subpart)
    return list(dict.fromkeys(claims))


def reference_claims_from_item(item: Dict) -> List[str]:
    claims = split_claims(str(item.get("gold_answer", "")))
    if claims:
        return list(dict.fromkeys(claims))
    for keyword in item.get("expected_keywords", []):
        keyword_text = str(keyword).strip()
        if keyword_text and len(informative_tokens(keyword_text)) >= 1:
            claims.append(keyword_text)
    return list(dict.fromkeys(claims))


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", normalize_text(text)))


def claim_support_score(claim: str, evidence_text: str) -> float:
    claim_tokens = set(informative_tokens(claim))
    if not claim_tokens:
        return 0.0
    evidence_tokens = set(informative_tokens(evidence_text))
    if not evidence_tokens:
        return 0.0
    token_coverage = len(claim_tokens.intersection(evidence_tokens)) / len(claim_tokens)
    if text_matches_keyword(evidence_text, claim):
        return max(token_coverage, 0.95)
    claim_numbers = numeric_tokens(claim)
    if claim_numbers and not claim_numbers.issubset(numeric_tokens(evidence_text)):
        token_coverage *= 0.65
    return token_coverage


def claim_is_supported(claim: str, evidence_text: str, *, threshold: float = 0.62) -> bool:
    return claim_support_score(claim, evidence_text) >= threshold


def claim_is_contradicted(claim: str, evidence_text: str) -> bool:
    claim_numbers = numeric_tokens(claim)
    evidence_numbers = numeric_tokens(evidence_text)
    if not claim_numbers or not evidence_numbers or claim_numbers.intersection(evidence_numbers):
        return False

    claim_terms = set(informative_tokens(claim)).difference(claim_numbers)
    evidence_terms = set(informative_tokens(evidence_text)).difference(evidence_numbers)
    if not claim_terms:
        return False
    return len(claim_terms.intersection(evidence_terms)) / len(claim_terms) >= 0.55


def supporting_chunks_for_claim(
    claim: str,
    retrieved: Sequence[Dict],
    *,
    threshold: float = 0.62,
) -> List[Dict[str, object]]:
    supports: List[Dict[str, object]] = []
    for rank, row in enumerate(retrieved, start=1):
        score = claim_support_score(claim, str(row.get("text", "")))
        if score >= threshold:
            supports.append(
                {
                    "chunk_id": row.get("chunk_id", ""),
                    "rank": rank,
                    "support_score": score,
                    "section_id": row.get("section_id", ""),
                    "title": row.get("title", ""),
                }
            )
    return supports


def support_ids_from_entries(entries: Sequence[Dict[str, object]]) -> List[str]:
    ids: List[str] = []
    for entry in entries:
        chunk_id = str(entry.get("chunk_id", "")).strip()
        if chunk_id:
            ids.append(chunk_id)
    return list(dict.fromkeys(ids))


def compute_attribution_metrics(
    claim_evidence_map: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    answer_claims = [row for row in claim_evidence_map if row.get("claim_type") == "answer"]
    gold_claims = [row for row in claim_evidence_map if row.get("claim_type") == "gold"]
    attributed_answer_claims = [
        row for row in answer_claims if row.get("context_nli") == "supported" and row.get("supporting_chunk_ids")
    ]
    attributed_gold_claims = [
        row for row in gold_claims if row.get("context_nli") == "supported" and row.get("supporting_chunk_ids")
    ]
    answer_supported_count = sum(1 for row in answer_claims if row.get("context_nli") == "supported")
    gold_supported_count = sum(1 for row in gold_claims if row.get("context_nli") == "supported")
    invalid_count = sum(int(row.get("invalid_attribution_count") or 0) for row in claim_evidence_map)

    precision = safe_ratio(len(attributed_answer_claims), len(answer_claims))
    recall = safe_ratio(len(attributed_gold_claims), len(gold_claims))
    attribution_f1 = harmonic_mean(precision, recall)
    evidence_coverage = safe_ratio(gold_supported_count, len(gold_claims))
    answer_support_attribution_rate = safe_ratio(len(attributed_answer_claims), answer_supported_count)
    gold_support_attribution_rate = safe_ratio(len(attributed_gold_claims), gold_supported_count)

    return {
        "evidence_attribution_precision": precision,
        "evidence_attribution_recall": recall,
        "evidence_attribution_f1": attribution_f1,
        "evidence_coverage": evidence_coverage,
        "answer_support_attribution_rate": answer_support_attribution_rate,
        "gold_support_attribution_rate": gold_support_attribution_rate,
        "attributed_answer_claim_count": len(attributed_answer_claims),
        "attributed_gold_claim_count": len(attributed_gold_claims),
        "invalid_attribution_count": invalid_count,
    }


def extract_proxy_entities(text: str) -> List[str]:
    if not text.strip():
        return []

    entities: List[str] = []
    section_refs = re.findall(r"§\s*\d+[a-zA-Z]?", text)
    entities.extend(section_refs)

    capitalized_phrases = re.findall(
        r"\b(?:[A-ZÄÖÜ][\w.-]{2,}(?:\s+[A-ZÄÖÜ][\w.-]{2,})*)\b",
        text,
        flags=re.UNICODE,
    )
    entities.extend(capitalized_phrases)

    for number in numeric_tokens(text):
        entities.append(number)

    normalized = [normalize_for_matching(entity) for entity in entities if entity.strip()]
    filtered = [entity for entity in normalized if len(entity) >= 2]
    return list(dict.fromkeys(filtered))


def compute_entity_metrics(
    *,
    item: Dict,
    answer: str,
    context_text: str,
) -> Dict[str, Optional[float]]:
    reference_entities = set(extract_proxy_entities(str(item.get("gold_answer", ""))))
    for keyword in item.get("expected_keywords", []):
        reference_entities.update(extract_proxy_entities(str(keyword)))

    if not reference_entities:
        return {
            "context_entities_recall": None,
            "answer_entity_precision": None,
        }

    context_entities = set(extract_proxy_entities(context_text))
    answer_entities = set(extract_proxy_entities(answer))

    context_entities_recall = len(reference_entities.intersection(context_entities)) / len(reference_entities)
    answer_entity_precision = (
        len(reference_entities.intersection(answer_entities)) / len(answer_entities)
        if answer_entities
        else None
    )
    return {
        "context_entities_recall": context_entities_recall,
        "answer_entity_precision": answer_entity_precision,
    }


def compute_noise_sensitivity_metrics(
    claim_evidence_map: Sequence[Dict[str, object]],
    *,
    answer_claim_count: int,
    unsupported_claim_count: int,
    contradicted_claim_count: int,
) -> Dict[str, Optional[float]]:
    if answer_claim_count <= 0:
        return {
            "noise_sensitivity_relevant": None,
            "noise_sensitivity_irrelevant": None,
        }

    irrelevant_noise_claims = 0
    for row in claim_evidence_map:
        if row.get("claim_type") != "answer":
            continue
        if row.get("context_nli") == "supported" and row.get("gold_nli") != "supported":
            irrelevant_noise_claims += 1

    relevant_noise_claims = min(
        answer_claim_count,
        max(unsupported_claim_count, 0) + max(contradicted_claim_count, 0),
    )
    return {
        "noise_sensitivity_relevant": relevant_noise_claims / answer_claim_count,
        "noise_sensitivity_irrelevant": irrelevant_noise_claims / answer_claim_count,
    }


def safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def harmonic_mean(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def classify_claim_diagnostic(
    *,
    gold_claim_count: int,
    answer_claim_count: int,
    context_claim_recall: Optional[float],
    answer_claim_recall: Optional[float],
    answer_claim_precision: Optional[float],
    grounded_claim_ratio: Optional[float],
    hallucinated_claim_ratio: Optional[float],
    context_utilization: Optional[float],
    contradicted_claim_count: int,
) -> str:
    if gold_claim_count == 0 and answer_claim_count == 0:
        return "no_claims_to_evaluate"
    if gold_claim_count > 0 and (context_claim_recall or 0.0) < 0.35:
        return "retrieval_claim_miss"
    if gold_claim_count > 0 and (context_claim_recall or 0.0) < 0.7:
        return "partial_evidence_retrieved"
    if contradicted_claim_count > 0:
        return "claim_contradiction"
    if (hallucinated_claim_ratio or 0.0) >= 0.35 or (grounded_claim_ratio is not None and grounded_claim_ratio < 0.65):
        return "unsupported_generated_claims"
    if gold_claim_count > 0 and (answer_claim_recall or 0.0) < 0.7:
        if context_utilization is not None and context_utilization < 0.7:
            return "answer_incomplete_from_available_context"
        return "answer_incomplete"
    if answer_claim_precision is not None and answer_claim_precision < 0.75:
        return "answer_contains_extra_claims"
    return "claim_level_ok"


def evaluate_claim_metrics(
    item: Dict,
    answer: str,
    context_text: str,
    retrieved: Sequence[Dict] | None = None,
) -> Dict[str, object]:
    gold_claims = reference_claims_from_item(item)
    answer_claims = split_claims(answer)
    retrieved_rows = list(retrieved or [])

    gold_supported_by_context = [
        claim for claim in gold_claims if claim_is_supported(claim, context_text)
    ]
    gold_present_in_answer = [
        claim for claim in gold_claims if claim_is_supported(claim, answer, threshold=0.55)
    ]
    answer_supported_by_gold = [
        claim
        for claim in answer_claims
        if any(claim_is_supported(claim, gold_claim, threshold=0.55) for gold_claim in gold_claims)
    ]
    answer_supported_by_context = [
        claim for claim in answer_claims if claim_is_supported(claim, context_text)
    ]
    answer_contradicted_by_context = [
        claim for claim in answer_claims if claim_is_contradicted(claim, context_text)
    ]
    claim_evidence_map: List[Dict[str, object]] = []
    for claim in gold_claims:
        supporting_entries = supporting_chunks_for_claim(claim, retrieved_rows)
        claim_evidence_map.append(
            {
                "claim_type": "gold",
                "claim": claim,
                "context_nli": "supported" if supporting_entries else "not_enough_info",
                "answer_nli": "supported" if claim_is_supported(claim, answer, threshold=0.55) else "not_enough_info",
                "supporting_chunk_ids": support_ids_from_entries(supporting_entries),
                "supporting_chunks": supporting_entries,
                "invalid_attribution_count": 0,
            }
        )
    for claim in answer_claims:
        supporting_entries = supporting_chunks_for_claim(claim, retrieved_rows)
        claim_evidence_map.append(
            {
                "claim_type": "answer",
                "claim": claim,
                "context_nli": "supported" if supporting_entries else "not_enough_info",
                "gold_nli": (
                    "supported"
                    if any(claim_is_supported(claim, gold_claim, threshold=0.55) for gold_claim in gold_claims)
                    else "not_enough_info"
                ),
                "supporting_chunk_ids": support_ids_from_entries(supporting_entries),
                "supporting_chunks": supporting_entries,
                "invalid_attribution_count": 0,
            }
        )
    attribution_metrics = compute_attribution_metrics(claim_evidence_map)

    context_claim_recall = safe_ratio(len(gold_supported_by_context), len(gold_claims))
    answer_claim_recall = safe_ratio(len(gold_present_in_answer), len(gold_claims))
    answer_claim_precision = safe_ratio(len(answer_supported_by_gold), len(answer_claims))
    grounded_claim_ratio = safe_ratio(len(answer_supported_by_context), len(answer_claims))
    hallucinated_claim_ratio = safe_ratio(
        len(answer_claims) - len(answer_supported_by_context),
        len(answer_claims),
    )
    context_utilization = safe_ratio(
        len(
            [
                claim
                for claim in gold_supported_by_context
                if claim_is_supported(claim, answer, threshold=0.55)
            ]
        ),
        len(gold_supported_by_context),
    )
    answer_claim_f1 = harmonic_mean(answer_claim_precision, answer_claim_recall)
    claim_diagnostic = classify_claim_diagnostic(
        gold_claim_count=len(gold_claims),
        answer_claim_count=len(answer_claims),
        context_claim_recall=context_claim_recall,
        answer_claim_recall=answer_claim_recall,
        answer_claim_precision=answer_claim_precision,
        grounded_claim_ratio=grounded_claim_ratio,
        hallucinated_claim_ratio=hallucinated_claim_ratio,
        context_utilization=context_utilization,
        contradicted_claim_count=len(answer_contradicted_by_context),
    )

    return {
        "gold_claims": gold_claims,
        "answer_claims": answer_claims,
        "gold_claim_count": len(gold_claims),
        "answer_claim_count": len(answer_claims),
        "context_claim_recall": context_claim_recall,
        "answer_claim_recall": answer_claim_recall,
        "answer_claim_precision": answer_claim_precision,
        "answer_claim_f1": answer_claim_f1,
        "grounded_claim_ratio": grounded_claim_ratio,
        "hallucinated_claim_ratio": hallucinated_claim_ratio,
        "context_utilization": context_utilization,
        "evidence_attribution_precision": attribution_metrics["evidence_attribution_precision"],
        "evidence_attribution_recall": attribution_metrics["evidence_attribution_recall"],
        "evidence_attribution_f1": attribution_metrics["evidence_attribution_f1"],
        "evidence_coverage": attribution_metrics["evidence_coverage"],
        "attributed_answer_claim_count": attribution_metrics["attributed_answer_claim_count"],
        "attributed_gold_claim_count": attribution_metrics["attributed_gold_claim_count"],
        "invalid_attribution_count": attribution_metrics["invalid_attribution_count"],
        "claim_evidence_map": claim_evidence_map,
        "unsupported_claim_count": len(answer_claims) - len(answer_supported_by_context),
        "missing_gold_claim_count": len(gold_claims) - len(gold_present_in_answer),
        "contradicted_claim_count": len(answer_contradicted_by_context),
        "claim_diagnostic": claim_diagnostic,
    }


# Combine coverage and grounding signals into one practical label for evaluation.
_NUMBER_WORD_MARKERS = {
    "null",
    "ein",
    "eins",
    "eine",
    "einer",
    "einem",
    "eines",
    "zwei",
    "zweimal",
    "drei",
    "dreimal",
    "vier",
    "viermal",
    "fünf",
    "fuenf",
    "fünfmal",
    "fuenfmal",
    "sechs",
    "sechsmal",
    "sieben",
    "siebenmal",
    "acht",
    "achtmal",
    "neun",
    "neunmal",
    "zehn",
    "elf",
    "zwölf",
    "zwoelf",
    "dreizehn",
    "vierzehn",
    "fünfzehn",
    "fuenfzehn",
    "sechzehn",
    "siebzehn",
    "achtzehn",
    "neunzehn",
    "zwanzig",
    "dreißig",
    "dreissig",
    "vierzig",
    "fünfzig",
    "fuenfzig",
    "sechzig",
    "siebzig",
    "achtzig",
    "neunzig",
    "hundert",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
}

_NUMBER_WORD_TO_DIGIT = {
    "null": "0",
    "zwei": "2",
    "zweimal": "2",
    "drei": "3",
    "dreimal": "3",
    "vier": "4",
    "viermal": "4",
    "fünf": "5",
    "fuenf": "5",
    "fünfmal": "5",
    "fuenfmal": "5",
    "sechs": "6",
    "sechsmal": "6",
    "sieben": "7",
    "siebenmal": "7",
    "acht": "8",
    "achtmal": "8",
    "neun": "9",
    "neunmal": "9",
    "zehn": "10",
    "elf": "11",
    "zwölf": "12",
    "zwoelf": "12",
    "dreizehn": "13",
    "vierzehn": "14",
    "fünfzehn": "15",
    "fuenfzehn": "15",
    "sechzehn": "16",
    "siebzehn": "17",
    "achtzehn": "18",
    "neunzehn": "19",
    "zwanzig": "20",
    "dreißig": "30",
    "dreissig": "30",
    "vierzig": "40",
    "fünfzig": "50",
    "fuenfzig": "50",
    "sechzig": "60",
    "siebzig": "70",
    "achtzig": "80",
    "neunzig": "90",
    "hundert": "100",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
}

_GENERIC_ANCHOR_STOPWORDS = {
    "the", "and", "or", "of", "for", "with", "from", "after", "before", "in", "on", "at",
    "der", "die", "das", "den", "dem", "des", "und", "oder", "von", "vom", "mit", "nach",
    "vor", "für", "im", "in", "am", "an", "zu", "zur", "zum", "bei", "auf",
}


def _canonical_number_token(token: str) -> str | None:
    normalized = normalize_text(token).replace(",", ".")
    if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
        return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized
    return _NUMBER_WORD_TO_DIGIT.get(normalized)


def _extract_number_spans(text: str) -> List[tuple[str, int, int]]:
    normalized = normalize_text(text)
    spans: List[tuple[str, int, int]] = []
    weak_one_forms = {"ein", "eins", "eine", "einer", "einem", "eines", "one"}
    pattern = r"\b\d+(?:[.,]\d+)?\b|\b[a-zäöüß]+\b"
    for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
        raw = match.group(0).casefold()
        if raw in weak_one_forms:
            continue
        number = _canonical_number_token(raw)
        if number is not None:
            spans.append((number, match.start(), match.end()))
    return spans


def _contentish_token(token: str) -> str:
    normalized = normalize_for_matching(token)
    parts = [part for part in normalized.split() if part and part not in _GENERIC_ANCHOR_STOPWORDS]
    return " ".join(parts)


def _nearby_unit_tokens(text: str, end: int, *, window: int = 48) -> List[str]:
    tail = text[end : end + window]
    units: List[str] = []
    for token in re.findall(r"\b[\wäöüÄÖÜß.-]{2,}\b", tail, flags=re.UNICODE)[:3]:
        normalized = lemmatize_text(token)
        if normalized and normalized not in _GENERIC_ANCHOR_STOPWORDS and not _canonical_number_token(normalized):
            units.append(normalized)
    return units


def _extract_code_markers(text: str) -> set[str]:
    normalized = normalize_text(text)
    markers: set[str] = set()
    for match in re.finditer(r"\b[a-z]{1,4}\d{1,4}[a-z]?\b|\b\d+[a-z]{1,4}\b", normalized, flags=re.IGNORECASE):
        token = match.group(0).casefold()
        if not re.fullmatch(r"\d+", token):
            markers.add(f"code:{token}")
    return markers


def _extract_named_phrase_markers(text: str) -> set[str]:
    markers: set[str] = set()
    proper = r"(?:[A-ZÄÖÜ][\w.-]{2,}|[A-Z]{2,})"
    phrase_pattern = rf"\b{proper}(?:\s+(?:of|and|und|&)\s+{proper}|\s+{proper}){{1,5}}\b"
    for match in re.finditer(phrase_pattern, text, flags=re.UNICODE):
        phrase_tokens = [
            _contentish_token(token)
            for token in re.findall(r"\b[\wäöüÄÖÜß.-]{2,}\b", match.group(0), flags=re.UNICODE)
        ]
        tokens = [token for token in phrase_tokens if token and token not in _GENERIC_ANCHOR_STOPWORDS]
        for start in range(len(tokens)):
            for end in range(start + 2, min(len(tokens), start + 4) + 1):
                markers.add(f"phrase:{' '.join(tokens[start:end])}")

    for token in re.findall(r"\b[A-ZÄÖÜ][\w.-]{3,}\b", text, flags=re.UNICODE):
        normalized = _contentish_token(token)
        if re.search(r"(semester|semestre|trimester|quarter|monat|month|jahr|year)$", normalized):
            markers.add(f"period:{normalized}")
    return markers


def _extract_decisive_answer_markers(text: str) -> set[str]:
    if not text.strip():
        return set()

    normalized = normalize_text(text)
    markers: set[str] = set()
    for number, _start, end in _extract_number_spans(normalized):
        markers.add(f"number:{number}")
        for unit in _nearby_unit_tokens(normalized, end):
            markers.add(f"quantity:{number}:{unit}")
            break

    markers.update(_extract_code_markers(text))
    markers.update(_extract_named_phrase_markers(text))
    return markers


def _marker_coverage(gold_markers: set[str], answer_markers: set[str]) -> float | None:
    if not gold_markers:
        return None
    return len(gold_markers.intersection(answer_markers)) / len(gold_markers)


def _critical_markers(markers: set[str]) -> set[str]:
    return {
        marker
        for marker in markers
        if marker.startswith(("number:", "code:"))
    }


def _markers_decisively_covered(gold_answer: str, answer: str) -> bool:
    gold_markers = _extract_decisive_answer_markers(gold_answer)
    if not gold_markers:
        return False
    answer_markers = _extract_decisive_answer_markers(answer)
    if gold_markers.issubset(answer_markers):
        return True

    gold_critical_markers = _critical_markers(gold_markers)
    if gold_critical_markers and not gold_critical_markers.issubset(answer_markers):
        return False

    marker_coverage = _marker_coverage(gold_markers, answer_markers)
    return marker_coverage is not None and marker_coverage >= 0.60


def _extract_polarity(text: str) -> Optional[str]:
    normalized = normalize_text(text)
    if re.match(r"^(ja|yes)\b", normalized):
        return "yes"
    if re.match(r"^(nein|no)\b", normalized):
        return "no"
    return None


def classify_answer(
    *,
    gold_answer: str,
    answer: str,
    gold_answer_overlap: Optional[float],
    answer_gold_support: Optional[float],
    answer_has_gold_substring: Optional[bool],
    faithfulness: Optional[float],
    claim_diagnostic: str | None = None,
) -> str:
    supported = faithfulness is None or faithfulness >= 0.45
    clearly_unsupported = faithfulness is not None and faithfulness < 0.35
    refusal = answer_is_refusal(answer)
    gold_overlap = gold_answer_overlap or 0.0
    gold_support = answer_gold_support or 0.0

    if answer_has_gold_substring and supported:
        return "correct"

    if refusal:
        if clearly_unsupported:
            return "unsupported"
        return "incorrect"

    gold_polarity = _extract_polarity(gold_answer)
    answer_polarity = _extract_polarity(answer)
    if gold_polarity and answer_polarity and gold_polarity != answer_polarity:
        return "incorrect"

    decisive_markers_covered = _markers_decisively_covered(gold_answer, answer)

    # Correct answers often include citations, program names, or harmless
    # explanatory context that claim splitting treats as extra claims. If the
    # answer is grounded and covers the decisive value(s) or almost all gold
    # information, prefer the answer-level correctness signal.
    if supported and (
        gold_support >= 0.75
        or (decisive_markers_covered and gold_overlap >= 0.25)
        or (
            decisive_markers_covered
            and claim_diagnostic in {"answer_contains_extra_claims", "claim_contradiction", "unsupported_generated_claims"}
        )
    ):
        return "correct"

    if claim_diagnostic in {"unsupported_generated_claims", "claim_contradiction"}:
        return "unsupported" if clearly_unsupported else "incorrect"

    if claim_diagnostic == "answer_incomplete_from_available_context":
        if supported and gold_overlap >= 0.20:
            return "correct"
        if supported and gold_overlap >= 0.12:
            return "partially_correct"
        if gold_overlap >= 0.12 or gold_support >= 0.04:
            return "partially_correct"
        return "incorrect"

    if claim_diagnostic == "retrieval_claim_miss":
        if gold_support < 0.05:
            return "unsupported" if clearly_unsupported else "incorrect"
        if supported and gold_overlap >= 0.25:
            return "correct"
        if supported and gold_overlap >= 0.20:
            return "partially_correct"
        if gold_overlap >= 0.20:
            return "partially_correct"
        return "incorrect"

    if supported and gold_overlap >= 0.25 and gold_support >= 0.05:
        return "correct"
    if supported and gold_overlap >= 0.20 and gold_support >= 0.03:
        return "partially_correct"
    if supported and gold_overlap >= 0.15:
        return "partially_correct"
    if supported and gold_support >= 0.06:
        return "partially_correct"

    if clearly_unsupported:
        return "unsupported"
    if gold_overlap >= 0.20 or gold_support >= 0.05:
        return "partially_correct"
    return "incorrect"


# Compute all answer-level heuristic signals in one place so reporting stays
# consistent across experiments.
def evaluate_answer_metrics(
    item: Dict,
    answer: str,
    retrieved: Sequence[Dict],
    claim_judge_result: ClaimJudgeResult | None = None,
    runtime_retrieval_result: Dict[str, object] | None = None,
) -> AnswerMetricResult:
    context_text = "\n".join(
        "\n".join(
            part
            for part in [
                str(row.get("section_id", "")),
                str(row.get("title", "")),
                str(row.get("text", "")),
            ]
            if part.strip()
        )
        for row in retrieved
    )
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
    is_answerable = expected_answerable(item)
    abstained = answer_is_refusal(answer)
    abstention_correct = None
    over_answered = False
    false_refusal = False
    if is_answerable:
        abstention_correct = not abstained
        false_refusal = abstained
    else:
        abstention_correct = abstained
        over_answered = not abstained

    fallback_claim_metrics = evaluate_claim_metrics(item, answer, context_text, retrieved)
    if (
        claim_judge_result is not None
        and claim_judge_result.status == "success"
        and claim_judge_result.metrics
    ):
        claim_metrics = {**fallback_claim_metrics, **claim_judge_result.metrics}
    else:
        claim_metrics = fallback_claim_metrics
    final_label = classify_answer(
        gold_answer=gold_answer,
        answer=answer,
        gold_answer_overlap=gold_answer_overlap,
        answer_gold_support=answer_gold_support,
        answer_has_gold_substring=answer_has_gold_substring,
        faithfulness=faithfulness,
        claim_diagnostic=claim_metrics["claim_diagnostic"],
    )
    entity_metrics = compute_entity_metrics(
        item=item,
        answer=answer,
        context_text=context_text,
    )
    noise_metrics = compute_noise_sensitivity_metrics(
        claim_evidence_map=claim_metrics["claim_evidence_map"],
        answer_claim_count=int(claim_metrics["answer_claim_count"]),
        unsupported_claim_count=int(claim_metrics["unsupported_claim_count"]),
        contradicted_claim_count=int(claim_metrics["contradicted_claim_count"]),
    )
    claim_level_correct = (
        claim_metrics["answer_claim_f1"] is not None
        and claim_metrics["answer_claim_f1"] >= 0.8
        and (claim_metrics["grounded_claim_ratio"] is None or claim_metrics["grounded_claim_ratio"] >= 0.75)
    )
    gold_polarity = _extract_polarity(gold_answer)
    answer_polarity = _extract_polarity(answer)
    polarity_matches = bool(gold_polarity and answer_polarity and gold_polarity == answer_polarity)
    decisive_markers_covered = _markers_decisively_covered(gold_answer, answer)
    answer_level_correct = final_label == "correct" and (
        bool(answer_has_gold_substring)
        or (answer_gold_support is not None and answer_gold_support >= 0.75)
        or (
            (gold_answer_overlap or 0.0) >= 0.50
            and (answer_gold_support or 0.0) >= 0.35
        )
        or decisive_markers_covered
        or (polarity_matches and (gold_answer_overlap or 0.0) >= 0.35)
    )
    if claim_metrics["claim_diagnostic"] in {
        "claim_contradiction",
        "unsupported_generated_claims",
    } and final_label == "partially_correct":
        final_label = "unsupported"
    elif (
        claim_metrics["claim_diagnostic"]
        in {"answer_incomplete", "answer_incomplete_from_available_context"}
        and final_label == "correct"
        and not claim_level_correct
        and not answer_level_correct
    ):
        final_label = "partially_correct"
    elif claim_level_correct and final_label in {"incorrect", "partially_correct"}:
        final_label = "correct"
    if not is_answerable:
        final_label = "correct" if abstained else "unsupported"
    elif false_refusal:
        final_label = "incorrect"

    return AnswerMetricResult(
        answer_accuracy_label=final_label,
        gold_answer_overlap=gold_answer_overlap, # the proportion of gold answer to the actual answer
        answer_gold_support=answer_gold_support, # the proportion of the actual answer that is covered by the gold answer
        proxy_faithfulness=faithfulness, # the proportion of answer to the retrieved context
        proxy_context_relevance=context_relevance, # the proportion of question to the retrieved context
        answer_has_gold_substring=answer_has_gold_substring, # whether the actual answer fully contains the gold answer
        expected_answerable=is_answerable,
        abstained=abstained,
        abstention_correct=abstention_correct,
        over_answered=over_answered,
        false_refusal=false_refusal,
        answerability_confidence=runtime_retrieval_result.get("score") if runtime_retrieval_result else None,
        runtime_retrieval_status=str(runtime_retrieval_result.get("status", "not_evaluated")) if runtime_retrieval_result else "not_evaluated",
        runtime_retrieval_action=str(runtime_retrieval_result.get("action", "answer")) if runtime_retrieval_result else "answer",
        runtime_retrieval_reason=str(runtime_retrieval_result.get("reason", "")) if runtime_retrieval_result else "",
        runtime_retrieval_score=runtime_retrieval_result.get("score") if runtime_retrieval_result else None,
        gold_claim_count=int(claim_metrics["gold_claim_count"]),
        answer_claim_count=int(claim_metrics["answer_claim_count"]),
        context_claim_recall=claim_metrics["context_claim_recall"],
        answer_claim_recall=claim_metrics["answer_claim_recall"],
        answer_claim_precision=claim_metrics["answer_claim_precision"],
        answer_claim_f1=claim_metrics["answer_claim_f1"],
        factual_correctness_precision=claim_metrics["answer_claim_precision"],
        factual_correctness_recall=claim_metrics["answer_claim_recall"],
        factual_correctness_f1=claim_metrics["answer_claim_f1"],
        grounded_claim_ratio=claim_metrics["grounded_claim_ratio"],
        hallucinated_claim_ratio=claim_metrics["hallucinated_claim_ratio"],
        noise_sensitivity_relevant=noise_metrics["noise_sensitivity_relevant"],
        noise_sensitivity_irrelevant=noise_metrics["noise_sensitivity_irrelevant"],
        context_utilization=claim_metrics["context_utilization"],
        context_entities_recall=entity_metrics["context_entities_recall"],
        answer_entity_precision=entity_metrics["answer_entity_precision"],
        evidence_attribution_precision=claim_metrics["evidence_attribution_precision"],
        evidence_attribution_recall=claim_metrics["evidence_attribution_recall"],
        evidence_attribution_f1=claim_metrics["evidence_attribution_f1"],
        evidence_coverage=claim_metrics["evidence_coverage"],
        attributed_answer_claim_count=int(claim_metrics["attributed_answer_claim_count"]),
        attributed_gold_claim_count=int(claim_metrics["attributed_gold_claim_count"]),
        invalid_attribution_count=int(claim_metrics["invalid_attribution_count"]),
        claim_evidence_map=list(claim_metrics["claim_evidence_map"]),
        unsupported_claim_count=int(claim_metrics["unsupported_claim_count"]),
        missing_gold_claim_count=int(claim_metrics["missing_gold_claim_count"]),
        contradicted_claim_count=int(claim_metrics["contradicted_claim_count"]),
        claim_diagnostic=str(claim_metrics["claim_diagnostic"]),
        claim_judge_used=bool(claim_judge_result.used) if claim_judge_result else False,
        claim_judge_status=claim_judge_result.status if claim_judge_result else "disabled",
        claim_judge_error=claim_judge_result.error if claim_judge_result else None,
        claim_judge_model=claim_judge_result.model if claim_judge_result else None,
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

    if not answer_metrics.expected_answerable:
        if answer_metrics.abstained:
            return DiagnosticResult(
                primary_error_reason="ok",
                secondary_error_reason="correct_abstention",
                explanation="The question is marked unanswerable and the system correctly refused to answer.",
            )
        return DiagnosticResult(
            primary_error_reason="over_answering",
            secondary_error_reason="unanswerable_question",
            explanation="The question is marked unanswerable, but the system still produced an answer.",
        )
    if answer_metrics.false_refusal:
        return DiagnosticResult(
            primary_error_reason="false_refusal",
            secondary_error_reason="answerable_question",
            explanation="The question is marked answerable, but the system refused to answer.",
        )

    if answer_metrics.claim_diagnostic == "retrieval_claim_miss":
        return DiagnosticResult(
            primary_error_reason="retrieval_claim_miss",
            secondary_error_reason="missing_gold_evidence",
            explanation="Most gold/reference claims are absent from the retrieved context, so the generator did not receive enough evidence.",
        )
    if answer_metrics.claim_diagnostic == "partial_evidence_retrieved":
        return DiagnosticResult(
            primary_error_reason="partial_retrieval",
            secondary_error_reason="missing_gold_claims",
            explanation="Only part of the needed reference claims is present in the retrieved context.",
        )
    if answer_metrics.claim_diagnostic == "claim_contradiction":
        return DiagnosticResult(
            primary_error_reason="claim_contradiction",
            secondary_error_reason="generation_or_context_conflict",
            explanation="The answer contains at least one claim that appears to conflict with numeric evidence in the retrieved context.",
        )
    if answer_metrics.claim_diagnostic == "unsupported_generated_claims":
        if weak_retrieval:
            return DiagnosticResult(
                primary_error_reason="retrieval_and_grounding_failure",
                secondary_error_reason="unsupported_generated_claims",
                explanation="The retrieved context is weak and a large share of generated claims is not supported by it.",
            )
        return DiagnosticResult(
            primary_error_reason="generation_hallucination",
            secondary_error_reason="unsupported_generated_claims",
            explanation="The retrieved context contains usable evidence, but the answer adds unsupported claims.",
        )
    if answer_metrics.claim_diagnostic == "answer_incomplete_from_available_context":
        return DiagnosticResult(
            primary_error_reason="answer_incomplete_from_good_context",
            secondary_error_reason="low_context_utilization",
            explanation="The retrieved context contains relevant claims, but the answer fails to use enough of them.",
        )
    if answer_metrics.claim_diagnostic == "answer_incomplete":
        return DiagnosticResult(
            primary_error_reason="answer_incomplete",
            secondary_error_reason="missing_gold_claims",
            explanation="The answer omits important reference claims.",
        )
    if answer_metrics.claim_diagnostic == "answer_contains_extra_claims":
        return DiagnosticResult(
            primary_error_reason="answer_contains_extra_claims",
            secondary_error_reason="low_claim_precision",
            explanation="The answer covers some reference claims but also includes extra claims not present in the gold answer.",
        )

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
    summary = {
        "mean_gold_answer_overlap": summarize_average(metric_rows, "gold_answer_overlap"),
        "mean_answer_gold_support": summarize_average(metric_rows, "answer_gold_support"),
        "mean_proxy_faithfulness": summarize_average(metric_rows, "proxy_faithfulness"),
        "mean_proxy_context_relevance": summarize_average(metric_rows, "proxy_context_relevance"),
        "mean_context_claim_recall": summarize_average(metric_rows, "context_claim_recall"),
        "mean_answer_claim_recall": summarize_average(metric_rows, "answer_claim_recall"),
        "mean_answer_claim_precision": summarize_average(metric_rows, "answer_claim_precision"),
        "mean_answer_claim_f1": summarize_average(metric_rows, "answer_claim_f1"),
        "mean_factual_correctness_precision": summarize_average(
            metric_rows, "factual_correctness_precision"
        ),
        "mean_factual_correctness_recall": summarize_average(
            metric_rows, "factual_correctness_recall"
        ),
        "mean_factual_correctness_f1": summarize_average(metric_rows, "factual_correctness_f1"),
        "mean_grounded_claim_ratio": summarize_average(metric_rows, "grounded_claim_ratio"),
        "mean_hallucinated_claim_ratio": summarize_average(metric_rows, "hallucinated_claim_ratio"),
        "mean_noise_sensitivity_relevant": summarize_average(
            metric_rows, "noise_sensitivity_relevant"
        ),
        "mean_noise_sensitivity_irrelevant": summarize_average(
            metric_rows, "noise_sensitivity_irrelevant"
        ),
        "mean_context_utilization": summarize_average(metric_rows, "context_utilization"),
        "mean_context_entities_recall": summarize_average(metric_rows, "context_entities_recall"),
        "mean_answer_entity_precision": summarize_average(metric_rows, "answer_entity_precision"),
        "mean_evidence_attribution_precision": summarize_average(
            metric_rows, "evidence_attribution_precision"
        ),
        "mean_evidence_attribution_recall": summarize_average(
            metric_rows, "evidence_attribution_recall"
        ),
        "mean_evidence_attribution_f1": summarize_average(metric_rows, "evidence_attribution_f1"),
        "mean_evidence_coverage": summarize_average(metric_rows, "evidence_coverage"),
        "mean_answerability_confidence": summarize_average(metric_rows, "answerability_confidence"),
        "n_answerable": sum(1 for row in metric_rows if row.get("expected_answerable") is True),
        "n_unanswerable": sum(1 for row in metric_rows if row.get("expected_answerable") is False),
        "n_abstained": sum(1 for row in metric_rows if row.get("abstained") is True),
        "n_correct_abstentions": sum(
            1 for row in metric_rows if row.get("expected_answerable") is False and row.get("abstained") is True
        ),
        "n_over_answered": sum(1 for row in metric_rows if row.get("over_answered") is True),
        "n_false_refusals": sum(1 for row in metric_rows if row.get("false_refusal") is True),
        "abstention_precision": safe_ratio(
            sum(
                1
                for row in metric_rows
                if row.get("expected_answerable") is False and row.get("abstained") is True
            ),
            sum(1 for row in metric_rows if row.get("abstained") is True),
        ),
        "abstention_recall": safe_ratio(
            sum(
                1
                for row in metric_rows
                if row.get("expected_answerable") is False and row.get("abstained") is True
            ),
            sum(1 for row in metric_rows if row.get("expected_answerable") is False),
        ),
        "over_answering_rate": safe_ratio(
            sum(1 for row in metric_rows if row.get("over_answered") is True),
            sum(1 for row in metric_rows if row.get("expected_answerable") is False),
        ),
        "false_refusal_rate": safe_ratio(
            sum(1 for row in metric_rows if row.get("false_refusal") is True),
            sum(1 for row in metric_rows if row.get("expected_answerable") is True),
        ),
        "n_attributed_answer_claims": sum(
            int(row.get("attributed_answer_claim_count") or 0) for row in metric_rows
        ),
        "n_attributed_gold_claims": sum(
            int(row.get("attributed_gold_claim_count") or 0) for row in metric_rows
        ),
        "n_invalid_attributions": sum(
            int(row.get("invalid_attribution_count") or 0) for row in metric_rows
        ),
        "n_unsupported_claims": sum(int(row.get("unsupported_claim_count") or 0) for row in metric_rows),
        "n_missing_gold_claims": sum(int(row.get("missing_gold_claim_count") or 0) for row in metric_rows),
        "n_contradicted_claims": sum(int(row.get("contradicted_claim_count") or 0) for row in metric_rows),
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
    summary["confidence_calibration"] = summarize_confidence_calibration(
        metric_rows,
        confidence_key="answerability_confidence",
        correct_fn=lambda row: row.get("answer_accuracy_label") == "correct",
    )
    return summary


def summarize_confidence_calibration(
    metric_rows: Sequence[Dict],
    *,
    confidence_key: str,
    correct_fn,
    n_bins: int = 5,
) -> Dict[str, object]:
    pairs: List[tuple[float, float]] = []
    for row in metric_rows:
        confidence = row.get(confidence_key)
        if confidence is None or confidence == "":
            continue
        try:
            conf = min(max(float(confidence), 0.0), 1.0)
        except (TypeError, ValueError):
            continue
        pairs.append((conf, 1.0 if correct_fn(row) else 0.0))

    if not pairs:
        return {
            "n_scored": 0,
            "mean_confidence": None,
            "accuracy": None,
            "brier_score": None,
            "expected_calibration_error": None,
            "bins": [],
        }

    mean_confidence = sum(conf for conf, _ in pairs) / len(pairs)
    accuracy = sum(correct for _, correct in pairs) / len(pairs)
    brier_score = sum((conf - correct) ** 2 for conf, correct in pairs) / len(pairs)

    bins: List[Dict[str, object]] = []
    ece = 0.0
    for bin_index in range(n_bins):
        low = bin_index / n_bins
        high = (bin_index + 1) / n_bins
        bucket = [
            (conf, correct)
            for conf, correct in pairs
            if (low <= conf < high) or (bin_index == n_bins - 1 and conf == 1.0)
        ]
        if not bucket:
            continue
        bucket_conf = sum(conf for conf, _ in bucket) / len(bucket)
        bucket_acc = sum(correct for _, correct in bucket) / len(bucket)
        bucket_weight = len(bucket) / len(pairs)
        ece += abs(bucket_acc - bucket_conf) * bucket_weight
        bins.append(
            {
                "range": [round(low, 3), round(high, 3)],
                "n": len(bucket),
                "mean_confidence": bucket_conf,
                "accuracy": bucket_acc,
            }
        )

    return {
        "n_scored": len(pairs),
        "mean_confidence": mean_confidence,
        "accuracy": accuracy,
        "brier_score": brier_score,
        "expected_calibration_error": ece,
        "bins": bins,
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


def is_relevant_grade(grade: int | float | None, *, min_grade: int = MIN_RELEVANT_GRADE) -> bool:
    if grade is None:
        return False
    try:
        return float(grade) >= float(min_grade)
    except (TypeError, ValueError):
        return False


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

    retrieved_grades = [retrieval_relevance_grade(item, row) for row in retrieved[:k]]
    all_candidate_grades = [retrieval_relevance_grade(item, row) for row in candidate_chunks]
    target_doc_id = item.get("doc_id")
    target_doc_ranks = [
        rank
        for rank, row in enumerate(retrieved[:k], start=1)
        if target_doc_id and metadata_value_matches(row.get("doc_id", ""), target_doc_id)
    ]

    first_relevant_rank = next(
        (rank for rank, grade in enumerate(retrieved_grades, start=1) if is_relevant_grade(grade)),
        None,
    )
    mrr_at_k = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0

    ideal_grades = sorted(all_candidate_grades, reverse=True)
    actual_dcg = dcg_at_k(retrieved_grades, k)
    ideal_dcg = dcg_at_k(ideal_grades, k)
    ndcg_at_k = min(1.0, actual_dcg / ideal_dcg) if ideal_dcg > 0 else None

    relevant_chunk_count = sum(1 for grade in all_candidate_grades if is_relevant_grade(grade))
    retrieved_relevant_count = sum(1 for grade in retrieved_grades if is_relevant_grade(grade))
    recall_at_k = (
        min(1.0, retrieved_relevant_count / relevant_chunk_count)
        if relevant_chunk_count > 0
        else None
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


def retrieval_relevance_grade(item: Dict, row: Dict) -> int:
    target_doc_id = item.get("doc_id")
    if not target_doc_id:
        return weak_chunk_relevance_grade(item, str(row.get("text", "")))
    row_matches_target = metadata_value_matches(row.get("doc_id", ""), target_doc_id) or metadata_value_matches(
        row.get("cpv_code", ""), target_doc_id
    )
    if not row_matches_target:
        return 0
    if row.get("program_id") == "cpv" or row.get("cpv_code"):
        return 3
    return weak_chunk_relevance_grade(item, str(row.get("text", "")))


# Count failure categories to show which error mode dominates an experiment.
def summarize_diagnostics(metric_rows: Sequence[Dict]) -> Dict:
    counts: Dict[str, int] = {}
    claim_counts: Dict[str, int] = {}
    judge_status_counts: Dict[str, int] = {}
    for row in metric_rows:
        counts[row["primary_error_reason"]] = counts.get(row["primary_error_reason"], 0) + 1
        claim_diagnostic = row.get("claim_diagnostic")
        if claim_diagnostic:
            claim_counts[claim_diagnostic] = claim_counts.get(claim_diagnostic, 0) + 1
        judge_status = row.get("claim_judge_status")
        if judge_status:
            judge_status_counts[judge_status] = judge_status_counts.get(judge_status, 0) + 1
    return {
        "counts_by_primary_reason": counts,
        "counts_by_claim_diagnostic": claim_counts,
        "counts_by_claim_judge_status": judge_status_counts,
        "most_common_reason": max(counts, key=counts.get) if counts else None,
        "most_common_claim_diagnostic": (
            max(claim_counts, key=claim_counts.get) if claim_counts else None
        ),
    }


# Replace missing metrics with zero so ranking code stays simple and stable.
def safe_metric(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(value)


# Convert summary metrics into one ranking score for experiment comparison.
def score_experiment(summary: Dict, weights: Dict[str, float]) -> Dict:
    noise_penalty = 1.0 - safe_metric(summary["answer_metrics"].get("mean_noise_sensitivity_relevant"))
    answer_score = (
        0.15 * safe_metric(summary["answer_metrics"].get("mean_proxy_faithfulness"))
        + 0.20 * safe_metric(summary["answer_metrics"].get("mean_grounded_claim_ratio"))
        + 0.25 * safe_metric(summary["answer_metrics"].get("mean_answer_claim_f1"))
        + 0.20 * safe_metric(summary["answer_metrics"].get("mean_factual_correctness_f1"))
        + 0.10 * safe_metric(summary["answer_metrics"].get("mean_evidence_attribution_f1"))
        + 0.10 * noise_penalty
    )
    correctness_score = 0.0
    if summary["n_questions"] > 0:
        correctness_score = (
            summary["n_correct"] + 0.5 * summary["answer_metrics"].get("n_partially_correct", 0)
        ) / summary["n_questions"]
    retrieval_score = (
        0.3 * safe_metric(summary.get("retrieval_metrics", {}).get("mean_ndcg_at_k"))
        + 0.25 * safe_metric(summary.get("retrieval_metrics", {}).get("mean_mrr_at_k"))
        + 0.15 * safe_metric(summary.get("retrieval_metrics", {}).get("mean_ragas_recall_at_k"))
        + 0.15 * safe_metric(summary["answer_metrics"].get("mean_context_claim_recall"))
        + 0.15 * safe_metric(summary["answer_metrics"].get("mean_context_entities_recall"))
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
                "mean_context_claim_recall": summary["answer_metrics"].get(
                    "mean_context_claim_recall"
                ),
                "mean_answer_claim_f1": summary["answer_metrics"].get("mean_answer_claim_f1"),
                "mean_grounded_claim_ratio": summary["answer_metrics"].get(
                    "mean_grounded_claim_ratio"
                ),
                "mean_hallucinated_claim_ratio": summary["answer_metrics"].get(
                    "mean_hallucinated_claim_ratio"
                ),
                "mean_evidence_attribution_f1": summary["answer_metrics"].get(
                    "mean_evidence_attribution_f1"
                ),
                "mean_evidence_coverage": summary["answer_metrics"].get("mean_evidence_coverage"),
                "mean_mrr_at_k": summary.get("retrieval_metrics", {}).get("mean_mrr_at_k"),
                "mean_ndcg_at_k": summary.get("retrieval_metrics", {}).get("mean_ndcg_at_k"),
                "mean_recall_at_k": summary.get("retrieval_metrics", {}).get("mean_recall_at_k"),
                "mean_ragas_recall_at_k": summary.get("retrieval_metrics", {}).get(
                    "mean_ragas_recall_at_k"
                ),
                "most_common_error_reason": summary["diagnostics"].get("most_common_reason"),
                "most_common_claim_diagnostic": summary["diagnostics"].get(
                    "most_common_claim_diagnostic"
                ),
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
