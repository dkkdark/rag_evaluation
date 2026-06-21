from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Sequence


def as_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def recommendation(
    *,
    component: str,
    priority: str,
    issue: str,
    evidence: str,
    recommendation_text: str,
    next_experiment: str,
    implementation_hint: str = "",
    success_signal: str = "",
    source: str = "Advisor",
) -> Dict[str, str]:
    return {
        "component": component,
        "priority": priority,
        "issue": issue,
        "evidence": evidence,
        "recommendation": recommendation_text,
        "next_experiment": next_experiment,
        "implementation_hint": implementation_hint,
        "success_signal": success_signal,
        "recommendation_source": source,
    }


def priority_rank(priority: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(priority, 9)


def pick_primary(recommendations: Sequence[Dict[str, str]]) -> Dict[str, str] | None:
    if not recommendations:
        return None
    return sorted(
        recommendations,
        key=lambda row: (priority_rank(row["priority"]), row["component"], row["issue"]),
    )[0]


def _display_failure_mode(value: object) -> str:
    mapping = {
        "gold_missing_from_top_k": "expected_answer_missing_from_top_k",
        "gold_present_but_not_ranked_first": "expected_answer_present_but_not_ranked_first",
        "same_class_wrong_code": "same_close_branch_wrong_answer",
        "same_branch_wrong_code": "same_branch_wrong_answer",
    }
    return mapping.get(str(value or ""), str(value or ""))


def _display_bottleneck(value: object) -> str:
    mapping = {
        "sibling_disambiguation": "close_candidate_disambiguation",
        "hierarchy_disambiguation": "hierarchy_disambiguation",
        "candidate_generation_or_retriever": "candidate_generation_or_retriever",
        "reranker_or_prompt_selection": "reranker_or_prompt_selection",
        "confidence_calibration": "confidence_calibration",
    }
    return mapping.get(str(value or ""), str(value or ""))


def _non_none_recommendations(recommendations: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return [rec for rec in recommendations if rec.get("component") != "none"]


def _resolve_prediction_calibration(summary: Dict[str, object]) -> Dict[str, object]:
    calibration = summary.get("calibration", {}) if isinstance(summary.get("calibration"), dict) else {}
    prediction_calibration = (
        calibration.get("prediction_confidence", {})
        if isinstance(calibration.get("prediction_confidence"), dict)
        else {}
    )
    if prediction_calibration:
        return prediction_calibration
    classifier = summary.get("classifier", {}) if isinstance(summary.get("classifier"), dict) else {}
    return classifier.get("calibration", {}) if isinstance(classifier.get("calibration"), dict) else {}


def _append_recommendation_block(
    lines: List[str],
    rec: Dict[str, object],
    *,
    include_count: bool = False,
) -> None:
    heading = f"- [{rec.get('priority')}] {rec.get('component')}: {rec.get('issue')}"
    if include_count:
        heading += f" ({rec.get('count')} questions)"
    lines.append(heading)
    if rec.get("evidence"):
        lines.append(f"  - Evidence: {rec.get('evidence')}")
    lines.append(f"  - Recommendation: {rec.get('recommendation')}")
    if rec.get("implementation_hint"):
        lines.append(f"  - Implementation hint: {rec.get('implementation_hint')}")
    if rec.get("success_signal"):
        lines.append(f"  - Success signal: {rec.get('success_signal')}")
    lines.append(f"  - Next experiment: `{rec.get('next_experiment')}`")


def _append_counts_section(lines: List[str], title: str, counts: Dict[str, object]) -> None:
    lines.append(f"### {title}")
    if counts:
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none")
    lines.append("")


def build_question_recommendations(row: Dict[str, object]) -> List[Dict[str, str]]:
    recs: List[Dict[str, str]] = []
    context_claim_recall = as_float(row.get("context_claim_recall"))
    answer_claim_f1 = as_float(row.get("answer_claim_f1"))
    grounded_claim_ratio = as_float(row.get("grounded_claim_ratio"))
    context_utilization = as_float(row.get("context_utilization"))
    attribution_f1 = as_float(row.get("evidence_attribution_f1"))
    evidence_coverage = as_float(row.get("evidence_coverage"))
    context_entities_recall = as_float(row.get("context_entities_recall"))
    factual_correctness_f1 = as_float(row.get("factual_correctness_f1"))
    prediction_confidence = as_float(row.get("prediction_confidence"))
    mrr = as_float(row.get("mrr_at_k"))
    ndcg = as_float(row.get("ndcg_at_k"))
    recall = as_float(row.get("recall_at_k"))
    runtime_status = str(row.get("runtime_retrieval_status") or "")
    claim_diagnostic = str(row.get("claim_diagnostic") or "")
    primary_reason = str(row.get("primary_error_reason") or "")
    auto_flag = str(row.get("auto_flag") or "")
    answer_gold_support = as_float(row.get("answer_gold_support"))
    proxy_faithfulness = as_float(row.get("proxy_faithfulness"))
    expected_answerable = as_bool(row.get("expected_answerable"))
    over_answered = as_bool(row.get("over_answered")) is True
    false_refusal = as_bool(row.get("false_refusal")) is True
    invalid_attribution_count = int(as_float(row.get("invalid_attribution_count")) or 0)
    gold_answer = str(row.get("gold_answer") or "").strip()
    answer_is_good = (
        auto_flag == "correct"
        and (answer_claim_f1 is None or answer_claim_f1 >= 0.50)
        and (grounded_claim_ratio is None or grounded_claim_ratio >= 0.90)
        and (answer_gold_support is None or answer_gold_support >= 0.70)
    )
    likely_gold_over_specified = (
        answer_is_good
        and context_claim_recall is not None
        and context_claim_recall < 0.70
        and grounded_claim_ratio is not None
        and grounded_claim_ratio >= 0.90
    )
    failure_mode = str(row.get("failure_mode") or "")
    likely_bottleneck = str(row.get("likely_bottleneck") or "")
    display_failure_mode = _display_failure_mode(failure_mode)
    gold_rank = as_float(row.get("gold_rank"))
    score_margin = as_float(row.get("score_margin_top1_top2"))
    duplicate_rate = as_float(row.get("duplicate_cpv_rate_at_k"))
    unique_cpv = as_float(row.get("unique_cpv_at_k"))
    unique_division = as_float(row.get("unique_division_at_k"))
    best_hierarchy_score = as_float(row.get("best_hierarchy_score_at_k"))
    low_margin_decision = as_bool(row.get("low_margin_decision")) is True
    duplicate_pressure = as_bool(row.get("duplicate_candidate_pressure")) is True
    low_diversity = as_bool(row.get("low_diversity_at_k")) is True
    short_query = as_bool(row.get("short_or_ambiguous_query")) is True

    if failure_mode and failure_mode != "ok":
        if likely_bottleneck == "candidate_generation_or_retriever":
            recs.append(
                recommendation(
                    component="retriever",
                    priority="P1",
                    issue="Expected answer is missing from the candidate list",
                    evidence=f"failure_mode={display_failure_mode}, best_hierarchy_score_at_k={best_hierarchy_score}",
                    recommendation_text="Treat this as a candidate generation problem before tuning the final prompt.",
                    next_experiment="Increase top-k to 10 or 20 and compare expected-answer coverage; then test richer candidate text and alternative embeddings.",
                    implementation_hint="Add descriptions, examples, synonyms, and parent/child labels to the retriever text. If coverage stays low, compare dense vs BM25/hybrid retrieval.",
                    success_signal="Expected-answer coverage and hit@k rise, even before top-1 accuracy improves.",
                )
            )
        elif likely_bottleneck == "reranker_or_prompt_selection":
            recs.append(
                recommendation(
                    component="selection",
                    priority="P1",
                    issue="Expected answer is retrieved but not selected",
                    evidence=f"expected_answer_rank={gold_rank}, score_margin_top1_top2={score_margin}",
                    recommendation_text="Focus on the decision layer: reranker, score fusion, or a contrastive prompt over the top candidates.",
                    next_experiment="Freeze retrieval, then compare current top-1 selection against a reranker or prompt that must contrast rank-1 vs candidates containing the expected answer.",
                    implementation_hint="Ask the selector to justify why the winning answer is better than close alternatives, especially when the score margin is small.",
                    success_signal="top-1 accuracy rises while hit@k stays roughly stable.",
                )
            )
        elif likely_bottleneck in {"sibling_disambiguation", "hierarchy_disambiguation"}:
            recs.append(
                recommendation(
                    component="hierarchy",
                    priority="P2",
                    issue="Prediction is in a related hierarchy branch but still wrong",
                    evidence=f"failure_mode={display_failure_mode}, best_hierarchy_score_at_k={best_hierarchy_score}",
                recommendation_text="Improve hierarchy-aware disambiguation instead of treating these as random retrieval misses.",
                next_experiment="Add branch definitions/examples and run a hierarchy-aware reranker for related alternatives.",
                implementation_hint="For each candidate, include parent labels, exclusions, domain-specific examples, and contrastive descriptions for related alternatives.",
                success_signal="same-branch misses convert into exact matches and mean_hierarchy_score_top1 stays high.",
            )
        )
        elif likely_bottleneck == "confidence_calibration":
            recs.append(
                recommendation(
                    component="calibration",
                    priority="P1",
                    issue="Wrong answer is predicted with high confidence",
                    evidence=f"prediction_confidence={prediction_confidence}, failure_mode={display_failure_mode}",
                    recommendation_text="Do not auto-accept high-confidence predictions until scores are calibrated.",
                    next_experiment="Fit threshold/reliability bins on a labeled validation set and route uncertain or overconfident-wrong bands to review.",
                    implementation_hint="Track high_confidence_wrong_rate by score bucket and compare raw score against score margin features.",
                    success_signal="High-confidence wrong rate drops and ECE/Brier score improve.",
                )
            )

    if failure_mode and failure_mode != "ok" and low_margin_decision:
        recs.append(
            recommendation(
                component="prompt",
                priority="P2",
                issue="Top candidates are too close to choose blindly",
                evidence=f"score_margin_top1_top2={score_margin}",
                recommendation_text="Use a contrastive selection prompt or reranker on low-margin cases instead of trusting rank 1.",
                next_experiment="Only rerank examples where score_margin_top1_top2 <= 0.05 and compare accuracy/cost.",
                implementation_hint="Prompt the model with top candidates, their hierarchy or category context, and a required reason for rejecting each close alternative.",
                success_signal="Low-margin accuracy improves without adding cost to easy cases.",
            )
        )

    if failure_mode and failure_mode != "ok" and duplicate_pressure:
        recs.append(
            recommendation(
                component="retriever",
                priority="P2",
                issue="Candidate list contains duplicate answers",
                evidence=f"unique_answers_at_k={unique_cpv}, unique_branches_at_k={unique_division}, duplicate_answer_rate_at_k={duplicate_rate}",
                recommendation_text="Deduplicate repeated candidates before final selection so top-k contains more real alternatives.",
                next_experiment="Compare raw top-k against a unique-answer top-k candidate set.",
                implementation_hint="Collapse duplicate answers, keep the best score per answer, and then evaluate whether coverage or selection improves.",
                success_signal="Higher unique answer count and better hit@k without lowering top-1 accuracy.",
            )
        )

    if failure_mode and failure_mode != "ok" and low_diversity and not duplicate_pressure:
        recs.append(
            recommendation(
                component="retriever",
                priority="P2",
                issue="Candidate list stays inside one broad branch",
                evidence=f"unique_answers_at_k={unique_cpv}, unique_branches_at_k={unique_division}, duplicate_answer_rate_at_k={duplicate_rate}",
                recommendation_text="Check whether candidate generation is over-committing to one branch before the final decision step.",
                next_experiment="Compare current top-k against a diversity-aware candidate set, then rerank both.",
                implementation_hint="This is not a duplicate problem; use it as a diagnostic to see whether the expected answer's branch is excluded too early.",
                success_signal="More candidate branches are available when needed, without reducing exact top-1 accuracy.",
            )
        )

    if failure_mode and failure_mode != "ok" and short_query:
        recs.append(
            recommendation(
                component="benchmark",
                priority="P3",
                issue="Query may be too short or ambiguous",
                evidence=f"query_token_count={row.get('query_token_count')}",
                recommendation_text="Treat this row as needing extra domain context before blaming only the classifier.",
                next_experiment="Add title, description, source/category metadata, or manual ambiguity label for this query and rerun.",
                implementation_hint="Short titles often need object description, accepted alternatives, or source context to disambiguate related classes.",
                success_signal="Manual audit confirms whether the expected answer is uniquely inferable from the provided query.",
            )
        )

    if failure_mode:
        return sorted(recs, key=lambda row: (priority_rank(row["priority"]), row["component"]))

    if expected_answerable is False and over_answered:
        recs.append(
            recommendation(
                component="abstention",
                priority="P0",
                issue="Over-answering on unanswerable question",
                evidence="Question is marked answerable=false but the system produced an answer.",
                recommendation_text="Add or tighten an abstention policy based on evidence coverage, answerability confidence, and contradiction checks.",
                next_experiment="Run with --abstain-on-weak-evidence and add more unanswerable questions to the benchmark.",
                implementation_hint="Gate answer generation on runtime_retrieval_status plus a confidence threshold, then log abstentions separately from normal errors.",
                success_signal="Higher abstention precision and lower over_answering_rate without sharply increasing false_refusal_rate.",
            )
        )
    if expected_answerable is True and false_refusal:
        recs.append(
            recommendation(
                component="abstention",
                priority="P1",
                issue="False refusal",
                evidence="Question is marked answerable=true but the system refused to answer.",
                recommendation_text="Relax the abstention threshold or improve retrieval before refusal so answerable questions are not blocked too early.",
                next_experiment="Compare runs with and without --abstain-on-weak-evidence on answerable-only questions.",
                implementation_hint="Audit weak_evidence thresholds first; if they are reasonable, prioritize retrieval quality before loosening the gate globally.",
                success_signal="Lower false_refusal_rate while keeping unsupported answers flat or lower.",
            )
        )

    if context_claim_recall is not None and context_claim_recall < 0.35 and not likely_gold_over_specified:
        recs.append(
            recommendation(
                component="retrieval",
                priority="P1",
                issue="Low evidence coverage",
                evidence=f"context_claim_recall={context_claim_recall:.2f}, mrr_at_k={mrr}",
                recommendation_text="Relevant evidence is mostly absent from top-k. Inspect metadata filters, try hybrid/BM25+dense retrieval, and increase top_k.",
                next_experiment="--retriever auto --chunking auto --top-k 10",
                implementation_hint="Check whether the expected evidence is missing entirely or just under-ranked. If missing entirely, focus on candidate generation before reranking.",
                success_signal="Higher context_claim_recall and ragas_recall_at_k with a visible drop in retrieval_claim_miss.",
            )
        )
    elif context_claim_recall is not None and context_claim_recall < 0.70 and not likely_gold_over_specified:
        recs.append(
            recommendation(
                component="retrieval",
                priority="P2",
                issue="Partial evidence coverage",
                evidence=f"context_claim_recall={context_claim_recall:.2f}",
                recommendation_text="Some required evidence is found, but not enough for complete answers. Try larger top_k, section-level chunks, or multi-query retrieval.",
                next_experiment="Compare --top-k 5 vs --top-k 10 and by_section vs fixed_words.",
                implementation_hint="Prefer coverage-oriented changes first: top-k, chunk overlap, or chunk granularity. Only optimize prompting after coverage improves.",
                    success_signal="Higher context_claim_recall and lower missing_reference_claim_count.",
            )
        )

    if mrr is not None and recall is not None and mrr >= 0.9 and recall < 0.25 and not answer_is_good:
        recs.append(
            recommendation(
                component="retrieval",
                priority="P2",
                issue="Top result is useful but evidence set is incomplete",
                evidence=f"mrr_at_k={mrr:.2f}, recall_at_k={recall:.2f}",
                recommendation_text="The first chunk is relevant but the answer likely needs additional evidence. Increase top_k or add query decomposition.",
                next_experiment="Run --top-k 10 and compare context_claim_recall/context_utilization.",
                implementation_hint="Keep the current retriever, but widen the evidence set so synthesis can combine more than one supporting chunk.",
                success_signal="Higher context_utilization and answer_claim_recall without hurting grounded_claim_ratio.",
            )
        )
    if mrr is not None and recall is not None and mrr < 0.5 and recall >= 0.5:
        recs.append(
            recommendation(
                component="reranking",
                priority="P1",
                issue="Relevant evidence exists but is ranked too low",
                evidence=f"mrr_at_k={mrr:.2f}, recall_at_k={recall:.2f}",
                recommendation_text="Add a reranker or improve score fusion so relevant chunks move into the first positions.",
                next_experiment="Retrieve top 20, rerank to top 5, then compare nDCG and first_relevant_rank.",
                implementation_hint="Apply reranking only after retrieval has already captured enough relevant evidence; otherwise reranking will only reshuffle weak candidates.",
                success_signal="Higher mrr_at_k and ndcg_at_k, with recall_at_k staying stable.",
            )
        )

    if (
        context_claim_recall is not None
        and context_claim_recall >= 0.75
        and grounded_claim_ratio is not None
        and grounded_claim_ratio < 0.65
    ):
        recs.append(
            recommendation(
                component="generation",
                priority="P1",
                issue="Grounding failure despite available evidence",
                evidence=f"context_claim_recall={context_claim_recall:.2f}, grounded_claim_ratio={grounded_claim_ratio:.2f}",
                recommendation_text="The generator receives enough evidence but adds unsupported claims. Strengthen the prompt: answer only from evidence, cite claim-level sources, refuse unsupported details.",
                next_experiment="Same retrieval, compare current prompt vs cite-first/strict-grounding prompt.",
                implementation_hint="Use a claim-by-claim synthesis prompt and explicitly require each claim to be backed by a retrieved chunk or omitted.",
                success_signal="Higher grounded_claim_ratio and lower hallucinated_claim_ratio without reducing answer_claim_recall.",
            )
        )

    if (
        context_claim_recall is not None
        and context_claim_recall >= 0.75
        and context_utilization is not None
        and context_utilization < 0.70
        and not answer_is_good
    ):
        recs.append(
            recommendation(
                component="generation",
                priority="P2",
                issue="Answer incomplete from available context",
                evidence=f"context_claim_recall={context_claim_recall:.2f}, context_utilization={context_utilization:.2f}",
                recommendation_text="The context contains the needed facts, but the answer omits some of them. Improve answer synthesis instructions or ask the model to cover each required claim.",
                next_experiment="Same retrieval, compare concise prompt vs checklist/claim-complete prompt.",
                implementation_hint="Make the generator produce a checklist of required points internally before writing the final answer.",
                success_signal="Higher answer_claim_recall and lower answer_incomplete_from_good_context counts.",
            )
        )

    if context_entities_recall is not None and context_entities_recall < 0.6:
        recs.append(
            recommendation(
                component="retrieval",
                priority="P2",
                issue="Reference entities are not well covered in context",
                evidence=f"context_entities_recall={context_entities_recall:.2f}",
                recommendation_text="Improve entity coverage with larger top-k, more examples, or hierarchy-aware retrieval so key terms and expected answer cues are not missed.",
                next_experiment="Compare current run against enriched examples and a reranked top-10 context.",
                implementation_hint="Add synonyms, normalized answer forms, or structured entity fields to retrieval text instead of relying only on raw prose.",
                success_signal="Higher context_entities_recall and better hit@k / target_doc_retrieved_at_k.",
            )
        )

    if (
        factual_correctness_f1 is not None
        and factual_correctness_f1 < 0.6
        and grounded_claim_ratio is not None
        and grounded_claim_ratio >= 0.7
    ):
        recs.append(
            recommendation(
                component="generation",
                priority="P2",
                issue="Answer is grounded but still not factually complete",
                evidence=f"factual_correctness_f1={factual_correctness_f1:.2f}, grounded_claim_ratio={grounded_claim_ratio:.2f}",
                recommendation_text="Keep the retrieved evidence, but improve claim completeness or final label selection because the answer misses reference facts even when it stays grounded.",
                next_experiment="Compare stricter answer synthesis against a reranked context on the same retrieval results.",
                implementation_hint="Treat this as a completeness problem, not a hallucination problem: preserve grounding constraints but force better coverage of reference claims.",
                success_signal="Higher factual_correctness_f1 and answer_claim_f1 with grounded_claim_ratio remaining high.",
            )
        )

    if (
        grounded_claim_ratio is not None
        and grounded_claim_ratio >= 0.75
        and attribution_f1 is not None
        and attribution_f1 < 0.70
        and not likely_gold_over_specified
    ):
        recs.append(
            recommendation(
                component="attribution",
                priority="P2",
                issue="Weak source attribution",
                evidence=f"grounded_claim_ratio={grounded_claim_ratio:.2f}, evidence_attribution_f1={attribution_f1:.2f}",
                recommendation_text="Claims appear grounded, but source mapping is weak. Require claim-level citations or extract evidence spans before answer generation.",
                next_experiment="Add cite-first answer format and compare evidence_attribution_f1.",
                implementation_hint="Return chunk ids or evidence spans with each generated claim so attribution becomes a first-class output, not a post-hoc reconstruction.",
                success_signal="Higher evidence_attribution_f1 and lower invalid_attribution_count.",
            )
        )
    if invalid_attribution_count > 0:
        recs.append(
            recommendation(
                component="attribution",
                priority="P1",
                issue="Invalid cited chunk IDs",
                evidence=f"invalid_attribution_count={invalid_attribution_count}",
                recommendation_text="The judge/model referenced chunk IDs not present in retrieved context. Constrain citation IDs to the retrieved chunk list.",
                next_experiment="Use structured citation schema with allowed chunk_id enum.",
                implementation_hint="Provide the model only the currently retrieved chunk ids and reject any attribution outside that set.",
                success_signal="invalid_attribution_count moves to zero.",
            )
        )

    if runtime_status in {"weak_evidence", "missing_evidence"} and expected_answerable is not False:
        recs.append(
            recommendation(
                component="runtime_gate",
                priority="P2",
                issue="Runtime evaluator sees weak evidence",
                evidence=f"runtime_retrieval_status={runtime_status}",
                recommendation_text="Do not answer immediately on weak evidence. Use query rewrite/retrieve-more before generation, or abstain for high-risk domains.",
                next_experiment="Run with --abstain-on-weak-evidence and compare false_refusal_rate.",
                implementation_hint="Introduce a two-step policy: weak_evidence -> retrieve_more or manual review; missing_evidence -> abstain.",
                success_signal="Lower unsupported or retrieval_claim_miss cases among weak-evidence questions.",
            )
        )

    if prediction_confidence is not None and prediction_confidence >= 0.85 and auto_flag in {"incorrect", "unsupported"}:
        recs.append(
            recommendation(
                component="calibration",
                priority="P1",
                issue="Overconfident wrong prediction",
                evidence=f"prediction_confidence={prediction_confidence:.2f}, auto_flag={auto_flag}",
                recommendation_text="Calibrate or threshold the classifier score before automatic acceptance because the current confidence is too optimistic on errors.",
                next_experiment="Export confidence bins and compare Brier/ECE before and after reranking or score scaling.",
                implementation_hint="Convert raw scores into review tiers such as auto-accept, human-review, and abstain instead of using them as-is.",
                success_signal="Lower ECE/Brier score and fewer wrong high-confidence decisions.",
            )
        )

    if (
        primary_reason in {"generation_hallucination", "answer_unsupported"}
        and answer_gold_support is not None
        and proxy_faithfulness is not None
        and answer_gold_support >= 0.80
        and proxy_faithfulness >= 0.75
    ):
        recs.append(
            recommendation(
                component="evaluation",
                priority="P1",
                issue="Possible judge/evaluator false negative",
                evidence=f"primary_error_reason={primary_reason}, answer_reference_support={answer_gold_support:.2f}, proxy_faithfulness={proxy_faithfulness:.2f}",
                recommendation_text="The automatic judge conflicts with heuristic support signals. Send this row to manual review before using it as a system failure.",
                next_experiment="Add a human label for this question and calibrate judge thresholds.",
                implementation_hint="Create a small adjudication set for rows where heuristic and judge-based signals strongly disagree.",
                success_signal="Fewer evaluator conflicts and better alignment between manual review and automatic labels.",
            )
        )

    if likely_gold_over_specified:
        recs.append(
            recommendation(
                component="benchmark",
                priority="P2",
                issue="Reference answer may include extra details beyond the question",
                evidence=(
                    f"auto_flag=correct, grounded_claim_ratio={grounded_claim_ratio:.2f}, "
                    f"context_claim_recall={context_claim_recall:.2f}"
                ),
                recommendation_text="The answer appears correct for the asked question, but the reference answer contains additional details that create missing-claim noise. Split optional explanatory details from required answer claims.",
                next_experiment="Add required_claims/optional_claims or shorten the reference answer to the direct answer.",
                implementation_hint="Keep one concise canonical answer plus optional explanatory notes, and do not score optional notes as hard failures.",
                success_signal="Lower benchmark-related false alarms without reducing real retrieval/generation error counts.",
            )
        )

    if expected_answerable is True and not gold_answer:
        recs.append(
            recommendation(
                component="benchmark",
                priority="P1",
                issue="Answerable question lacks a reference answer",
                evidence="expected_answerable=true but reference_answer is empty.",
                recommendation_text="Add a reference answer and expected evidence before trusting correctness or claim-level metrics for this row.",
                next_experiment="Update benchmark annotations: reference answer + expected evidence.",
                implementation_hint="At minimum, provide one concise reference answer and a few expected keywords or evidence cues.",
                success_signal="This question stops generating benchmark warnings and gains stable correctness metrics.",
            )
        )

    if not recs and claim_diagnostic == "claim_level_ok":
        recs.append(
            recommendation(
                component="none",
                priority="P3",
                issue="No immediate issue detected",
                evidence="claim_diagnostic=claim_level_ok",
                recommendation_text="No targeted remediation needed for this question. Keep it as a regression example.",
                next_experiment="Include this row in regression tests for the current configuration.",
            )
        )
    return sorted(recs, key=lambda row: (priority_rank(row["priority"]), row["component"]))


def build_summary_recommendations(summary: Dict[str, object]) -> List[Dict[str, str]]:
    recs: List[Dict[str, str]] = []
    answer_metrics = summary.get("answer_metrics", {}) if isinstance(summary.get("answer_metrics"), dict) else {}
    retrieval_metrics = summary.get("retrieval_metrics", {}) if isinstance(summary.get("retrieval_metrics"), dict) else {}
    diagnostics = summary.get("diagnostics", {}) if isinstance(summary.get("diagnostics"), dict) else {}
    classifier = summary.get("classifier", {}) if isinstance(summary.get("classifier"), dict) else {}
    ranking = classifier.get("ranking_metrics", {}) if isinstance(classifier.get("ranking_metrics"), dict) else {}
    calibration = classifier.get("calibration", {}) if isinstance(classifier.get("calibration"), dict) else {}
    cpv_diagnostics = (
        classifier.get("cpv_diagnostics", {})
        if isinstance(classifier.get("cpv_diagnostics"), dict)
        else {}
    )
    evidence_graph = summary.get("evidence_graph", {}) if isinstance(summary.get("evidence_graph"), dict) else {}
    evidence_rates = evidence_graph.get("rates", {}) if isinstance(evidence_graph.get("rates"), dict) else {}
    evidence_patterns = evidence_graph.get("error_patterns", {}) if isinstance(evidence_graph.get("error_patterns"), dict) else {}
    component_signals = evidence_graph.get("component_signals", {}) if isinstance(evidence_graph.get("component_signals"), dict) else {}
    top_confusion_pairs = evidence_graph.get("top_confusion_pairs", []) if isinstance(evidence_graph.get("top_confusion_pairs"), list) else []
    top_wrong_predicted_codes = evidence_graph.get("top_wrong_predicted_codes", []) if isinstance(evidence_graph.get("top_wrong_predicted_codes"), list) else []

    top1 = as_float(ranking.get("exact_top1_accuracy"))
    hit_at_k = as_float(ranking.get("hit_at_k"))
    hierarchy_similarity = as_float(
        ranking.get("mean_hierarchy_score_top1")
        if ranking.get("mean_hierarchy_score_top1") is not None
        else ranking.get("mean_cpv_hierarchy_similarity_top1")
    )
    ece = as_float(calibration.get("expected_calibration_error"))
    brier = as_float(calibration.get("brier_score"))
    explanation_coverage = as_float(classifier.get("explanation_coverage"))
    latency_ms = as_float(classifier.get("mean_latency_ms"))
    mean_context_claim_recall = as_float(answer_metrics.get("mean_context_claim_recall"))
    mean_factual_f1 = as_float(answer_metrics.get("mean_factual_correctness_f1"))
    mean_grounded = as_float(answer_metrics.get("mean_grounded_claim_ratio"))
    mean_entity_recall = as_float(answer_metrics.get("mean_context_entities_recall"))
    mean_noise = as_float(answer_metrics.get("mean_noise_sensitivity_relevant"))
    mean_mrr = as_float(retrieval_metrics.get("mean_mrr_at_k"))
    mean_recall = as_float(retrieval_metrics.get("mean_recall_at_k"))
    dominant_reason = str(diagnostics.get("most_common_reason") or "")
    dominant_bottleneck = str(cpv_diagnostics.get("dominant_bottleneck") or "")
    display_dominant_bottleneck = _display_bottleneck(dominant_bottleneck)
    gold_present_rate = as_float(cpv_diagnostics.get("gold_present_at_k_rate"))
    high_conf_wrong_rate = as_float(cpv_diagnostics.get("high_confidence_wrong_rate"))
    low_margin_rate = as_float(cpv_diagnostics.get("low_margin_decision_rate"))
    duplicate_pressure_rate = as_float(cpv_diagnostics.get("duplicate_candidate_pressure_rate"))
    low_diversity_rate = as_float(cpv_diagnostics.get("low_diversity_at_k_rate"))
    short_query_rate = as_float(cpv_diagnostics.get("short_or_ambiguous_query_rate"))
    error_duplicate_pressure_rate = as_float(cpv_diagnostics.get("error_duplicate_candidate_pressure_rate"))
    error_low_diversity_rate = as_float(cpv_diagnostics.get("error_low_diversity_at_k_rate"))
    error_short_query_rate = as_float(cpv_diagnostics.get("error_short_or_ambiguous_query_rate"))

    if dominant_bottleneck == "candidate_generation_or_retriever" or (
        gold_present_rate is not None and gold_present_rate < 0.70
    ):
        recs.append(
            recommendation(
                component="retriever",
                priority="P1",
                issue="Candidate coverage limits the maximum score",
                evidence=f"expected_answer_present_at_k={gold_present_rate}, dominant_bottleneck={display_dominant_bottleneck}",
                recommendation_text="Improve candidate generation alongside selection work, because the final selector cannot recover labels that never appear in top-k.",
                next_experiment="Run top-k 10/20 and compare dense, BM25, and hybrid retrieval on expected-answer coverage.",
                implementation_hint="Enrich candidate text with labels, descriptions, parent labels, child labels, examples, and domain synonyms. If the expected answer still does not appear, the selector never had a fair chance.",
                success_signal="Expected-answer coverage and hit@k rise; then top-1 can be improved with reranking.",
            )
        )

    if dominant_bottleneck == "reranker_or_prompt_selection" or (
        top1 is not None and hit_at_k is not None and hit_at_k - top1 >= 0.10
    ):
        recs.append(
            recommendation(
                component="selection",
                priority="P1",
                issue="Retrieved candidates are not being selected well",
                evidence=f"top1={top1}, hit_at_k={hit_at_k}, low_margin_decision_rate={low_margin_rate}",
                recommendation_text="Freeze retrieval and improve the final selection step with reranking or a contrastive prompt.",
                next_experiment="Evaluate the same top-k candidates with a cross-encoder/reranker or a prompt that must compare the top candidates side by side.",
                implementation_hint="This is the likely prompt/reranker failure pattern: the correct answer is available but not promoted to rank 1.",
                success_signal="top-1 accuracy and MRR improve while expected-answer coverage remains stable.",
            )
        )

    if dominant_bottleneck in {"sibling_disambiguation", "hierarchy_disambiguation"}:
        recs.append(
            recommendation(
                component="hierarchy",
                priority="P2",
                issue="Errors are concentrated inside related hierarchy branches",
                evidence=f"dominant_bottleneck={display_dominant_bottleneck}, mean_hierarchy_score_top1={hierarchy_similarity}",
                recommendation_text="Add hierarchy-aware disambiguation for related but still incorrect answers.",
                next_experiment="Build a branch-aware reranking evaluation for rows with close-branch misses.",
                implementation_hint="The system is sometimes directionally close; add exclusions, examples, and contrastive definitions for related alternatives rather than only increasing broad recall.",
                success_signal="Near misses inside the same branch turn into exact matches.",
            )
        )

    if high_conf_wrong_rate is not None and high_conf_wrong_rate >= 0.10:
        recs.append(
            recommendation(
                component="calibration",
                priority="P1",
                issue="Too many wrong predictions are high-confidence",
                evidence=f"high_confidence_wrong_rate={high_conf_wrong_rate:.2f}, ece={ece}",
                recommendation_text="Add confidence calibration and review thresholds before automatic acceptance.",
                next_experiment="Create reliability bins and test auto-accept/manual-review thresholds using confidence, margin, and entropy.",
                implementation_hint="Wrong high-confidence rows are often more dangerous than low-confidence misses; route them to review until calibration improves.",
                success_signal="Lower high_confidence_wrong_rate, ECE, and Brier score.",
            )
        )

    if error_duplicate_pressure_rate is not None and error_duplicate_pressure_rate >= 0.20:
        recs.append(
            recommendation(
                component="retriever",
                priority="P2",
                issue="Top-k candidates contain too many duplicates",
                evidence=f"error_duplicate_candidate_pressure_rate={error_duplicate_pressure_rate:.2f}",
                recommendation_text="Deduplicate repeated candidates in error cases and measure whether unique candidate coverage improves.",
                next_experiment="Compare raw top-k against unique-answer top-k on incorrect rows.",
                implementation_hint="Duplicates matter most when they appear in errors; if duplicates are mostly correct rows, this should not drive the next iteration.",
                success_signal="Higher unique answer count and better hit@k/selection accuracy.",
            )
        )

    if error_low_diversity_rate is not None and error_low_diversity_rate >= 0.40:
        recs.append(
            recommendation(
                component="retriever",
                priority="P2",
                issue="Candidate lists lack hierarchy diversity",
                evidence=f"error_low_diversity_at_k_rate={error_low_diversity_rate:.2f}, overall_low_diversity_at_k_rate={low_diversity_rate}",
                recommendation_text="Measure whether the retriever is over-committing to one hierarchy branch too early.",
                next_experiment="Compare unrestricted retrieval with a diversity-aware candidate set, then rerank both.",
                implementation_hint="Diversity is diagnostic, not always better; use it to discover whether the expected answer's branch is being excluded too early.",
                success_signal="More relevant branches appear in top-k without reducing exact top-1.",
            )
        )

    if error_short_query_rate is not None and error_short_query_rate >= 0.40:
        recs.append(
            recommendation(
                component="benchmark",
                priority="P3",
                issue="Many queries may be under-specified",
                evidence=f"error_short_or_ambiguous_query_rate={error_short_query_rate:.2f}, overall_short_or_ambiguous_query_rate={short_query_rate}",
                recommendation_text="Add richer source context before treating every miss as a model failure.",
                next_experiment="Rerun a subset with title + description + buyer/category metadata and compare failure modes.",
                implementation_hint="Short titles often need domain context or accepted alternative answers to become fair classification examples.",
                success_signal="Manual audit labels fewer rows as ambiguous and top-1 accuracy becomes more stable.",
            )
        )

    if top1 is not None and hit_at_k is not None and hit_at_k - top1 >= 0.15:
        recs.append(
            recommendation(
                component="reranking",
                priority="P1",
                issue="Correct candidate often exists but is not selected as top-1",
                evidence=f"top1={top1:.2f}, hit_at_k={hit_at_k:.2f}",
                recommendation_text="Prioritize a stronger reranking or decision layer before changing the whole retriever.",
                next_experiment="Keep the same candidate generator and compare current ranking against reranked top-10 outputs.",
                implementation_hint="Use a second-stage reranker, hierarchy-aware scoring, or better score fusion before touching retrieval recall.",
                success_signal="top1 rises materially while hit@k stays flat or improves.",
            )
        )

    if hierarchy_similarity is not None and hierarchy_similarity >= 0.70 and top1 is not None and top1 < 0.70:
        recs.append(
            recommendation(
                component="hierarchy",
                priority="P2",
                issue="Most errors are close rather than random",
                evidence=f"mean_hierarchy_score_top1={hierarchy_similarity:.2f}",
                recommendation_text="Exploit this by adding close-candidate disambiguation features instead of only broad retrieval changes.",
                next_experiment="Compare plain ranking against branch-aware reranking inside the predicted branch.",
                implementation_hint="Add label-detail fields, examples, or domain synonyms that distinguish neighboring answers within the same subtree.",
                success_signal="Higher hierarchy similarity and improved top-1 accuracy.",
            )
        )

    if ece is not None and ece >= 0.15:
        recs.append(
            recommendation(
                component="calibration",
                priority="P1",
                issue="Confidence is not reliable enough for automatic decisions",
                evidence=f"expected_calibration_error={ece:.2f}, brier_score={brier}",
                recommendation_text="Introduce review thresholds or post-hoc calibration before using scores operationally.",
                next_experiment="Export reliability bins and compare raw scores against thresholded review tiers.",
                implementation_hint="Map raw confidence to actions such as auto-accept, manual-review, and abstain rather than trusting the raw score directly.",
                success_signal="Lower ECE/Brier score and fewer high-confidence wrong predictions.",
            )
        )

    if explanation_coverage is not None and explanation_coverage < 0.80:
        recs.append(
            recommendation(
                component="explanation",
                priority="P2",
                issue="Not enough explanations are returned for failure analysis",
                evidence=f"explanation_coverage={explanation_coverage:.2f}",
                recommendation_text="Return explanations consistently so ranking failures can be separated from reasoning failures.",
                next_experiment="Require an explanation field for every prediction and compare advisor usefulness before and after.",
                implementation_hint="Always emit a short justification for top-1 and, if possible, a contrastive note against the next-best candidate.",
                success_signal="Explanation coverage approaches 100% and manual analysis becomes faster.",
            )
        )

    if latency_ms is not None and latency_ms > 1500:
        recs.append(
            recommendation(
                component="latency",
                priority="P3",
                issue="Classifier latency may become a deployment bottleneck",
                evidence=f"mean_latency_ms={latency_ms:.0f}",
                recommendation_text="Track quality against latency so improvements do not make the system impractical to use.",
                next_experiment="Benchmark the same classifier with and without optional explanation or reranking stages.",
                implementation_hint="Log p50/p95 latency separately from mean latency and test whether slower steps actually move accuracy or calibration.",
                success_signal="Stable quality with lower or at least justified latency overhead.",
            )
        )

    if (
        mean_context_claim_recall is not None
        and mean_factual_f1 is not None
        and mean_context_claim_recall >= 0.75
        and mean_factual_f1 < 0.60
    ):
        recs.append(
            recommendation(
                component="generation",
                priority="P1",
                issue="Evidence is available, but final answers still miss or distort facts",
                evidence=f"mean_context_claim_recall={mean_context_claim_recall:.2f}, mean_factual_correctness_f1={mean_factual_f1:.2f}",
                recommendation_text="Treat this as a synthesis/completeness problem rather than a retrieval problem.",
                next_experiment="Freeze retrieval and compare stricter synthesis prompts or structured answer templates.",
                implementation_hint="Use claim-complete prompting, cite-first generation, or a checklist-style output format.",
                success_signal="Higher factual_correctness_f1 with grounded_claim_ratio staying high.",
            )
        )

    if (
        mean_context_claim_recall is not None
        and mean_context_claim_recall < 0.45
        and mean_entity_recall is not None
        and mean_entity_recall < 0.60
    ):
        recs.append(
            recommendation(
                component="retrieval",
                priority="P1",
                issue="Coverage is weak at both fact and entity level",
                evidence=f"context_claim_recall={mean_context_claim_recall:.2f}, context_entities_recall={mean_entity_recall:.2f}",
                recommendation_text="Focus on retrieval coverage first; answer-side tuning will have limited effect until more of the right evidence is retrieved.",
                next_experiment="Try enriched examples, higher top-k, and alternative chunk granularities before changing prompts.",
                implementation_hint="Prefer recall-oriented retrieval changes such as example expansion, chunk overlap, metadata filters, and branch-aware retrieval.",
                success_signal="Higher context_claim_recall and context_entities_recall, followed by lower retrieval_claim_miss counts.",
            )
        )

    if mean_noise is not None and mean_noise >= 0.25:
        recs.append(
            recommendation(
                component="robustness",
                priority="P2",
                issue="The system is too sensitive to noisy or weakly relevant context",
                evidence=f"mean_noise_sensitivity_relevant={mean_noise:.2f}",
                recommendation_text="Tighten evidence selection before answer generation so weak matches do not dominate the final answer.",
                next_experiment="Compare current pipeline against a stricter top-k or reranked evidence set.",
                implementation_hint="Use a stronger top-k filter, more selective reranking, or query-rewrite before generation on weak-evidence cases.",
                success_signal="Lower noise sensitivity with equal or better factual correctness.",
            )
        )

    if dominant_reason == "retrieval_claim_miss" and mean_mrr is not None and mean_recall is not None and mean_mrr < 0.5:
        recs.append(
            recommendation(
                component="retrieval",
                priority="P1",
                issue="The dominant bottleneck is missing evidence early in the pipeline",
                evidence=f"most_common_reason={dominant_reason}, mean_mrr_at_k={mean_mrr:.2f}, mean_recall_at_k={mean_recall:.2f}",
                recommendation_text="Do not spend the next iteration on answer prompting first; retrieval is still the limiting factor.",
                next_experiment="Run a retrieval-focused comparison before any new answer-generation changes.",
                implementation_hint="Treat this as a candidate generation and coverage problem first, then revisit synthesis once retrieval stabilizes.",
                success_signal="Retrieval_claim_miss stops dominating and downstream answer metrics improve naturally.",
            )
        )

    parent_child_error_rate = as_float(evidence_rates.get("parent_child_error_rate"))
    sibling_error_rate = as_float(evidence_rates.get("sibling_error_rate"))
    low_margin_same_branch_error_rate = as_float(evidence_rates.get("low_margin_same_branch_error_rate"))
    short_query_gold_missing_rate = as_float(evidence_rates.get("short_query_gold_missing_rate"))
    low_diversity_gold_missing_rate = as_float(evidence_rates.get("low_diversity_gold_missing_rate"))
    duplicate_pressure_error_rate = as_float(evidence_rates.get("duplicate_pressure_error_rate"))
    notice_example_supported_error_rate = as_float(evidence_rates.get("notice_example_supported_error_rate"))
    hierarchy_signal = as_float(component_signals.get("hierarchy"))
    selection_signal = as_float(component_signals.get("selection"))
    retriever_signal = as_float(component_signals.get("retriever"))
    query_enrichment_signal = as_float(component_signals.get("query_enrichment"))
    examples_signal = as_float(component_signals.get("examples"))
    graph_coverage_signal = as_float(component_signals.get("graph_coverage"))
    graph_noise_signal = as_float(component_signals.get("graph_noise"))
    answer_synthesis_signal = as_float(component_signals.get("answer_synthesis"))
    refusal_policy_signal = as_float(component_signals.get("refusal_policy"))
    context_selection_signal = as_float(component_signals.get("context_selection"))
    versioning_signal = as_float(component_signals.get("versioning"))
    bridge_signal = as_float(component_signals.get("bridge_composition"))
    gold_section_missing_error_rate = as_float(evidence_rates.get("gold_section_missing_error_rate"))
    gold_path_missing_error_rate = as_float(evidence_rates.get("gold_path_missing_error_rate"))
    gold_section_present_not_used_error_rate = as_float(evidence_rates.get("gold_section_present_not_used_error_rate"))
    same_branch_wrong_rule_error_rate = as_float(evidence_rates.get("same_branch_wrong_rule_error_rate"))
    wrong_version_branch_error_rate = as_float(evidence_rates.get("wrong_version_branch_error_rate"))
    missing_bridge_fact_error_rate = as_float(evidence_rates.get("missing_bridge_fact_error_rate"))

    if hierarchy_signal is not None and hierarchy_signal >= 0.18:
        recs.append(
            recommendation(
                component="hierarchy",
                priority="P1",
                issue="Evidence graph shows repeated parent-child or sibling confusions",
                evidence=(
                    f"parent_child_error_rate={parent_child_error_rate}, "
                    f"sibling_error_rate={sibling_error_rate}, "
                    f"parent_child_errors={evidence_patterns.get('parent_child_errors')}"
                ),
                recommendation_text="Strengthen hierarchy-aware disambiguation before changing broad retrieval.",
                next_experiment="Compare the current selector against a branch-only or parent-vs-child disambiguation layer on the same candidate pool.",
                implementation_hint="This pattern means the system already reaches the correct taxonomy neighborhood but still resolves the final node incorrectly.",
                success_signal="Parent-child and sibling confusions shrink while hit@k stays roughly stable.",
                source="Graph-aware",
            )
        )

    if selection_signal is not None and selection_signal >= 0.20:
        recs.append(
            recommendation(
                component="selection",
                priority="P1",
                issue="Evidence graph shows many low-margin same-branch failures",
                evidence=f"low_margin_same_branch_error_rate={low_margin_same_branch_error_rate}",
                recommendation_text="Improve the final decision layer on close candidates rather than adding more retrieval breadth first.",
                next_experiment="Run a selective final judge or contrastive reranker only on low-margin or same-branch cases.",
                implementation_hint="These are the cases where the gold code is nearby, but the top candidate wins by an insufficiently discriminative score.",
                success_signal="First ranked candidate correct rises without reducing candidate coverage.",
                source="Graph-aware",
            )
        )

    if retriever_signal is not None and retriever_signal >= 0.12:
        recs.append(
            recommendation(
                component="retriever",
                priority="P2",
                issue="Evidence graph shows gold missing together with low candidate diversity",
                evidence=f"low_diversity_gold_missing_rate={low_diversity_gold_missing_rate}",
                recommendation_text="Tune candidate generation and branch diversity before adding more ranking complexity.",
                next_experiment="Compare current retrieval against a diversity-aware or larger top-k pool, then rerank both with the same selector.",
                implementation_hint="The retriever appears to over-commit to one branch early, so the correct branch may never enter the shortlist.",
                success_signal="Gold-missing cases fall and unique branch coverage improves.",
                source="Graph-aware",
            )
        )

    if query_enrichment_signal is not None and query_enrichment_signal >= 0.10:
        recs.append(
            recommendation(
                component="query_enrichment",
                priority="P2",
                issue="Evidence graph shows many short-query misses where the gold code never enters top-k",
                evidence=f"short_query_gold_missing_rate={short_query_gold_missing_rate}",
                recommendation_text="Apply selective object-focused query enrichment only to short or ambiguous queries.",
                next_experiment="Enrich only short queries with object terms and close product-family synonyms, then compare against the unchanged baseline.",
                implementation_hint="Do not rewrite all queries globally; target only the weak lexical cases where retrieval evidence is too sparse.",
                success_signal="Short-query misses decrease without broad noise increase on well-specified queries.",
                source="Graph-aware",
            )
        )

    if examples_signal is not None and examples_signal >= 0.30:
        top_pair = top_confusion_pairs[0] if top_confusion_pairs else {}
        top_wrong = top_wrong_predicted_codes[0] if top_wrong_predicted_codes else {}
        recs.append(
            recommendation(
                component="examples",
                priority="P2",
                issue="Evidence graph suggests example quality or profile evidence may bias specific wrong codes",
                evidence=(
                    f"notice_example_supported_error_rate={notice_example_supported_error_rate}, "
                    f"top_wrong_predicted_code={top_wrong.get('cpv_code', '')}, "
                    f"top_confusion_pair={top_pair.get('predicted_code', '')}->{top_pair.get('gold_code', '')}"
                ),
                recommendation_text="Inspect and clean the notice examples and profile evidence for the most frequent wrong-vs-gold confusion pairs.",
                next_experiment="Take the top confusion pairs and compare their notice examples side by side before changing global retrieval settings.",
                implementation_hint="If one wrong code repeatedly wins with strong evidence support, its examples or profile text may be too generic or overly attractive.",
                success_signal="The same wrong code stops dominating across repeated related queries.",
                source="Graph-aware",
            )
        )

    if duplicate_pressure_error_rate is not None and duplicate_pressure_error_rate >= 0.15:
        recs.append(
            recommendation(
                component="deduplication",
                priority="P3",
                issue="Evidence graph shows repeated duplicate-candidate pressure in failures",
                evidence=f"duplicate_pressure_error_rate={duplicate_pressure_error_rate}",
                recommendation_text="Collapse near-duplicate candidates before final selection so the shortlist contains more true alternatives.",
                next_experiment="Compare raw top-k against a deduplicated code-level shortlist while keeping the same retrieval scores.",
                implementation_hint="Duplicate pressure is not always the main bottleneck, but when it appears in many errors it can suppress useful alternatives.",
                success_signal="Unique candidate coverage rises and some same-branch misses turn into exact matches.",
                source="Graph-aware",
            )
        )

    if graph_coverage_signal is not None and graph_coverage_signal >= 0.15:
        recs.append(
            recommendation(
                component="graph_coverage",
                priority="P1",
                issue="Graph-aware QA summary shows repeated misses where the required section or evidence path never enters context",
                evidence=(
                    f"gold_section_missing_error_rate={gold_section_missing_error_rate}, "
                    f"gold_path_missing_error_rate={gold_path_missing_error_rate}, "
                    f"missing_relation_evidence_error_rate={evidence_rates.get('missing_relation_evidence_error_rate')}"
                ),
                recommendation_text="Treat these as candidate-generation or evidence-linking failures before changing answer prompts.",
                next_experiment="Compare the same questions with higher top-k, BM25/hybrid retrieval, and section-title-aware reranking while keeping answer generation fixed.",
                implementation_hint="The graph-aware signal is useful here because it distinguishes section-not-found cases from answer-synthesis failures.",
                success_signal="Gold-section and gold-path missing rates decline before answer-side metrics move.",
                source="Graph-aware",
            )
        )

    if context_selection_signal is not None and context_selection_signal >= 0.12:
        recs.append(
            recommendation(
                component="context_selection",
                priority="P2",
                issue="Graph-aware QA summary shows related-rule or cross-version confusions inside the retrieved neighborhood",
                evidence=(
                    f"same_branch_wrong_rule_error_rate={same_branch_wrong_rule_error_rate}, "
                    f"wrong_version_branch_error_rate={wrong_version_branch_error_rate}, "
                    f"cross_doc_pressure_error_rate={evidence_rates.get('cross_doc_pressure_error_rate')}"
                ),
                recommendation_text="Add rule- and version-aware disambiguation rather than only broadening retrieval.",
                next_experiment="Contrast base MPO vs amendment/correction candidates explicitly and rerank same-document competing sections for questions that mention exam forms, deadlines, or conditions.",
                implementation_hint="These failures are often close-rule mistakes such as selecting a related exam form or the wrong regulation version rather than missing the program entirely.",
                success_signal="Same-branch and wrong-version error rates fall without reducing retrieval recall.",
                source="Graph-aware",
            )
        )

    if bridge_signal is not None and bridge_signal >= 0.10:
        recs.append(
            recommendation(
                component="bridge_composition",
                priority="P2",
                issue="Graph-aware QA summary shows missing bridge facts on multi-hop or relation questions",
                evidence=f"missing_bridge_fact_error_rate={missing_bridge_fact_error_rate}",
                recommendation_text="Focus on combining linked facts across sections instead of only retrieving one locally relevant chunk.",
                next_experiment="Evaluate multi-hop questions with KG-organized context or a checklist answer mode that must cover each linked condition explicitly.",
                implementation_hint="This is the QA analogue of a graph bridge miss: one needed fact is present, but the linked consequence or condition is not assembled into the final answer.",
                success_signal="Multi-hop incomplete-answer cases shrink while context-claim recall stays high.",
                source="Graph-aware",
            )
        )

    if answer_synthesis_signal is not None and answer_synthesis_signal >= 0.20:
        recs.append(
            recommendation(
                component="answer_synthesis",
                priority="P1",
                issue="Graph-aware QA summary shows answers failing even when graph-supported evidence is already available",
                evidence=(
                    f"graph_supported_incomplete_error_rate={evidence_rates.get('graph_supported_incomplete_error_rate')}, "
                    f"gold_section_present_not_used_error_rate={gold_section_present_not_used_error_rate}"
                ),
                recommendation_text="Do not interpret all remaining failures as retrieval misses; the answer step is often leaving available evidence unused.",
                next_experiment="Freeze retrieval and compare grounded_llm against cite_first and claim_checklist on the graph-supported failure subset.",
                implementation_hint="This category is especially useful because KG shows that evidence exists, which narrows the problem to synthesis, completeness, or citation behavior.",
                success_signal="Graph-supported incomplete-answer rates fall while retrieval metrics remain roughly stable.",
                source="Graph-aware",
            )
        )

    if refusal_policy_signal is not None and refusal_policy_signal >= 0.12:
        recs.append(
            recommendation(
                component="refusal_policy",
                priority="P2",
                issue="Graph-aware QA summary shows false refusals despite evidence-supported context",
                evidence=f"graph_supported_false_refusal_rate={evidence_rates.get('graph_supported_false_refusal_rate')}",
                recommendation_text="Tighten refusal criteria on cases where the evidence graph already supports the required rule or relation.",
                next_experiment="Audit weak-evidence thresholds only on graph-supported refusal rows and compare with a less conservative abstention policy.",
                implementation_hint="This helps separate genuine low-evidence abstentions from cases where the decision layer is overly cautious after retrieval has already succeeded.",
                success_signal="False refusals decrease without a matching rise in unsupported answers.",
                source="Graph-aware",
            )
        )

    if graph_noise_signal is not None and graph_noise_signal >= 0.20:
        recs.append(
            recommendation(
                component="graph_noise",
                priority="P2",
                issue="Graph-aware QA summary shows noisy KG expansion around wrong answers",
                evidence=f"graph_noise_error_rate={evidence_rates.get('graph_noise_error_rate')}",
                recommendation_text="Constrain graph expansion or rerank graph-added chunks more aggressively before passing them to answer generation.",
                next_experiment="Compare safe_branch against conservative/direct_only profiles and inspect whether graph-only additions still help on the same rows.",
                implementation_hint="Graph-aware analysis is useful here because it identifies when KG is contributing evidence pressure without actually improving the answer path.",
                success_signal="Graph-noise errors fall while KG-added evidence precision rises.",
                source="Graph-aware",
            )
        )

    return sorted(recs, key=lambda row: (priority_rank(row["priority"]), row["component"], row["issue"]))


def apply_question_recommendations(row: Dict[str, object]) -> Dict[str, object]:
    recs = build_question_recommendations(row)
    primary = pick_primary(recs)
    row["recommendations"] = json.dumps(recs, ensure_ascii=False)
    row["recommendation_count"] = len(recs)
    row["recommended_component"] = primary["component"] if primary else ""
    row["recommendation_priority"] = primary["priority"] if primary else ""
    row["recommended_action"] = primary["recommendation"] if primary else ""
    row["recommendation_reason"] = primary["evidence"] if primary else ""
    row["recommended_experiment"] = primary["next_experiment"] if primary else ""
    row["needs_manual_review"] = any(
        rec["component"] == "evaluation" or rec["priority"] == "P0" for rec in recs
    )
    return row


def benchmark_warnings(rows: Sequence[Dict[str, object]]) -> List[str]:
    warnings: List[str] = []
    total = len(rows)
    if total == 0:
        return ["No evaluation rows were produced."]
    answerable_values = [as_bool(row.get("expected_answerable")) for row in rows]
    if all(value is True for value in answerable_values):
        warnings.append("No unanswerable questions are marked in this run; abstention quality cannot be evaluated.")
    no_gold = sum(1 for row in rows if as_bool(row.get("expected_answerable")) is not False and not str(row.get("gold_answer") or "").strip())
    if no_gold:
        warnings.append(f"{no_gold} answerable questions have no reference answer; correctness and claim metrics are less reliable.")
    no_keywords = sum(1 for row in rows if not str(row.get("expected_keywords") or "").strip())
    if no_keywords / total > 0.5:
        warnings.append("More than half of questions have no expected_keywords; retrieval proxy metrics may be weak.")
    return warnings


def build_run_advisor(summary: Dict[str, object], rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    all_recs: List[Dict[str, str]] = []
    for row in rows:
        try:
            recs = json.loads(str(row.get("recommendations") or "[]"))
            if isinstance(recs, list):
                all_recs.extend(rec for rec in recs if isinstance(rec, dict))
        except json.JSONDecodeError:
            continue

    filtered_recs = _non_none_recommendations(all_recs)
    component_counts = Counter(str(rec.get("component", "")) for rec in filtered_recs)
    issue_counts = Counter(str(rec.get("issue", "")) for rec in filtered_recs)
    priority_counts = Counter(str(rec.get("priority", "")) for rec in filtered_recs)

    grouped: Dict[tuple[str, str], Dict[str, object]] = {}
    for rec in filtered_recs:
        key = (str(rec.get("component", "")), str(rec.get("issue", "")))
        item = grouped.setdefault(
            key,
            {
                "component": key[0],
                "issue": key[1],
                "priority": rec.get("priority", "P3"),
                "count": 0,
                "evidence": rec.get("evidence", ""),
                "recommendation": rec.get("recommendation", ""),
                "next_experiment": rec.get("next_experiment", ""),
                "implementation_hint": rec.get("implementation_hint", ""),
                "success_signal": rec.get("success_signal", ""),
            },
        )
        item["count"] = int(item["count"]) + 1
        if priority_rank(str(rec.get("priority"))) < priority_rank(str(item["priority"])):
            item["priority"] = rec.get("priority", item["priority"])
            item["evidence"] = rec.get("evidence", item["evidence"])
            item["recommendation"] = rec.get("recommendation", item["recommendation"])
            item["next_experiment"] = rec.get("next_experiment", item["next_experiment"])
            item["implementation_hint"] = rec.get("implementation_hint", item["implementation_hint"])
            item["success_signal"] = rec.get("success_signal", item["success_signal"])

    top_recommendations = sorted(
        grouped.values(),
        key=lambda row: (priority_rank(str(row["priority"])), -int(row["count"]), str(row["component"])),
    )[:8]
    summary_recommendations = build_summary_recommendations(summary)

    answer_metrics = summary.get("answer_metrics", {}) if isinstance(summary.get("answer_metrics"), dict) else {}
    retrieval_metrics = summary.get("retrieval_metrics", {}) if isinstance(summary.get("retrieval_metrics"), dict) else {}
    prediction_calibration = _resolve_prediction_calibration(summary)
    classifier = summary.get("classifier", {}) if isinstance(summary.get("classifier"), dict) else {}
    is_cpv_classifier = classifier.get("type") in {"ted_cpv", "api_classifier", "prepared_rag_results"}
    cpv_diagnostics = (
        classifier.get("cpv_diagnostics", {})
        if isinstance(classifier.get("cpv_diagnostics"), dict)
        else {}
    )
    evidence_graph = summary.get("evidence_graph", {}) if isinstance(summary.get("evidence_graph"), dict) else {}
    evidence_rates = evidence_graph.get("rates", {}) if isinstance(evidence_graph.get("rates"), dict) else {}
    warnings = [] if is_cpv_classifier else benchmark_warnings(rows)
    health = {
        "correct": summary.get("n_correct"),
        "incorrect": summary.get("n_incorrect"),
        "mean_mrr_at_k": retrieval_metrics.get("mean_mrr_at_k"),
        "mean_ndcg_at_k": retrieval_metrics.get("mean_ndcg_at_k"),
        "prediction_brier_score": prediction_calibration.get("brier_score"),
        "prediction_ece": prediction_calibration.get("expected_calibration_error"),
    }
    if not is_cpv_classifier:
        health.update(
            {
                "mean_context_claim_recall": answer_metrics.get("mean_context_claim_recall"),
                "mean_factual_correctness_f1": answer_metrics.get("mean_factual_correctness_f1"),
                "mean_grounded_claim_ratio": answer_metrics.get("mean_grounded_claim_ratio"),
                "mean_context_entities_recall": answer_metrics.get("mean_context_entities_recall"),
                "mean_noise_sensitivity_relevant": answer_metrics.get("mean_noise_sensitivity_relevant"),
                "mean_evidence_attribution_f1": answer_metrics.get("mean_evidence_attribution_f1"),
                "over_answering_rate": answer_metrics.get("over_answering_rate"),
                "false_refusal_rate": answer_metrics.get("false_refusal_rate"),
            }
        )
    else:
        health.update(
            {
                "expected_answer_present_at_k_rate": cpv_diagnostics.get("gold_present_at_k_rate"),
                "high_confidence_wrong_rate": cpv_diagnostics.get("high_confidence_wrong_rate"),
                "low_margin_decision_rate": cpv_diagnostics.get("low_margin_decision_rate"),
                "mean_unique_answers_at_k": cpv_diagnostics.get("mean_unique_cpv_at_k"),
                "dominant_bottleneck": cpv_diagnostics.get("dominant_bottleneck"),
                "evidence_graph_parent_child_error_rate": evidence_rates.get("parent_child_error_rate"),
                "evidence_graph_sibling_error_rate": evidence_rates.get("sibling_error_rate"),
                "evidence_graph_low_margin_same_branch_error_rate": evidence_rates.get("low_margin_same_branch_error_rate"),
                "evidence_graph_short_query_gold_missing_rate": evidence_rates.get("short_query_gold_missing_rate"),
            }
        )
    return {
        "n_questions": len(rows),
        "component_counts": dict(component_counts),
        "issue_counts": dict(issue_counts),
        "priority_counts": dict(priority_counts),
        "top_recommendations": top_recommendations,
        "summary_recommendations": summary_recommendations,
        "benchmark_warnings": warnings,
        "health": health,
        "evidence_graph": evidence_graph,
    }


def write_quality_report(path: str, advisor: Dict[str, object], summary: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append(f"# RAG Evaluation Quality Report: {summary.get('experiment', 'run')}")
    lines.append("")
    lines.append("## Executive Summary")
    health = advisor.get("health", {}) if isinstance(advisor.get("health"), dict) else {}
    lines.append(f"- Questions: {advisor.get('n_questions', 0)}")
    for label, key in [
        ("Correct", "correct"),
        ("Incorrect", "incorrect"),
        ("Mean context claim recall", "mean_context_claim_recall"),
        ("Mean factual correctness F1", "mean_factual_correctness_f1"),
        ("Mean grounded claim ratio", "mean_grounded_claim_ratio"),
        ("Mean context entities recall", "mean_context_entities_recall"),
        ("Mean noise sensitivity (relevant)", "mean_noise_sensitivity_relevant"),
        ("Mean evidence attribution F1", "mean_evidence_attribution_f1"),
        ("Mean MRR@K", "mean_mrr_at_k"),
        ("Mean nDCG@K", "mean_ndcg_at_k"),
        ("Expected answer present@K rate", "expected_answer_present_at_k_rate"),
        ("High-confidence wrong rate", "high_confidence_wrong_rate"),
        ("Low-margin decision rate", "low_margin_decision_rate"),
        ("Mean unique answers@K", "mean_unique_answers_at_k"),
        ("Dominant bottleneck", "dominant_bottleneck"),
        ("Prediction Brier score", "prediction_brier_score"),
        ("Prediction ECE", "prediction_ece"),
    ]:
        if health.get(key) is not None:
            lines.append(f"- {label}: {health.get(key)}")
    lines.append("")

    warnings = advisor.get("benchmark_warnings", [])
    if warnings:
        lines.append("## Benchmark Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    classifier = summary.get("classifier", {}) if isinstance(summary.get("classifier"), dict) else {}
    cpv_diagnostics = (
        classifier.get("cpv_diagnostics", {})
        if isinstance(classifier.get("cpv_diagnostics"), dict)
        else {}
    )
    helpful_data = cpv_diagnostics.get("additional_data_that_would_help", [])
    if isinstance(helpful_data, list) and helpful_data:
        lines.append("## Additional Data That Would Help")
        for item in helpful_data:
            if isinstance(item, dict):
                lines.append(f"- {item.get('data')}: {item.get('why')}")
        lines.append("")

    top = advisor.get("top_recommendations", [])
    summary_top = advisor.get("summary_recommendations", [])
    if summary_top:
        lines.append("## Run-Level Recommendations")
        for rec in summary_top:
            _append_recommendation_block(lines, rec)
        lines.append("")

    lines.append("## Top Recommendations")
    if not top:
        lines.append("- No major remediation recommendation was generated.")
    else:
        for rec in top:
            _append_recommendation_block(lines, rec, include_count=True)
    lines.append("")

    lines.append("## Counts")
    for title, key in [
        ("Components", "component_counts"),
        ("Issues", "issue_counts"),
        ("Priorities", "priority_counts"),
    ]:
        counts = advisor.get(key, {})
        _append_counts_section(lines, title, counts if isinstance(counts, dict) else {})

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path
