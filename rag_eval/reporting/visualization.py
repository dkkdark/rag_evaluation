from __future__ import annotations

import html
import os
from statistics import mean
from typing import Dict, List, Sequence

from rag_eval.evaluation.metrics import is_relevant_grade


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_scores(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if abs(high - low) < 1e-12:
        return [1.0 if high > 0 else 0.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def _slug_label(summary: Dict) -> str:
    chunking = summary["chunking_strategy"]
    retriever = summary["retriever"]
    chunk_size = summary["chunk_size"]
    overlap = summary["chunk_overlap"]
    return f"{chunking} + {retriever} (size={chunk_size}, overlap={overlap})"


def _correctness_rate(summary: Dict) -> float:
    n_questions = max(int(summary.get("n_questions", 0)), 1)
    partial = int(summary.get("answer_metrics", {}).get("n_partially_correct", 0))
    return (
        int(summary.get("n_correct", 0)) + 0.5 * partial
    ) / n_questions


def _retrieval_score(summary: Dict) -> float:
    retrieval = summary.get("retrieval_metrics", {})
    ndcg = float(retrieval.get("mean_ndcg_at_k") or 0.0)
    mrr = float(retrieval.get("mean_mrr_at_k") or 0.0)
    ragas = float(retrieval.get("mean_ragas_recall_at_k") or 0.0)
    return 0.4 * ndcg + 0.3 * mrr + 0.3 * ragas


def _rank_profile(
    retrieved_rows: Sequence[Dict],
    *,
    top_k: int,
    unique_relevance: bool = False,
) -> tuple[List[int], List[float], List[float], List[int], List[int]]:
    grouped: Dict[str, List[Dict]] = {}
    for row in retrieved_rows:
        question_id = str(row.get("question_id", ""))
        score = _safe_float(row.get("score"))
        rank = row.get("rank")
        if not question_id or score is None or rank is None:
            continue
        grouped.setdefault(question_id, []).append(dict(row))

    per_rank_scores: Dict[int, List[float]] = {}
    per_rank_relevance: Dict[int, List[float]] = {}
    per_rank_relevant_counts: Dict[int, int] = {}
    per_rank_total_counts: Dict[int, int] = {}
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda item: int(item["rank"]))
        normalized = _normalize_scores([float(item["score"]) for item in ordered])
        seen_relevance_keys: set[str] = set()
        for item, normalized_score in zip(ordered, normalized):
            rank = int(item["rank"])
            if rank > top_k:
                continue
            relevance_grade = _safe_float(item.get("relevance_grade"))
            is_relevant = _truthy_flag(item.get("is_relevant")) or is_relevant_grade(relevance_grade)
            if unique_relevance and is_relevant:
                relevance_key = _relevance_identity(item)
                if relevance_key in seen_relevance_keys:
                    is_relevant = False
                else:
                    seen_relevance_keys.add(relevance_key)
            per_rank_scores.setdefault(rank, []).append(normalized_score)
            per_rank_relevance.setdefault(rank, []).append(1.0 if is_relevant else 0.0)
            per_rank_relevant_counts[rank] = per_rank_relevant_counts.get(rank, 0) + (1 if is_relevant else 0)
            per_rank_total_counts[rank] = per_rank_total_counts.get(rank, 0) + 1

    if not per_rank_scores:
        return [], [], [], [], []

    ranks = sorted(per_rank_scores)
    avg_scores = [mean(per_rank_scores[rank]) for rank in ranks]
    relevant_rates = [mean(per_rank_relevance.get(rank, [0.0])) for rank in ranks]
    relevant_counts = [per_rank_relevant_counts.get(rank, 0) for rank in ranks]
    total_counts = [per_rank_total_counts.get(rank, 0) for rank in ranks]
    return ranks, avg_scores, relevant_rates, relevant_counts, total_counts


def _relevance_identity(row: Dict) -> str:
    for key in ["cpv_code", "doc_id", "chunk_id"]:
        value = str(row.get(key, "") or "").strip()
        if value:
            return f"{key}:{value}"
    return f"rank:{row.get('rank', '')}"


def _truthy_flag(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def write_strategy_score_profile_svg(
    retrieved_rows: Sequence[Dict],
    output_path: str,
    *,
    experiment_slug: str,
    top_k: int,
) -> str | None:
    ranks, avg_scores, _, _, _ = _rank_profile(retrieved_rows, top_k=top_k)
    if not ranks:
        return None

    width = 1100
    height = 640
    margin_left = 110
    margin_right = 70
    margin_top = 110
    margin_bottom = 110
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    n_bars = len(ranks)
    gap = min(18, max(6, plot_width // max(n_bars * 8, 1)))
    bar_width = max(10, (plot_width - gap * max(n_bars - 1, 0)) / max(n_bars, 1))

    def x_for_index(index: int) -> float:
        return margin_left + index * (bar_width + gap)

    def y_for_score(score: float) -> float:
        return margin_top + (1.0 - score) * plot_height

    bars: List[str] = []
    labels: List[str] = []
    for index, (rank, score) in enumerate(zip(ranks, avg_scores)):
        x = x_for_index(index)
        y = y_for_score(score)
        bar_height = max(4.0, margin_top + plot_height - y)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
            'rx="7" fill="#68a8ff" fill-opacity="0.96" />'
        )
        labels.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{margin_top + plot_height + 34:.1f}" '
            f'text-anchor="middle" font-size="20" fill="#5d6270">#{rank}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Retrieval score profile for {html.escape(experiment_slug)}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#fffdf8" />
      <stop offset="100%" stop-color="#f7f4ec" />
    </linearGradient>
    <pattern id="dots" width="18" height="18" patternUnits="userSpaceOnUse">
      <circle cx="2" cy="2" r="1.15" fill="#eadfca" fill-opacity="0.75" />
    </pattern>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#bg)" />
  <rect width="{width}" height="{height}" fill="url(#dots)" />
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#8d8d8d" stroke-width="3" />
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left}" y2="{margin_top}" stroke="#8d8d8d" stroke-width="3" />
  <text x="{width - margin_right - 20}" y="{margin_top + plot_height + 60}" font-size="24" fill="#7c7f87" font-family="Helvetica, Arial, sans-serif">Chunks</text>
  <text x="36" y="{margin_top + 52}" transform="rotate(-90 36 {margin_top + 52})" font-size="24" fill="#7c7f87" font-family="Helvetica, Arial, sans-serif">Score</text>
  {''.join(bars)}
  {''.join(labels)}
</svg>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    return output_path


def write_chunk_relevance_comparison_svg(
    retrieved_rows: Sequence[Dict],
    output_path: str,
    *,
    top_k: int,
    chart_label: str,
    unique_relevance: bool = False,
) -> str | None:
    ranks, avg_scores, relevant_rates, relevant_counts, total_counts = _rank_profile(
        retrieved_rows, top_k=top_k, unique_relevance=unique_relevance
    )
    if not ranks:
        return None

    width = 1120
    height = 660
    margin_left = 110
    margin_right = 70
    margin_top = 110
    margin_bottom = 120
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    n_groups = len(ranks)
    gap = min(20, max(8, plot_width // max(n_groups * 9, 1)))
    group_width = max(26, (plot_width - gap * max(n_groups - 1, 0)) / max(n_groups, 1))
    bar_width = max(10.0, (group_width - 10.0) / 2.0)

    def x_for_index(index: int) -> float:
        return margin_left + index * (group_width + gap)

    def y_for_value(value: float) -> float:
        return margin_top + (1.0 - max(0.0, min(1.0, value))) * plot_height

    bars: List[str] = []
    labels: List[str] = []
    captions: List[str] = []
    for index, rank in enumerate(ranks):
        group_x = x_for_index(index)
        score = avg_scores[index]
        relevant_rate = relevant_rates[index]
        score_y = y_for_value(score)
        relevant_y = y_for_value(relevant_rate)
        score_height = max(4.0, margin_top + plot_height - score_y)
        relevant_height = max(4.0, margin_top + plot_height - relevant_y)
        bars.append(
            f'<rect x="{group_x:.1f}" y="{score_y:.1f}" width="{bar_width:.1f}" height="{score_height:.1f}" rx="6" fill="#68a8ff" fill-opacity="0.96" />'
        )
        bars.append(
            f'<rect x="{group_x + bar_width + 10:.1f}" y="{relevant_y:.1f}" width="{bar_width:.1f}" height="{relevant_height:.1f}" rx="6" fill="#58c28c" fill-opacity="0.96" />'
        )
        labels.append(
            f'<text x="{group_x + group_width / 2:.1f}" y="{margin_top + plot_height + 34:.1f}" text-anchor="middle" font-size="20" fill="#5d6270">#{rank}</text>'
        )
        captions.append(
            f'<text x="{group_x + group_width / 2:.1f}" y="{margin_top + plot_height + 58:.1f}" text-anchor="middle" font-size="14" fill="#808696">{relevant_counts[index]}/{total_counts[index]}</text>'
        )

    legend_x = width - 355
    legend_y = 42
    title = "Unique Relevant Chunks" if unique_relevance else "Scored vs Correct Chunks"
    caption = "green labels = unique relevant/total" if unique_relevance else "green labels = relevant/total"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Chunk score vs relevant chunk profile for {html.escape(chart_label)}">
  <rect width="{width}" height="{height}" fill="#fffdf8" />
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#8d8d8d" stroke-width="3" />
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left}" y2="{margin_top}" stroke="#8d8d8d" stroke-width="3" />
  <text x="28" y="42" font-size="28" fill="#444b57" font-family="Helvetica, Arial, sans-serif">{html.escape(title)}</text>
  <text x="28" y="74" font-size="18" fill="#7b808a" font-family="Helvetica, Arial, sans-serif">{html.escape(chart_label)}</text>
  <rect x="{legend_x}" y="{legend_y}" width="20" height="20" rx="5" fill="#68a8ff" />
  <text x="{legend_x + 30}" y="{legend_y + 16}" font-size="18" fill="#4f5665" font-family="Helvetica, Arial, sans-serif">avg normalized score</text>
  <rect x="{legend_x}" y="{legend_y + 30}" width="20" height="20" rx="5" fill="#58c28c" />
  <text x="{legend_x + 30}" y="{legend_y + 46}" font-size="18" fill="#4f5665" font-family="Helvetica, Arial, sans-serif">relevant chunk rate</text>
  <text x="{width - margin_right - 35}" y="{margin_top + plot_height + 86}" font-size="18" fill="#7c7f87" font-family="Helvetica, Arial, sans-serif">{html.escape(caption)}</text>
  <text x="{width - margin_right - 20}" y="{margin_top + plot_height + 60}" font-size="24" fill="#7c7f87" font-family="Helvetica, Arial, sans-serif">Chunks</text>
  <text x="36" y="{margin_top + 52}" transform="rotate(-90 36 {margin_top + 52})" font-size="24" fill="#7c7f87" font-family="Helvetica, Arial, sans-serif">Rate / Score</text>
  {''.join(bars)}
  {''.join(labels)}
  {''.join(captions)}
</svg>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    return output_path


def write_strategy_metric_overview_svg(summary: Dict, output_path: str) -> str:
    answer = float(summary.get("answer_metrics", {}).get("mean_factual_correctness_f1") or 0.0)
    retrieval = _retrieval_score(summary)
    correctness = _correctness_rate(summary)
    context = float(summary.get("answer_metrics", {}).get("mean_context_entities_recall") or 0.0)
    grounded = float(summary.get("answer_metrics", {}).get("mean_grounded_claim_ratio") or 0.0)

    metrics = [
        ("Factual correctness", answer, "#68a8ff"),
        ("Retrieval quality", retrieval, "#58c28c"),
        ("Correctness", correctness, "#f1b54c"),
        ("Context entity recall", context, "#ef7d8f"),
        ("Grounded claims", grounded, "#8f9cff"),
    ]

    width = 900
    height = 420
    start_x = 290
    bar_max = 510
    top_y = 72
    row_gap = 62

    bars: List[str] = []
    for idx, (label, value, color) in enumerate(metrics):
        y = top_y + idx * row_gap
        bars.append(
            f'<text x="36" y="{y + 20}" font-size="22" fill="#4f5665" font-family="Helvetica, Arial, sans-serif">{html.escape(label)}</text>'
            f'<rect x="{start_x}" y="{y}" width="{bar_max}" height="26" rx="13" fill="#e9e5da" />'
            f'<rect x="{start_x}" y="{y}" width="{bar_max * max(0.0, min(1.0, value)):.1f}" height="26" rx="13" fill="{color}" />'
            f'<text x="{start_x + bar_max + 18}" y="{y + 20}" font-size="21" fill="#4f5665" font-family="Helvetica, Arial, sans-serif">{value:.2f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Strategy metric overview">
  <rect width="{width}" height="{height}" fill="#fffdf8" />
  <text x="36" y="36" font-size="28" fill="#444b57" font-family="Helvetica, Arial, sans-serif">Strategy KPI Overview</text>
  <text x="36" y="392" font-size="18" fill="#7b808a" font-family="Helvetica, Arial, sans-serif">{html.escape(_slug_label(summary))}</text>
  {''.join(bars)}
</svg>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    return output_path


def write_strategy_diagnostics_svg(summary: Dict, output_path: str) -> str | None:
    counts = summary.get("diagnostics", {}).get("counts_by_primary_reason", {}) or {}
    if not counts:
        return None

    items = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
    max_count = max(count for _, count in items) or 1
    width = 920
    height = 130 + 70 * len(items)
    left = 270
    bar_max = 560

    rows: List[str] = []
    for idx, (label, count) in enumerate(items):
        y = 60 + idx * 70
        rows.append(
            f'<text x="28" y="{y + 22}" font-size="20" fill="#4f5665" font-family="Helvetica, Arial, sans-serif">{html.escape(label)}</text>'
            f'<rect x="{left}" y="{y}" width="{bar_max}" height="28" rx="14" fill="#ece7db" />'
            f'<rect x="{left}" y="{y}" width="{bar_max * (count / max_count):.1f}" height="28" rx="14" fill="#68a8ff" />'
            f'<text x="{left + bar_max + 16}" y="{y + 22}" font-size="20" fill="#4f5665" font-family="Helvetica, Arial, sans-serif">{count}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Strategy diagnostics">
  <rect width="{width}" height="{height}" fill="#fffdf8" />
  <text x="28" y="34" font-size="28" fill="#444b57" font-family="Helvetica, Arial, sans-serif">Main Failure Reasons</text>
  {''.join(rows)}
</svg>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    return output_path


def build_strategy_improvement_summary(summary: Dict) -> Dict[str, object]:
    answer_metrics = summary.get("answer_metrics", {})
    retrieval = summary.get("retrieval_metrics", {})
    reason = summary.get("diagnostics", {}).get("most_common_reason")
    strategy = summary.get("chunking_strategy")
    retriever = summary.get("retriever")
    overlap = int(summary.get("chunk_overlap") or 0)
    llm_enabled = bool(summary.get("llm", {}).get("enabled"))
    reranker_enabled = bool(summary.get("reranker", {}).get("enabled"))

    correctness = _correctness_rate(summary)
    retrieval_score = _retrieval_score(summary)
    context = float(answer_metrics.get("mean_context_entities_recall") or 0.0)
    factual = float(answer_metrics.get("mean_factual_correctness_f1") or 0.0)
    noise = float(answer_metrics.get("mean_noise_sensitivity_relevant") or 0.0)
    mrr = float(retrieval.get("mean_mrr_at_k") or 0.0)
    ndcg = float(retrieval.get("mean_ndcg_at_k") or 0.0)
    recall = float(retrieval.get("mean_recall_at_k") or 0.0)
    ragas = float(retrieval.get("mean_ragas_recall_at_k") or 0.0)

    improvements: List[str] = []
    blockers: List[str] = []

    if mrr < 0.6 or ndcg < 0.55:
        blockers.append("relevant chunks are not ranked high enough")
        improvements.append(
            "Improve ranking strength: try `hybrid` or `bm25`, and compare against smaller chunk sizes."
        )
    if recall < 0.2 or ragas < 0.65:
        blockers.append("the retrieved context covers too little of the required evidence")
        improvements.append(
            "Increase coverage: raise `top-k`, add overlap, and compare with larger chunks or `by_section`."
        )
    if context < 0.65:
        blockers.append("the retrieved context contains too much weak or noisy text")
        improvements.append(
            "Make the context more precise: reduce chunk size or switch to `by_paragraph` if chunks are currently too broad."
        )
    if factual < 0.6 and retrieval_score >= 0.6:
        blockers.append("retrieval is decent, but final factual completeness is still limited")
        improvements.append(
            "Improve synthesis or reranking: the system is close to the right evidence but does not consistently produce fully correct answers."
        )
    if noise >= 0.25:
        blockers.append("answers are too sensitive to noisy or weakly matching context")
        improvements.append(
            "Reduce noise sensitivity: keep only the strongest chunks or add a second-stage reranker before generation."
        )
    if reason in {"answer_synthesis_failure", "answer_incomplete_from_good_context"}:
        blockers.append("retrieval is reasonably strong, but the final answer is not synthesized well from the retrieved context")
        improvements.append(
            "Improve answer synthesis: enable LLM generation or use larger chunks so supporting facts are not split apart."
        )
    if reason in {"retrieval_miss", "wrong_chunks_ranked_high", "partial_retrieval"}:
        improvements.append(
            "Focus on retrieval first: revisit retriever and chunking together instead of only changing prompting or answer generation."
        )
    if not llm_enabled and correctness < 0.5:
        improvements.append(
            "Enable `--llm-enable` if the goal includes final answer quality, not only retrieval evaluation."
        )
    if strategy == "by_paragraph":
        improvements.append(
            "For `by_paragraph`, compare against `fixed_words` or `by_section` if the context is too fragmented."
        )
    if strategy == "by_section" and context < 0.7:
        improvements.append(
            "For `by_section`, sections may be too broad: compare with `fixed_words` or `fixed_tokens` for more precise targeting."
        )
    if strategy in {"fixed_words", "fixed_tokens"} and overlap == 0 and recall < 0.3:
        improvements.append(
            "Add overlap so facts near chunk boundaries are less likely to be lost."
        )
    if not reranker_enabled and mrr < 0.6 and recall >= 0.4:
        improvements.append(
            "Turn on the lightweight reranker: relevant evidence exists, but it likely needs a better final ordering."
        )
    if retriever == "tfidf" and ragas < 0.7:
        improvements.append(
            "Test `bm25` or `hybrid` if questions are often phrased differently from the source document."
        )
    if retriever == "dense" and ndcg < 0.55:
        improvements.append(
            "For `dense`, test `hybrid` if exact legal or policy terminology matters."
        )

    improvements = list(dict.fromkeys(improvements))
    blockers = list(dict.fromkeys(blockers))

    successful = correctness >= 0.7 and retrieval_score >= 0.7 and context >= 0.7
    if successful:
        headline = "This strategy looks stable."
        if not improvements:
            improvements.append("Keep this configuration as a strong baseline and only fine-tune `top-k` and overlap.")
    else:
        headline = "This strategy does not look stable yet."
        if not blockers:
            blockers.append("the main bottleneck is only visible in aggregate metrics and not as a single dominant failure mode")
        if not improvements:
            improvements.append("Compare this setup with an alternative retriever and a nearby chunk size to isolate the bottleneck.")

    return {
        "headline": headline,
        "successful": successful,
        "main_blockers": blockers,
        "improvements": improvements,
        "snapshot": {
            "correctness_rate": correctness,
            "retrieval_score": retrieval_score,
            "context_relevance": context,
            "most_common_reason": reason,
        },
    }


def write_strategy_showcase_md(
    summary: Dict,
    output_path: str,
    *,
    score_profile_svg: str | None,
    metric_overview_svg: str | None,
    diagnostics_svg: str | None,
    recommendation: Dict[str, object],
) -> str:
    answer_metrics = summary.get("answer_metrics", {})
    diagnostics = summary.get("diagnostics", {})
    counts = diagnostics.get("counts_by_primary_reason", {}) or {}
    sorted_reasons = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    lines = [
        f"# Strategy Showcase: {summary['experiment']}",
        "",
        f'"n_correct": {int(summary.get("n_correct", 0))},',
        f'"n_partially_correct": {int(answer_metrics.get("n_partially_correct", 0))},',
        f'"n_incorrect": {int(answer_metrics.get("n_incorrect", 0))}',
        "",
        "## Reason Analysis",
        "",
    ]

    if sorted_reasons:
        dominant_reason = recommendation["snapshot"]["most_common_reason"]
        lines.append(
            f"The dominant failure mode is `{dominant_reason}`, which suggests the main bottleneck is currently concentrated in one repeated error pattern."
        )
        lines.append("")
        for reason, count in sorted_reasons:
            lines.append(f'- "{reason}": {count}')
        if recommendation["main_blockers"]:
            lines.append("")
            lines.append("Interpretation:")
            for item in recommendation["main_blockers"]:
                lines.append(f"- {item}")
        if recommendation["improvements"]:
            lines.append("")
            lines.append("What this implies:")
            for item in recommendation["improvements"]:
                lines.append(f"- {item}")
    else:
        lines.append("No diagnostic reasons were recorded for this strategy.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return output_path


def write_strategy_showcase_bundle(
    *,
    summary: Dict,
    retrieved_rows: Sequence[Dict],
    experiment_dir: str,
) -> Dict[str, object]:
    score_profile_svg = write_strategy_score_profile_svg(
        retrieved_rows,
        os.path.join(experiment_dir, "strategy_score_profile.svg"),
        experiment_slug=summary["experiment"],
        top_k=int(summary["top_k"]),
    )
    metric_overview_svg = write_strategy_metric_overview_svg(
        summary, os.path.join(experiment_dir, "strategy_metric_overview.svg")
    )
    chunk_alignment_svg = write_chunk_relevance_comparison_svg(
        retrieved_rows,
        os.path.join(experiment_dir, "strategy_chunk_alignment.svg"),
        top_k=int(summary["top_k"]),
        chart_label=summary["experiment"],
    )
    unique_chunk_alignment_svg = write_chunk_relevance_comparison_svg(
        retrieved_rows,
        os.path.join(experiment_dir, "strategy_unique_chunk_alignment.svg"),
        top_k=int(summary["top_k"]),
        chart_label=summary["experiment"],
        unique_relevance=True,
    )
    diagnostics_svg = write_strategy_diagnostics_svg(
        summary, os.path.join(experiment_dir, "strategy_diagnostics.svg")
    )
    recommendation = build_strategy_improvement_summary(summary)
    showcase_md = write_strategy_showcase_md(
        summary,
        os.path.join(experiment_dir, "strategy_showcase.md"),
        score_profile_svg=score_profile_svg,
        metric_overview_svg=metric_overview_svg,
        diagnostics_svg=diagnostics_svg,
        recommendation=recommendation,
    )

    return {
        "enabled": True,
        "score_profile_svg": score_profile_svg,
        "chunk_alignment_svg": chunk_alignment_svg,
        "unique_chunk_alignment_svg": unique_chunk_alignment_svg,
        "metric_overview_svg": metric_overview_svg,
        "diagnostics_svg": diagnostics_svg,
        "showcase_md": showcase_md,
        "improvement_summary": recommendation,
    }


def write_run_showcase_index(experiment_summaries: Sequence[Dict], output_path: str) -> str:
    lines = ["# Strategy Showcase Index", ""]
    for summary in experiment_summaries:
        showcase = summary.get("showcase", {})
        if not showcase.get("enabled"):
            continue
        lines.append(f"## {summary['experiment']}")
        lines.append("")
        lines.append(
            f"- config: `{summary['chunking_strategy']}` + `{summary['retriever']}`"
        )
        headline = showcase.get("improvement_summary", {}).get("headline")
        if headline:
            lines.append(f"- summary: {headline}")
        showcase_md = showcase.get("showcase_md")
        if showcase_md:
            lines.append(f"- report: `{os.path.relpath(showcase_md, os.path.dirname(output_path))}`")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return output_path


def write_classifier_metric_overview_svg(summary: Dict, output_path: str) -> str:
    classifier = summary.get("classifier", {})
    retrieval = summary.get("retrieval_metrics", {})
    answer = summary.get("answer_metrics", {})
    ranking = classifier.get("ranking_metrics", {})
    hierarchy_score = ranking.get("mean_hierarchy_score_top1")
    if hierarchy_score is None:
        hierarchy_score = ranking.get("mean_cpv_hierarchy_similarity_top1")
    metrics = [
        ("Top-1 accuracy", float(ranking.get("exact_top1_accuracy") or 0.0), "#68a8ff"),
        ("Hierarchy score", float(hierarchy_score or 0.0), "#58c28c"),
        ("Factual correctness", float(answer.get("mean_factual_correctness_f1") or 0.0), "#f1b54c"),
        ("Calibration (1-ECE)", max(0.0, 1.0 - float(classifier.get("calibration", {}).get("expected_calibration_error") or 0.0)), "#ef7d8f"),
        ("Grounded claims", float(answer.get("mean_grounded_claim_ratio") or 0.0), "#8f9cff"),
    ]

    width = 900
    height = 420
    start_x = 290
    bar_max = 510
    top_y = 72
    row_gap = 62

    bars: List[str] = []
    for idx, (label, value, color) in enumerate(metrics):
        y = top_y + idx * row_gap
        normalized = max(0.0, min(1.0, value))
        bars.append(
            f'<text x="36" y="{y + 20}" font-size="22" fill="#4f5665" font-family="Helvetica, Arial, sans-serif">{html.escape(label)}</text>'
            f'<rect x="{start_x}" y="{y}" width="{bar_max}" height="26" rx="13" fill="#e9e5da" />'
            f'<rect x="{start_x}" y="{y}" width="{bar_max * normalized:.1f}" height="26" rx="13" fill="{color}" />'
            f'<text x="{start_x + bar_max + 18}" y="{y + 20}" font-size="21" fill="#4f5665" font-family="Helvetica, Arial, sans-serif">{value:.2f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Classifier metric overview">
  <rect width="{width}" height="{height}" fill="#fffdf8" />
  <text x="36" y="36" font-size="28" fill="#444b57" font-family="Helvetica, Arial, sans-serif">Classifier KPI Overview</text>
  <text x="36" y="392" font-size="18" fill="#7b808a" font-family="Helvetica, Arial, sans-serif">{html.escape(summary.get("experiment", "external_classifier"))}</text>
  {''.join(bars)}
</svg>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    return output_path


def build_classifier_improvement_summary(summary: Dict) -> Dict[str, object]:
    n_questions = max(int(summary.get("n_questions", 0)), 1)
    n_partial = int(summary.get("answer_metrics", {}).get("n_partially_correct", 0))
    n_incorrect = int(summary.get("n_incorrect", 0))
    ranking = summary.get("classifier", {}).get("ranking_metrics", {})
    calibration = summary.get("classifier", {}).get("calibration", {})
    hit_at_k = float(ranking.get("hit_at_k") or 0.0)
    top1 = float(ranking.get("exact_top1_accuracy") or 0.0)
    mrr = float(summary.get("retrieval_metrics", {}).get("mean_mrr_at_k") or 0.0)
    margin = float(summary.get("classifier", {}).get("avg_top1_top2_margin") or 0.0)
    explanation_coverage = float(summary.get("classifier", {}).get("explanation_coverage") or 0.0)
    dominant_reason = summary.get("diagnostics", {}).get("most_common_reason")
    hierarchy_similarity = float(
        ranking.get("mean_hierarchy_score_top1")
        or ranking.get("mean_cpv_hierarchy_similarity_top1")
        or 0.0
    )
    ece = calibration.get("expected_calibration_error")
    noise = float(summary.get("answer_metrics", {}).get("mean_noise_sensitivity_relevant") or 0.0)

    improvements: List[str] = []
    blockers: List[str] = []

    if n_partial / n_questions >= 0.15 and hit_at_k > top1:
        blockers.append("the right label is often already present in top-k, but the classifier still chooses the wrong top-1 label")
        improvements.append(
            "Improve the reranking or final decision step so the classifier selects the correct candidate more consistently."
        )
    if n_incorrect / n_questions >= 0.2:
        blockers.append("the correct label is too often missing from the candidate list")
        improvements.append(
            "Improve candidate coverage: enrich the label representations with more real examples, synonyms, or structured relations."
        )
    if hierarchy_similarity > top1 + 0.15:
        blockers.append("many misses stay inside a related hierarchy branch, but the classifier still fails at the final selection")
        improvements.append(
            "Add hierarchy-aware reranking so related candidates inside the same branch are separated more precisely."
        )
    if hierarchy_similarity >= 0.75 and top1 < 0.65:
        improvements.append(
            "Use hierarchy as a scoring feature or fallback because the classifier is often near the right branch already."
        )
    if margin < 0.05:
        blockers.append("the score separation between the first and second candidate is weak")
        improvements.append(
            "Calibrate scoring or add a stronger reranker, because the classifier does not separate close candidates clearly enough."
        )
    if ece is not None and float(ece) > 0.15:
        blockers.append("confidence is poorly calibrated")
        improvements.append(
            "Introduce a confidence threshold or calibration layer before using the score for automated acceptance."
        )
    if noise >= 0.25:
        improvements.append(
            "Reduce noise sensitivity by tightening evidence selection and removing broad examples that support the wrong label."
        )
    if explanation_coverage < 0.8:
        improvements.append(
            "Return explanations more consistently so error analysis can distinguish ranking errors from reasoning errors."
        )
    if dominant_reason in {"gold_missing_from_top_k", "expected_answer_missing_from_top_k"}:
        improvements.append(
            "Focus first on retrieval and candidate generation, not only on explanation prompting."
        )
    if dominant_reason in {"gold_present_but_not_ranked_first", "expected_answer_present_but_not_ranked_first"}:
        improvements.append(
            "The most promising improvement is better top-1 selection from already relevant candidates."
        )
    if top1 >= 0.75 and hit_at_k >= 0.9 and mrr >= 0.8:
        headline = "This classifier already looks strong."
        if not improvements:
            improvements.append(
                "Use this setup as a baseline and focus on edge cases where closely related labels are confused."
            )
    else:
        headline = "This classifier still has clear improvement potential."
        if not blockers:
            blockers.append("performance is uneven across queries, even if no single failure mode dominates")
        if not improvements:
            improvements.append(
                "Compare the classifier against an enriched label set or a stronger reranking stage to isolate the bottleneck."
            )

    return {
        "headline": headline,
        "successful": top1 >= 0.75 and hit_at_k >= 0.9,
        "main_blockers": list(dict.fromkeys(blockers)),
        "improvements": list(dict.fromkeys(improvements)),
        "snapshot": {
            "top1_accuracy": top1,
            "hit_at_k": hit_at_k,
            "mrr_at_k": mrr,
            "mean_hierarchy_score_top1": hierarchy_similarity,
            "mean_cpv_hierarchy_similarity_top1": hierarchy_similarity,
            "calibration_ece": ece,
            "most_common_reason": dominant_reason,
        },
    }


def write_classifier_showcase_md(
    summary: Dict,
    output_path: str,
    *,
    recommendation: Dict[str, object],
) -> str:
    answer_metrics = summary.get("answer_metrics", {})
    diagnostics = summary.get("diagnostics", {})
    counts = diagnostics.get("counts_by_primary_reason", {}) or {}
    sorted_reasons = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    lines = [
        f"# Classifier Showcase: {summary['experiment']}",
        "",
        f'"n_correct": {int(summary.get("n_correct", 0))},',
        f'"n_partially_correct": {int(answer_metrics.get("n_partially_correct", 0))},',
        f'"n_incorrect": {int(answer_metrics.get("n_incorrect", 0))}',
        "",
        "## Reason Analysis",
        "",
    ]

    if sorted_reasons:
        lines.append(
            f"The dominant failure mode is `{recommendation['snapshot']['most_common_reason']}`."
        )
        lines.append("")
        for reason, count in sorted_reasons:
            lines.append(f'- "{reason}": {count}')
        if recommendation["main_blockers"]:
            lines.append("")
            lines.append("Main blockers:")
            for item in recommendation["main_blockers"]:
                lines.append(f"- {item}")
        if recommendation["improvements"]:
            lines.append("")
            lines.append("What should be improved:")
            for item in recommendation["improvements"]:
                lines.append(f"- {item}")
    else:
        lines.append("No diagnostic reasons were recorded.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return output_path


def write_classifier_showcase_bundle(
    *,
    summary: Dict,
    ranking_rows: Sequence[Dict],
    experiment_dir: str,
) -> Dict[str, object]:
    score_profile_svg = write_strategy_score_profile_svg(
        ranking_rows,
        os.path.join(experiment_dir, "classifier_score_profile.svg"),
        experiment_slug=summary["experiment"],
        top_k=int(summary["top_k"]),
    )
    metric_overview_svg = write_classifier_metric_overview_svg(
        summary, os.path.join(experiment_dir, "classifier_metric_overview.svg")
    )
    chunk_alignment_svg = write_chunk_relevance_comparison_svg(
        ranking_rows,
        os.path.join(experiment_dir, "classifier_chunk_alignment.svg"),
        top_k=int(summary["top_k"]),
        chart_label=summary.get("experiment", "classifier"),
    )
    unique_chunk_alignment_svg = write_chunk_relevance_comparison_svg(
        ranking_rows,
        os.path.join(experiment_dir, "classifier_unique_chunk_alignment.svg"),
        top_k=int(summary["top_k"]),
        chart_label=summary.get("experiment", "classifier"),
        unique_relevance=True,
    )
    diagnostics_svg = write_strategy_diagnostics_svg(
        summary, os.path.join(experiment_dir, "classifier_diagnostics.svg")
    )
    recommendation = build_classifier_improvement_summary(summary)
    showcase_md = write_classifier_showcase_md(
        summary,
        os.path.join(experiment_dir, "classifier_showcase.md"),
        recommendation=recommendation,
    )
    return {
        "enabled": True,
        "score_profile_svg": score_profile_svg,
        "chunk_alignment_svg": chunk_alignment_svg,
        "unique_chunk_alignment_svg": unique_chunk_alignment_svg,
        "metric_overview_svg": metric_overview_svg,
        "diagnostics_svg": diagnostics_svg,
        "showcase_md": showcase_md,
        "improvement_summary": recommendation,
    }
