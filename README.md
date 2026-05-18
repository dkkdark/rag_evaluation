# RAG Evaluation Admin

This project evaluates RAG and classifier outputs from a small local web UI. It can score document-question answering over PDF regulations, local TED/CPV classification, external classifier APIs, and already prepared RAG/classifier result files.

## Quick Start

1. Install dependencies:

   ```bash
   python -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. Put input files in the expected folders:

   - PDF regulations: `data/files/**/*.pdf*`
   - Evaluation questions: `data/questions_by_file.json`
   - CPV catalog: `data/cpv_ted_train_catalog.csv`
   - CPV test queries: `data/cpv_ted_test_queries.json`
   - Prepared result workbook: `data/eval_dataset.xlsx`

3. Start the web UI:

   ```bash
   .venv/bin/python -m rag_eval.entrypoints.web_admin
   ```

4. Open the printed URL, usually:

   ```text
   http://127.0.0.1:8000
   ```

5. Choose `Mode`, choose a `Classifier`, fill the visible settings, then click `Start evaluation`.

Runs are written under `outputs/<run_name>/`. The most important files are `run_summary.json`, `rag_results.csv`, `retrieved_chunks.csv`, `answer_metrics.csv`, `retrieval_metrics.csv`, `diagnostics.csv`, and `quality_report.md`.

## Code Layout

- `rag_eval/apps/`: CLI and web admin application code.
- `rag_eval/entrypoints/`: executable module entrypoints used with `python -m`.
- `rag_eval/core/`: shared dataclasses and text normalization helpers.
- `rag_eval/data/`: PDF parsing and TED/CPV dataset preparation.
- `rag_eval/retrieval/`: chunking, retrievers, KG retrieval, and experiment orchestration.
- `rag_eval/evaluation/`: metrics, LLM judging, LLM answer calls, and quality advice.
- `rag_eval/classifiers/`: TED/CPV, API classifier, and prepared-results evaluation.
- `rag_eval/reporting/`: SVG and markdown report generation.

## Mode vs Classifier

`Mode` controls whether the system runs one configuration or compares many configurations.

- `classifier`: run one concrete classifier/evaluation target.
- `sweep`: compare multiple chunking and retrieval settings for PDF-based RAG. Use `auto` values for chunking/retriever when you want a broader comparison.

`Classifier` controls what is being evaluated.

- `examination_regulations`: PDF-based RAG over examination/regulation documents from `data/files/`.
- `ted_cpv`: local CPV classifier that ranks CPV catalog entries for TED-style queries.
- `api_classifier`: calls an external HTTP classifier and evaluates its ranked CPV predictions.
- `prepared_rag_results`: evaluates predictions already stored in an Excel workbook.

## Running Each Classifier in the Web UI

### `examination_regulations`

Use this when you want to evaluate question answering over PDF regulation documents.

Required inputs:

- `PDF files`: select one or more files detected under `data/files/`.
- `Questions`: JSON file with questions, gold answers, optional expected keywords, and optional metadata filters.
- `Retriever`: `tfidf`, `bm25`, `dense`, `hybrid`, or `auto` in sweep mode.
- `Chunking`: `fixed_words`, `fixed_tokens`, `by_section`, `by_paragraph`, or `auto` in sweep mode.

Useful options:

- `Top K`: how many chunks are retrieved per question.
- `LLM answers`: generate an answer from retrieved chunks. If disabled, the system uses an extractive fallback.
- `LLM judge`: adds claim-level support/contradiction metrics.
- `Abstain weak`: refuses to answer when runtime retrieval signals say evidence is weak.
- `KG retrieval`: adds graph-augmented retrieval and KG output metrics.

Main outputs:

- `raw_documents.txt`: extracted PDF text.
- `sections.csv` and `paragraphs.csv`: parsed document structure.
- `retrieved_chunks.csv`: chunks retrieved for each question.
- `rag_results.csv`: one row per question with answer, retrieval, diagnostics, and recommendations.

### `ted_cpv`

Use this when you want to evaluate the built-in local CPV classifier. It treats CPV catalog entries as retrievable candidates and checks whether the gold CPV code appears near the top.

Required inputs:

- `CPV catalog`: usually `data/cpv_ted_train_catalog.csv`.
- `CPV queries`: usually `data/cpv_ted_test_queries.json`.
- `Retriever`: retrieval method used to rank CPV labels/descriptions.

Useful options:

- `Use examples`: appends real TED examples to CPV catalog text before ranking.
- `Top K`: how many CPV candidates are evaluated.
- `Rerank top N` and `Rerank weight`: optional lexical reranking over the first candidates.

Main outputs:

- `retrieved_chunks.csv`: ranked CPV candidates per query.
- `rag_results.csv`: query-level correctness and CPV-specific ranking metrics.

### `api_classifier`

Use this when an external service produces ranked CPV predictions and you want this project to evaluate them.

Required inputs:

- `API classifier URL`: endpoint that accepts JSON requests.
- `API token env`: environment variable containing a bearer token, if needed.
- `CPV catalog`: catalog used to attach labels and compute hierarchy metrics.
- `CPV queries`: test queries sent to the API.

Request sent to the API:

```json
{
  "id": "query-id",
  "query": "query text",
  "top_k": 5
}
```

Expected response shape:

```json
{
  "id": "query-id",
  "query": "query text",
  "predictions": [
    {"label": "12345678", "score": 0.91}
  ],
  "explanation": "optional"
}
```

Prediction objects may use `label`, `cpv_code`, `answer`, `id`, or `code` for the predicted code. Scores may use `score`, `confidence`, or `probability`.

Main outputs:

- `retrieved_chunks.csv`: normalized API predictions as ranked candidates.
- `rag_results.csv`: correctness, hierarchy, confidence, and diagnostic fields.

### `prepared_rag_results`

Use this when you already have RAG or classifier predictions in an Excel workbook and only want to score them.

Required inputs:

- `Prepared results`: usually `data/eval_dataset.xlsx`.
- `CPV catalog`: catalog used for labels and hierarchy-aware scoring.

Supported workbook columns include:

- Current/simple layout: `ID`, `Query (BANF)`, `Expected CPV`, `Predicted CPV`, `RRF Score`.
- Top-K layout: `Predicted CPV #1`, `Vector Score #1`, `Predicted CPV #2`, `Vector Score #2`, and so on.
- Optional answer/context columns: `LLM Answer`, `Answer`, `Chunks`, `Chunk Text`, `Retrieved Chunks`, `Chunk ID`.

Multiple rows with the same `ID` are treated as ranked candidates for one query.

Main outputs:

- `prepared_source_rows.csv`: normalized source workbook rows.
- `retrieved_chunks.csv`: prepared predictions converted into ranked candidates.
- `rag_results.csv`: query-level scoring and recommendations.

## Metrics

Most metrics are between `0` and `1`, where higher is better, unless the description says otherwise. Empty values usually mean the metric was not applicable for that row.

### Overall Result Counts

- `n_questions`: number of evaluated questions or queries.
- `n_correct`: predictions/answers marked correct.
- `n_partially_correct`: answers with some correct evidence but incomplete coverage.
- `n_incorrect`: answers that miss the expected answer.
- `n_unsupported`: answers that appear unsupported by retrieved context.
- `n_needs_manual_review`: cases where automatic scoring is not confident enough.

### CPV Classification Metrics

- `gold_hit_at_k`: whether the gold CPV code appears anywhere in the top-K predictions.
- `gold_first_rank`: first rank where the gold CPV code appears. Lower is better.
- `reciprocal_rank`: `1 / gold_first_rank`; top-1 hit is `1.0`, rank-2 hit is `0.5`.
- `exact_top1_accuracy`: share of queries where the first predicted CPV code equals the gold code.
- `hit_at_k`: share of queries where the gold code appears anywhere in top-K.
- `mrr_at_k`: average reciprocal rank across queries.
- `cpv_common_prefix_length_top1`: how many leading digits the top prediction shares with the gold CPV code.
- `cpv_hierarchy_distance_top1`: hierarchy distance between top prediction and gold. Lower is better.
- `mean_cpv_hierarchy_similarity_top1`: average closeness in the CPV hierarchy. Higher is better.
- `same_division_top1`, `same_group_top1`, `same_class_top1`, `same_category_top1`, `same_branch_top1`: whether top-1 lands in the same CPV hierarchy level as the gold code.
- `ancestor_hit_at_k`: whether a broader ancestor/related hierarchy match appears in top-K.

### Retrieval Metrics

- `mrr_at_k`: how early the first relevant chunk/candidate appears. `1.0` means rank 1.
- `ndcg_at_k`: ranking quality that rewards stronger relevant items near the top.
- `recall_at_k`: share of all relevant chunks/candidates found in top-K.
- `ragas_recall_at_k`: share of reference facts or keywords covered by retrieved context.
- `first_relevant_rank`: first top-K position considered relevant.
- `n_relevant_chunks`: how many relevant chunks/candidates exist in the available candidate pool.
- `n_retrieved_relevant_chunks`: how many relevant chunks/candidates were retrieved in top-K.
- `target_doc_retrieved_at_k`: whether the expected document/code appears in top-K.
- `first_target_doc_rank`: first rank where the expected document/code appears.
- `n_retrieved_target_doc_chunks`: how many top-K items came from the expected document/code.
- `questions_with_relevant_chunk`: number of questions with at least one relevant retrieved item.
- `questions_with_target_doc_at_k`: number of questions where top-K includes the target document/code.

### Answer Quality Metrics

- `answer_accuracy_label`: automatic label such as `correct`, `partially_correct`, `incorrect`, `unsupported`, or `needs_manual_review`.
- `gold_answer_overlap`: token overlap between generated answer and gold answer.
- `answer_gold_support`: how much new information in the answer is supported by the gold answer.
- `proxy_faithfulness`: lightweight estimate of whether answer content is grounded in retrieved context.
- `proxy_context_relevance`: lightweight estimate of how relevant retrieved context is to the answer.
- `answer_has_gold_substring`: whether the gold answer appears directly in the generated answer.
- `answerability_confidence`: runtime confidence that the question can be answered from retrieved evidence.
- `runtime_retrieval_status`: `good_evidence`, `weak_evidence`, or `missing_evidence`.
- `runtime_retrieval_action`: suggested action, such as answer, retrieve more, or abstain.

### Claim and Grounding Metrics

These are most useful when answers contain sentences or claims. They become stronger when the LLM judge is enabled.

- `gold_claim_count`: number of reference claims extracted from the gold answer.
- `answer_claim_count`: number of claims extracted from the generated answer.
- `context_claim_recall`: share of gold claims found in retrieved context.
- `answer_claim_recall`: share of gold claims covered by the answer.
- `answer_claim_precision`: share of answer claims supported by the gold/reference information.
- `answer_claim_f1`: balance of answer claim precision and recall.
- `factual_correctness_precision`: how much of the answer is factually correct.
- `factual_correctness_recall`: how much expected factual content the answer covers.
- `factual_correctness_f1`: balanced factual correctness score.
- `grounded_claim_ratio`: share of answer claims supported by retrieved context.
- `hallucinated_claim_ratio`: share of answer claims not supported by retrieved context. Lower is better.
- `unsupported_claim_count`: number of answer claims not supported by context.
- `missing_gold_claim_count`: number of expected claims missing from the answer/context.
- `contradicted_claim_count`: number of claims judged contradictory.
- `claim_diagnostic`: compact reason for claim-level failure, such as missing evidence or unsupported generated claims.

### Evidence Attribution Metrics

These describe whether answer claims point back to valid supporting chunks.

- `evidence_attribution_precision`: share of answer attributions that are valid.
- `evidence_attribution_recall`: share of expected/gold claims with valid supporting evidence.
- `evidence_attribution_f1`: balance of attribution precision and recall.
- `evidence_coverage`: share of gold claims that have evidence in retrieved context.
- `attributed_answer_claim_count`: answer claims with valid evidence.
- `attributed_gold_claim_count`: gold claims with valid evidence.
- `invalid_attribution_count`: references to chunks that do not actually support the claim. Lower is better.

### Abstention Metrics

These matter when the dataset marks some questions as unanswerable or when `Abstain weak` is enabled.

- `expected_answerable`: whether the question is expected to be answerable.
- `abstained`: whether the system refused to answer.
- `abstention_correct`: whether abstention was correct for an unanswerable question.
- `n_abstained`: total abstentions.
- `n_correct_abstentions`: abstentions on questions marked unanswerable.
- `abstention_precision`: of all abstentions, how many were correct.
- `abstention_recall`: of all unanswerable questions, how many were correctly abstained.
- `over_answering_rate`: how often the system answered when it should abstain. Lower is better.
- `false_refusal_rate`: how often the system abstained despite answerable evidence. Lower is better.

### Calibration and Confidence

- `prediction_confidence`: confidence score attached to the top prediction.
- `mean_confidence`: average confidence.
- `accuracy`: accuracy for rows that had confidence values.
- `brier_score`: calibration error where lower is better.
- `expected_calibration_error`: gap between confidence and actual correctness. Lower is better.
- `bins`: confidence buckets showing whether predictions are overconfident or underconfident.

### Diagnostics

- `primary_error_reason`: broad failure category.
- `secondary_error_reason`: more specific failure category.
- `diagnostic_explanation`: plain-language explanation for the failure.
- `counts_by_primary_reason`: aggregate count of major failure types.
- `counts_by_claim_diagnostic`: aggregate count of claim-level failure types.
- `most_common_reason`: most frequent diagnostic issue in a run.

## Reading a Run

Start with these files:

1. `run_summary.json`: high-level run metadata, selected classifier, and links to output artifacts.
2. `quality_report.md`: human-readable recommendations.
3. `rag_results.csv`: the main per-question table.
4. `retrieved_chunks.csv`: what the retriever/classifier actually returned.
5. `answer_metrics_summary.json` and `retrieval_metrics_summary.json`: aggregate scores.
6. `diagnostics.csv`: why each row failed or needs review.
