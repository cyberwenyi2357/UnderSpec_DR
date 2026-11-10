# ResearchQA ambiguity dataset

ResearchQA queries, their underspecified variants, the answers of three deep research agents to each
version, an LLM judge's verdict on whether each answer recognized the ambiguity, and each answer's
rubric-coverage score from Gemini 3.1 Pro Preview.

- 500 ResearchQA questions (sample of the test split), with domain metadata.
- 4 query variants per question where applicable: `fully_underspecified`, `without_scope`,
  `without_term`, `without_preference`.
- 3 agents: GPT-5.4 with tools (`gpt-5.4`), DR Tulu (`dr-tulu`), Tongyi DeepResearch (`tongyi`).
- Ambiguity judge (gpt-5.4-mini): `stated_assumption_and_proceeded`, `asked_clarifying_question` or
  `no_recognition`, with a short evidence quote.
- Coverage judge (Gemini 3.1 Pro Preview): how well the answer covers each ResearchQA rubric item
  (Not at all ... Completely), averaged into a 0-1 `coverage_score`.

## Files

### `data/queries.jsonl` - one row per question

```json
{"id": "...", "general_domain": "...", "subdomain": "...", "field": "...",
 "query": "<original query>", "fully_underspecified": "...", "without_scope": "...",
 "without_term": "...", "without_preference": "..."}
```

A variant is `null` when the question has no such variant (not every question has a scope, a term or
a preference that can be removed).

| Variant | Questions |
|---|---|
| query (original) | 500 |
| fully_underspecified | 500 |
| without_scope | 274 |
| without_term | 238 |
| without_preference | 27 |

### `data/answers/{model}/{variant}.jsonl` - `{"id", "answer"}`

### `data/ambiguity_judge/{model}/{variant}.jsonl` - `{"id", "category", "evidence", "error"}`

### `data/coverage_scores/{model}/{variant}.jsonl` - `{"id", "coverage_score", "rubric_judges"}`

`rubric_judges` is the list of `{"rubric", "score"}` verdicts behind `coverage_score`.

### `data/combined.jsonl` - one row per model x variant x answer (long format)

```json
{"id": "...", "model": "dr-tulu", "variant": "without_term", "query": "<variant text>",
 "original_query": "...", "answer": "...", "ambiguity_category": "...", "ambiguity_evidence": "...",
 "coverage_score": 0.75}
```

`coverage_score` is `null` for answers without a Gemini score (see Coverage).

### `analysis/results_using_Gemini3.1ProPreview_as_judge/`

Outputs of the analyses run on the Gemini scores, copied as is: query-vs-variant paired statistics
and Wilcoxon tests (`query_vs_variant_diff*.json`, `wilcoxon_*.json`), per-variant averages, figures,
the list of removed clarifying-question answers, and extracted example cases. Some files also cover
ScholarQA-Bench and DeepResearch Bench.

## Coverage

Answers per model and variant (all have an ambiguity-judge result unless noted):

| Model | query | fully_underspecified | without_scope | without_term | without_preference |
|---|---|---|---|---|---|
| gpt-5.4 | 455 | 382 | 233 | 189 | 24 (23 judged) |
| dr-tulu | 500 | 500 | 274 | 238 | 28 (27 judged) |
| tongyi | 500 | 500 | 274 | 238 | 27 |

Gemini coverage scores per model and variant:

| Model | query | fully_underspecified | without_scope | without_term | without_preference |
|---|---|---|---|---|---|
| gpt-5.4 | 454 | 313 | 229 | 171 | 24 |
| dr-tulu | 500 | 499 | 274 | 237 | 27 |
| tongyi | 500 | 494 | 274 | 237 | 27 |

Notes:

- GPT-5.4 has fewer answers than the other agents (missing runs), so compare models on shared ids.
- Answers that only asked a clarifying question were not scored (listed in
  `analysis/.../removed_clarifying_question_researchQA.json`), which is most of the gap between
  answers and scores, especially for GPT-5.4 `fully_underspecified`.
- `174539437127211056-s15` has a `without_preference` answer from GPT-5.4 and DR Tulu although the
  question has no `without_preference` variant; it has no judge result and `query: null` in
  `combined.jsonl`.
- Answers are the agents' final reports, with inline `<cite id="...">` tags kept as produced.

## Rebuilding

`build.py` regenerates `data/` and `analysis/` from the ResearchQA working repo:

```bash
python build.py --src /path/to/ResearchQA
```

Sources there: `researchQA_sample{450,50}_*.jsonl` (queries and variants), `test.json` (domain
metadata), `answer_researchQA_sample500_{variant}_{5.4,qwen,tongyi}_extracted.json` (answers) and
`results/ambiguity_judge/{variant}_{5.4,qwen,tongyi}.jsonl` (judge results),
`results_using_Gemini3.1ProPreview_as_judge/{gpt 5.4,qwen,tongyi}/results_{variant}_*.json` (coverage
scores) and the rest of `results_using_Gemini3.1ProPreview_as_judge/` (analysis outputs).
