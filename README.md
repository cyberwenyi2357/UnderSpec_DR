# ResearchQA ambiguity dataset

ResearchQA queries, their underspecified variants, the answers of three deep research agents to each
version, and an LLM judge's verdict on whether each answer recognized the ambiguity.

- 500 ResearchQA questions (sample of the test split), with domain metadata.
- 4 query variants per question where applicable: `fully_underspecified`, `without_scope`,
  `without_term`, `without_preference`.
- 3 agents: GPT-5.4 with tools (`gpt-5.4`), DR Tulu (`dr-tulu`), Tongyi DeepResearch (`tongyi`).
- Ambiguity judge (gpt-5.4-mini): `stated_assumption_and_proceeded`, `asked_clarifying_question` or
  `no_recognition`, with a short evidence quote.

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

### `data/combined.jsonl` - one row per model x variant x answer (long format)

```json
{"id": "...", "model": "dr-tulu", "variant": "without_term", "query": "<variant text>",
 "original_query": "...", "answer": "...", "ambiguity_category": "...", "ambiguity_evidence": "..."}
```

## Coverage

Answers per model and variant (all judged unless noted):

| Model | query | fully_underspecified | without_scope | without_term | without_preference |
|---|---|---|---|---|---|
| gpt-5.4 | 455 | 382 | 233 | 189 | 24 (23 judged) |
| dr-tulu | 500 | 500 | 274 | 238 | 28 (27 judged) |
| tongyi | 500 | 500 | 274 | 238 | 27 |

Notes:

- GPT-5.4 has fewer answers than the other agents (missing runs), so compare models on shared ids.
- `174539437127211056-s15` has a `without_preference` answer from GPT-5.4 and DR Tulu although the
  question has no `without_preference` variant; it has no judge result and `query: null` in
  `combined.jsonl`.
- Answers are the agents' final reports, with inline `<cite id="...">` tags kept as produced.

## Rebuilding

`build.py` regenerates `data/` from the ResearchQA working repo:

```bash
python build.py --src /path/to/ResearchQA
```

Sources there: `researchQA_sample{450,50}_*.jsonl` (queries and variants), `test.json` (domain
metadata), `answer_researchQA_sample500_{variant}_{5.4,qwen,tongyi}_extracted.json` (answers) and
`results/ambiguity_judge/{variant}_{5.4,qwen,tongyi}.jsonl` (judge results).
