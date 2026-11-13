# Deep research agents on underspecified queries

How do deep research agents behave when a research question is underspecified? This repository holds
queries from three benchmarks, rewritten into underspecified variants. It also has the answers of three
deep research agents to every version, quality scores for those answers, and an LLM judge's verdict on
whether each answer noticed the ambiguity. It includes the code to regenerate the answers, scores and
search-behaviour metrics.

| Benchmark | Questions | Answer-quality score |
|---|---|---|
| [ResearchQA](https://github.com/realliyifei/ResearchQA) (sample of the test split) | 500 | Rubric coverage, judged by Gemini 3.1 Pro Preview |
| [DeepResearch Bench](https://github.com/Ayanami0730/deep_research_bench) | 100 (50 zh, 50 en) | RACE (Gemini 3.1 Pro Preview) and FACT citation accuracy |
| [ScholarQA-Bench](https://github.com/AkariAsai/ScholarQABench) (OpenScholar) | 292 (`cs` 100, `cs_2` 84, `multi` 108) | Rubric (`cs`), ingredient recall (`cs_2`), Prometheus (`multi`) |

| Agent | Model id in files | Inference code |
|---|---|---|
| DR Tulu ([`rl-research/DR-Tulu-8B`](https://huggingface.co/rl-research/DR-Tulu-8B)) | `dr-tulu` | `drtulu_inference/` |
| GPT-5.4 with the same tools as DR Tulu | `gpt-5.4` | `drtulu_inference/` (`auto_search_sft-gpt5.4.yaml`) |
| Tongyi DeepResearch (`Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`) | `tongyi` | `tongyi_inference/` |

## Query variants

Each original query (`query`) is rewritten by gpt-5.4-mini (`query/query_rewrite.py`, script default)
into up to four underspecified variants:

| Variant | What is removed |
|---|---|
| `fully_underspecified` | All of the below; named entities become generic descriptions |
| `without_scope` | Explicit scope limitations (scenario, domain, time period, location) |
| `without_term` | Proper nouns (named systems, algorithms, frameworks, organizations), simplified to a generic mention or abbreviation |
| `without_preference` | User preference information (format, depth, audience, evaluation criteria) |

A variant only exists when the original query has something to remove:

| Benchmark | query | fully_underspecified | without_scope | without_term | without_preference |
|---|---|---|---|---|---|
| ResearchQA | 500 | 499 | 274 | 238 | 27 |
| DeepResearch Bench | 100 | 100 | 74 | 45 | 58 |
| ScholarQA-Bench | 292 | 287 | 129 | 126 | 65 |

## Repository layout

```
query/                      queries and their variants, one file per benchmark
  query_rewrite.py          GPT prompt / OpenAI Batch script that produced the variants
rated_answer/               answers + scores + ambiguity judge, one file per benchmark x agent (9 files)
drtulu_inference/           DR Tulu agent framework: runs DR-Tulu-8B or GPT-5.4
tongyi_inference/           Tongyi DeepResearch ReAct inference
answer_quality_analysis/    answer scorers
  researchqa/               rubric coverage
  deepresearch_bench/       RACE and FACT
  scholarQABench/           rubric, Prometheus and citation scorers
  ambiguity_awareness_judge.py  did the answer recognize the query's ambiguity (gpt-5.4-mini)
trajectory_analysis/        search-behaviour metrics on tool-call trajectories
```

## Data

### `query/{benchmark}.jsonl`

One row per question (`researchqa`, `deepresearch_bench`, `scholarqa_bench`). A variant field is `null`
when the question has no such variant.

```json
{"id": "...", "query": "<original query>", "fully_underspecified": "...", "without_scope": "...",
 "without_term": "...", "without_preference": "...", "<metadata>": "..."}
```

The metadata fields differ by benchmark:

| Benchmark | Metadata fields | Notes |
|---|---|---|
| ResearchQA | `general_domain`, `subdomain`, `field` | |
| DeepResearch Bench | `topic`, `language` | For ids 22 and 41, `query` is one character different from the benchmark's `query.jsonl`, which has a typo there. |
| ScholarQA-Bench | `subset` | `(subset, id)` identifies a question. In `multi`, `id` is the OpenScholar id. |

### `rated_answer/{benchmark}_{model}.json`

Everything for one agent on one benchmark (19-29 MB for the largest files):

```json
{"benchmark": "researchqa", "model": "gpt-5.4", "description": "...",
 "score_fields": {"<field>": "<what it means>"}, "ambiguity_judge": "<judge description>",
 "n_questions": 500,
 "questions": [
   {"id": "...", "<metadata>": "...", "original_query": "...",
    "variants": {
      "query":                {"text": "...", "answer": "...", "scores": {...}, "ambiguity_judge": {...}},
      "fully_underspecified": {...}, "without_scope": {...}, "without_term": {...}, "without_preference": {...}}}]}
```

- `text` is the query the agent was given. `answer` is the agent's final report, with inline
  `<cite id="...">` tags kept as produced.
- `ambiguity_judge` is `{"category", "evidence", "error"}`. `category` is one of
  `asked_clarifying_question`, `stated_assumption_and_proceeded` or `no_recognition`, judged by
  gpt-5.4-mini.
- A variant is listed only when the question has it. Inside a variant, `answer`, `scores` or
  `ambiguity_judge` is `null` when it is missing.

What `scores` contains depends on the benchmark:

| Benchmark | `scores` | Meaning |
|---|---|---|
| ResearchQA | `{coverage_score, rubric_judges}` | Gemini 3.1 Pro Preview rates each rubric item from Not at all to Completely. The ratings are averaged into a 0-1 score, and `rubric_judges` keeps the per-item ratings. |
| DeepResearch Bench | `{race, fact}` | `race`: comprehensiveness, insight, instruction following, readability and overall, relative to the benchmark's reference report (0.5 = on par). `fact`: `n_citations`, `n_supported`, `citation_accuracy` and every checked claim. FACT exists for `gpt-5.4` and `dr-tulu` only. |
| ScholarQA-Bench | `{metric, final_score, details}` | `metric` is `rubric` (`cs`), `ingredient_recall` (`cs_2`) or `prometheus` (`multi`). `final_score` is 0-1. Each ScholarQA variant also has `source_id` (row id in the source file) and `answer_source`. |

Coverage caveats:

- **Clarifying-question-only answers have no score.** Answers that only asked a clarifying question
  were not scored. This is most of the gap between answers and scores, especially for GPT-5.4
  `fully_underspecified`.
- **ScholarQA-Bench `cs_2` has scores but no answer text,** and the ambiguity judge was run only on
  the `cs` subset.
- **ScholarQA-Bench `multi` ids:** GPT-5.4's positional ids were mapped to OpenScholar ids by
  original query.

## Reproducing

Each step has its own README with setup and API keys. No keys are included; copy the `.env.example`
files and fill them in.

1. **Query variants**: `query/query_rewrite.py` sends each query to GPT through the OpenAI Batch API.
   One round asks whether the query has a scope, term or preference; a second round rewrites it. The
   script imports `dr_agent`, so run it with `drtulu_inference` on the path:
   ```bash
   PYTHONPATH=drtulu_inference python query/query_rewrite.py --input <queries.jsonl> --input-field query \
       --output <variants.jsonl> --all-lines
   ```
2. **Agent answers and trajectories**
   - DR Tulu and GPT-5.4: see [drtulu_inference/README.md](drtulu_inference/README.md).
     `scripts/eval_query_template_batch.py --query-file ../query/<benchmark>.jsonl` runs every variant.
   - Tongyi: see [tongyi_inference/README.md](tongyi_inference/README.md). `make_eval_data.py` builds
     one dataset per variant from `query/`, and the `queue_runs*.sh` scripts run them.
3. **Answer quality**: `answer_quality_analysis/{researchqa,deepresearch_bench,scholarQABench}/`. The
   evaluation has two dimensions:
   - **Rubric satisfaction**: how far the answer satisfies the query-specific rubric. Each variant is
     scored against the original question's rubric, criteria or reference, so an answer to an
     underspecified variant is still judged on what the original user wanted.
   - **Citation support**: whether the answer's claims are backed by the sources it cites.

   | Benchmark | Rubric satisfaction | Citation support |
   |---|---|---|
   | ResearchQA | Rubric coverage (`researchqa/`). Gemini 3.1 Pro Preview rates every rubric item, giving `coverage_score`. | No separate check. The rubric includes "does the response cite paper X" items, which count toward `coverage_score`. |
   | DeepResearch Bench | RACE (`deepresearch_bench/`): comprehensiveness, insight, instruction following and readability, relative to the benchmark's reference report. | FACT (`deepresearch_bench/`): claims with a cited URL are extracted, the pages are fetched, and gpt-5.4-mini judges whether each page supports its claim. `citation_accuracy` = supported / (supported + unsupported). Run for `gpt-5.4` and `dr-tulu` only. |
   | ScholarQA-Bench | `scholarQABench/`: rubric score for `cs`, ingredient recall for `cs_2`, Prometheus (organization, coverage, relevance) for `multi`. | `scholarQABench/scripts/citation_correctness_eval.py`: citation precision and recall with an NLI attribution model. Its results are not in `rated_answer/`. |
4. **Ambiguity awareness**: `answer_quality_analysis/ambiguity_awareness_judge.py` asks gpt-5.4-mini
   whether each answer in `rated_answer/` recognized its query's ambiguity. It runs on the OpenAI Batch
   API, and `fetch` writes `results/ambiguity_judge/{file}.jsonl`, one
   `{"id", "subset", "variant", "category", "evidence", "error"}` row per answer:
   ```bash
   python answer_quality_analysis/ambiguity_awareness_judge.py submit rated_answer/*.json  # --limit N to test, --force to redo
   python answer_quality_analysis/ambiguity_awareness_judge.py fetch                       # once the batch is done
   ```
5. **Search behaviour**: `trajectory_analysis/` computes the following on the agents' tool-call logs:
   - search query drift
   - transition category (Specialization / Generalization / Exploration / Repetition)
   - query provenance
   - document novelty
   - context-driven term adoption rate

   See [trajectory_analysis/README.md](trajectory_analysis/README.md) for the input schemas.

The scorer READMEs under `answer_quality_analysis/` were written for an earlier layout of this
repository. Their example commands point to `data/...` paths: answers are now inside
`rated_answer/`, and queries are in `query/`.
