# Answer scoring code (extracted from ScholarQABench)

Keep this layout: scripts import each other (run_utils, scholarqacs_rubric, gemini_batch).
API keys go in a `.env` at this folder's root (OPENAI_API_KEY, GEMINI_API_KEY, ...); none are included.

## 1. Rubric-based scoring (ScholarQA-CS), LLM judge
- scripts/rubric_eval.py          entry point (--qa-dir, --test-config, --rubrics, --snippets)
- scripts/scholarqacs_rubric.py   rubric scoring logic (RubricCorpusQaGenericMetric)
- scripts/run_utils.py            LLM calls (litellm) and JSON parsing
- scripts/gemini_batch.py         optional Gemini batch backend
  Needs test_configs_snippets.json from data/scholarqa_cs/ in the original repo.

## 2. Prometheus scoring (organization / coverage / relevance; ScholarQA-Multi)
- scripts/prometheus_eval.py      runs the Prometheus judge (vLLM, GPU)
- rubrics/prometheus_rubrics_v8.json
- scripts/aggregate_scores.py     averages the three dimensions into a final score
- scripts/score_diff_query_vs_fully_underspecified.py  per-question score difference between variants

## 3. Citation accuracy / string-match correctness
- scripts/citation_correctness_eval.py  citation precision/recall (NLI model), accuracy, ROUGE-L
- scripts/convert_cite_tags_for_citation_eval.py  converts answer citation tags into the eval format
- scripts/aggregate_citation_eval.py   aggregates citation results

## 4. Ambiguity awareness judge
- ../ambiguity_awareness_judge.py      OpenAI Batch API judge (submit / fetch), shared by all three benchmarks
