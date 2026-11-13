# Rubric-coverage scoring (ResearchQA)

The code that produced `data/coverage_scores/`, copied from the ResearchQA repo (`compute_coverage.py`,
`researchqa_loader.py`, `scripts/submit_coverage_batch.py`, `scripts/fetch_coverage_batch_results.py`).

For every rubric item of a question, an LLM judge rates how well the answer covers it: Not at all /
Barely / Moderately / Mostly / Completely. Items are sent 8 per request (failed requests are retried up
to 3 times, then the question is skipped); the ratings map to 0-1 and are averaged into `coverage_score`.
Every variant of a question is scored against the original question's rubric.

## Setup

```bash
pip install openai google-genai python-dotenv tqdm retry
export GOOGLE_API_KEY=...    # Gemini judge
export OPENAI_API_KEY=...    # only for an OpenAI judge model
```

The rubrics are in `data/query/researchqa_rubrics.json` (the 500 questions of this dataset, in
ResearchQA's `test.json` format). `make_rubric_file.py --test_json <ResearchQA test.json>` rebuilds it.

## Real-time scoring

```bash
python scoring/compute_coverage.py \
    --data data/query/researchqa_rubrics.json \
    --responses data/answers/dr-tulu/without_term.jsonl \
    --output scores_without_term_dr-tulu.json \
    --model <gemini model id>        # any "gemini-*" id uses Gemini; anything else uses OpenAI
```

`--responses` takes this repo's `data/answers/{model}/{variant}.jsonl` files or a JSON map
`{id: {"answer": ...}}`. The run saves after every question and skips already-scored ids when rerun, so
an interrupted run (e.g. a daily quota) can be resumed with the same command.

## Gemini Batch API scoring (cheaper, asynchronous)

```bash
python scoring/submit_coverage_batch.py --data data/query/researchqa_rubrics.json \
    --responses data/answers/gpt-5.4/query.jsonl --output scores_query_gpt-5.4.json --model <gemini model id>
python scoring/fetch_coverage_batch_results.py --meta scores_query_gpt-5.4.json.batch_meta.json --wait
```

The batch path builds byte-identical prompts (it reuses `build_prompt`) and writes the same output format.

## Output

`{id: {"coverage_score": float, "rubric_judges": [{"rubric": ..., "score": ...}, ...]}}` - the source of
`data/coverage_scores/{model}/{variant}.jsonl`.

## Provenance

- The GPT-5.4 scores were produced with the Batch API scripts (the source files end in `_batch`), the DR
  Tulu and Tongyi scores with `compute_coverage.py`, all with Gemini 3.1 Pro Preview as the judge. The
  exact model id string passed to `--model` was not recorded in the outputs.
- Checked in this repo: `compute_coverage_score()` reproduces every stored `coverage_score` from its
  `rubric_judges` (4260 of 4260), and each stored `rubric_judges` list matches the question's rubric in
  `researchqa_rubrics.json` item for item.
- Answers that only asked a clarifying question were removed before scoring, so they have no score.
