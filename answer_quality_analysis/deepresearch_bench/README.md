# RACE and FACT scoring (DeepResearch Bench)

The code that produced `data/deepresearch_bench/race_scores/` and `data/deepresearch_bench/fact_scores/`,
copied from the deep_research_bench working repo (a fork of DeepResearch Bench, Apache-2.0, see `LICENSE`).
Run everything from this folder: the scripts import `utils/` and `prompt/` as top-level packages.

## Setup

```bash
pip install -r requirements.txt
```

API keys go in a `.env` in this folder (none are included):

- `GEMINI_API_KEY` - RACE (cleaning and scoring) and FACT claim extraction
- `JINA_API_KEY` - FACT page fetching
- `OPENAI_API_KEY` - FACT validation with gpt-5.4-mini

DeepResearch Bench's task files are bundled, at the paths the code expects: `data/prompt_data/query.jsonl`
(the 100 tasks), `data/criteria_data/criteria.jsonl` (per-task RACE criteria and weights) and
`data/test_data/cleaned_data/reference.jsonl` (the reference reports RACE compares against).

## RACE (report quality)

An LLM judge first cleans the answer (strips formatting artifacts), then compares it with the task's
reference report on task-specific criteria grouped into four dimensions (comprehensiveness, insight,
instruction following, readability). Each score is `target / (target + reference)`, so 0.5 means on par
with the reference. Every variant is scored against the original task's criteria and reference.

```bash
# answers of one model x variant -> data/test_data/raw_data/<name>.jsonl
python prepare_race_input.py --answers ../../data/deepresearch_bench/answers/gpt-5.4/query.jsonl \
    --name gpt5.4_query --exclude_judge ../../data/deepresearch_bench/ambiguity_judge/gpt-5.4/query.jsonl

# Gemini Batch API (how the dataset's scores were produced): cleaning runs synchronously in submit
python -m utils.race_batch submit gpt5.4_query --output_dir results/race/gpt5.4_query --model_name gemini-3.1-pro-preview
python -m utils.race_batch fetch gpt5.4_query --output_dir results/race/gpt5.4_query    # repeat until done

# or fully synchronous (DeepResearch Bench's original entry point)
python deepresearch_bench_race.py gpt5.4_query --eval_model gemini --model_name gemini-3.1-pro-preview \
    --output_dir results/race/gpt5.4_query
```

Output: `results/race/<name>/raw_results.jsonl` (per-answer scores) and `race_result.txt` (averages).
RACE matches answers to criteria and references by prompt text, which `prepare_race_input.py` takes care of.

## FACT (citation accuracy)

Claims with a cited URL are extracted from the answer, the URLs are fetched, and a judge checks each claim
against its page: `supported`, `unsupported` or `unknown` (page unusable). Citation accuracy is
supported / (supported + unsupported). This measures precision only: uncited claims are not penalized.

The answers cite with `<cite id="...">` tags that point into the agent's trajectory, so FACT input is built
from the agent's original output (with `query_full_traces`), not from this dataset's answer files:

```bash
D=fact/gpt5.4_query
python resolve_cite_urls.py --input <agent output>/answer_deepresearch_bench_query_5.4.jsonl --output $D/input.jsonl \
    --exclude_judge ../../data/deepresearch_bench/ambiguity_judge/gpt-5.4/query.jsonl
python -m utils.extract     --raw_data_path $D/input.jsonl --output_path $D/extracted.jsonl \
    --query_data_path data/prompt_data/query.jsonl --n_total_process 10
python -m utils.deduplicate --raw_data_path $D/extracted.jsonl --output_path $D/deduplicated.jsonl \
    --query_data_path data/prompt_data/query.jsonl --n_total_process 10
python -m utils.scrape      --raw_data_path $D/deduplicated.jsonl --output_path $D/scraped.jsonl --n_total_process 10

# validation with gpt-5.4-mini on the OpenAI Batch API, any number of directories in one job
python fact_validate_openai_batch.py submit fact/gpt5.4_query fact/dr-tulu_query ...
python fact_validate_openai_batch.py fetch    # repeat until done; also writes fact_result_gpt.txt
```

Output per directory: `validated_gpt.jsonl` (per-claim verdicts) and `fact_result_gpt.txt`
(`total_citations` and `total_valid_citations` averaged per answer, and `valid_rate` = citation accuracy).
The synchronous Gemini path, `python -m utils.validate ...` then `python -m utils.stat ...`, writes
`validated.jsonl` / `fact_result.txt` instead.

Every stage resumes by answer id, so an interrupted stage can be rerun with the same command. Don't use
`fact_validate_openai_batch.py submit --limit` on a real directory: fetch writes every answer of the
directory, so the citations that were not submitted are recorded as errors and the answers count as done.

## How the dataset's scores were produced

- RACE: Gemini 3.1 Pro Preview for both cleaning and scoring, through `utils.race_batch` (temperature 0).
  Answers that only asked a clarifying question were scored and then removed; `--exclude_judge` drops them
  up front instead, which leaves the other answers' scores unchanged. A few answers ended with an `error`
  and have no score.
- FACT, GPT-5.4 and DR Tulu only:
  - Citations were resolved with the code in `resolve_cite_urls.py`. Rerunning it on the agents' outputs
    reproduces the files that were used byte for byte. 98-99% of DR Tulu's tags and 76-93% of GPT-5.4's
    resolve; the rest are short `S12`/`W3`-style ids with no document behind them.
  - Claims were extracted by gemini-3.8-flash (`FACT_Model` in `utils/api.py`) and the pages fetched with
    Jina Reader.
  - Claims were validated by gpt-5.4-mini with `fact_validate_openai_batch.py`. Clarifying-question
    answers were removed before validation.
  - 16 of the 10,560 validation requests failed and one reply used a wrong key. Those claims count as
    errors and as `unknown` respectively.
- Tongyi has no FACT score. Its DeepResearch Bench answers carry no inline citation markers (no `<cite>`
  tags, a markdown link in at most a few answers), so extraction finds nothing to check.
- `utils/api.py` has the model ids that were current at the time: `FACT_Model = "gemini-3.8-flash"` and the
  RACE default `Model = "gemini-pro-latest"`, overridden by `--model_name gemini-3.1-pro-preview`.
  DeepResearch Bench's original Gemini 2.5 preview ids have been retired. Gemini calls use temperature 0
  and no thinking configuration.
- `utils/scrape.py` differs from DeepResearch Bench's: it fetches all URLs of a file through one process
  pool and caches pages on disk (`<output>.url_cache.jsonl`). The original opened a new pool per answer
  and was too slow for these answers.

## Changes from the working repo

- `utils/api.py`: the Jina client is created on first use, so importing `utils` (e.g. for RACE) no longer
  needs `JINA_API_KEY`.
- `fact_validate_openai_batch.py`: the FACT directories are given on the command line instead of a fixed
  list, and the bookkeeping directory is configurable (`--bookkeeping_dir`).
- `prepare_race_input.py` and `resolve_cite_urls.py` are new. `resolve_cite_urls.py` is the cite-tag
  resolution that had been run inline.
- Not included: criteria generation (`criteria.jsonl` is bundled), the Gemini Batch version of
  extract/validate (only used for an intermediate Gemini validation), and the ambiguity judge (see
  `../ambiguity_awareness_judge.py`).
