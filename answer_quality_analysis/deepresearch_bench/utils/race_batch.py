"""
Submit and fetch RACE scoring jobs through the Gemini Batch API.

Batch mode is asynchronous (results can take minutes to ~24h, and jobs expire
after 48h), so this is split into two independent commands instead of one
blocking call:

    python -m utils.race_batch submit <target_model> --output_dir results/race/<model> \\
        --model_name gemini-3.1-pro-preview [--skip_cleaning] [--only_zh] [--only_en] [--limit N]

    python -m utils.race_batch fetch <target_model> --output_dir results/race/<model>

`submit` cleans the articles (same as deepresearch_bench_race.py), builds the
same per-task scoring prompt used in synchronous mode, uploads them as a
single JSONL batch request, and saves the batch job name + the parameters
needed to rebuild the task list to `<output_dir>/batch_job.json`.

`fetch` reads that file, checks the job status, and if it has finished,
downloads the results, recomputes the weighted RACE scores the same way
deepresearch_bench_race.py does, and writes `raw_results.jsonl` /
`race_result.txt` to `<output_dir>`. If the job hasn't finished yet, it just
prints the current state and exits without writing anything.
"""

import argparse
import json
import logging
import os
import time

from utils.api import Model as DEFAULT_GEMINI_MODEL
from utils.clean_article import ArticleCleaner
from utils.api import AIClient
from utils.io_utils import load_jsonl
from utils.json_extractor import extract_json_from_markdown
from utils.score_calculator import calculate_weighted_scores
from deepresearch_bench_race import format_criteria_list, CRITERIA_FILE, REFERENCE_FILE, MAX_RETRIES
from prompt.score_prompt_zh import generate_merged_score_prompt as zh_merged_score_prompt
from prompt.score_prompt_en import generate_merged_score_prompt as en_merged_score_prompt

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXPECTED_DIMS = ["comprehensiveness", "insight", "instruction_following", "readability"]

SUCCESS_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
TERMINAL_STATES = SUCCESS_STATES | {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def _get_genai_client():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    from google import genai
    return genai.Client(api_key=api_key, http_options={'timeout': 600000})


def load_tasks(target_model, cleaned_data_dir, query_file, only_zh, only_en, limit):
    """Rebuild the same (tasks_to_process, criteria_map, target_map, reference_map)
    that deepresearch_bench_race.process_language_data loads, but for both
    languages combined in one pass (batch submission bundles everything into
    a single job)."""
    all_tasks = load_jsonl(query_file)
    if only_zh:
        all_tasks = [t for t in all_tasks if t.get('language') == 'zh']
    elif only_en:
        all_tasks = [t for t in all_tasks if t.get('language') == 'en']

    if limit is not None and limit > 0:
        all_tasks = all_tasks[:limit]

    task_prompts = {t['prompt'] for t in all_tasks if 'prompt' in t}

    all_criteria = load_jsonl(CRITERIA_FILE)
    criteria_map = {c['prompt']: c for c in all_criteria if c.get('prompt') in task_prompts}

    target_file = os.path.join(cleaned_data_dir, f"{target_model}.jsonl")
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Cleaned target file not found: {target_file}. Run cleaning first (or drop --skip_cleaning).")
    all_target_articles = load_jsonl(target_file)
    target_articles_map = {a['prompt']: a for a in all_target_articles if a.get('prompt') in task_prompts}

    all_reference_articles = load_jsonl(REFERENCE_FILE)
    reference_articles_map = {a['prompt']: a for a in all_reference_articles if a.get('prompt') in task_prompts}

    tasks_to_process = []
    for task in all_tasks:
        prompt = task.get('prompt')
        if prompt in criteria_map and prompt in target_articles_map and prompt in reference_articles_map:
            tasks_to_process.append(task)
        else:
            missing = [name for name, m in (("criteria", criteria_map), ("target", target_articles_map),
                                             ("reference", reference_articles_map)) if prompt not in m]
            logger.warning(f"Skipping task id={task.get('id')}: missing {missing}")

    return tasks_to_process, criteria_map, target_articles_map, reference_articles_map


def build_user_prompt(task, target_articles_map, reference_articles_map, criteria_map):
    prompt = task['prompt']
    language = task.get('language', 'en')
    target_article = target_articles_map[prompt].get('article', '')
    reference_article = reference_articles_map[prompt].get('article', '')
    criteria_data = criteria_map[prompt]
    criteria_list_str = format_criteria_list(criteria_data)
    merged_score_prompt = zh_merged_score_prompt if language == "zh" else en_merged_score_prompt
    return merged_score_prompt.format(
        task_prompt=prompt,
        article_1=target_article,
        article_2=reference_article,
        criteria_list=criteria_list_str
    )


def cmd_submit(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_cleaning:
        clean_agent = AIClient(backend="gemini", model=args.model_name)
        cleaner = ArticleCleaner(clean_agent)
        # Mirrors main()'s calling order in deepresearch_bench_race.py: clean_articles
        # processes every item in the raw file regardless of the language argument, and
        # is a no-op on the second call once all ids are already in the output file.
        if not args.only_en:
            cleaner.clean_articles(args.target_model, args.raw_data_dir, args.cleaned_data_dir,
                                    args.max_workers, MAX_RETRIES, args.limit, "zh")
        if not args.only_zh:
            cleaner.clean_articles(args.target_model, args.raw_data_dir, args.cleaned_data_dir,
                                    args.max_workers, MAX_RETRIES, args.limit, "en")

    tasks_to_process, criteria_map, target_articles_map, reference_articles_map = load_tasks(
        args.target_model, args.cleaned_data_dir, args.query_file, args.only_zh, args.only_en, args.limit
    )

    if not tasks_to_process:
        logger.error("No tasks with complete criteria/target/reference data found. Aborting.")
        return

    requests_path = os.path.join(args.output_dir, "batch_requests.jsonl")
    num_written = 0
    with open(requests_path, 'w', encoding='utf-8') as f:
        for task in tasks_to_process:
            task_id = task['id']
            try:
                user_prompt = build_user_prompt(task, target_articles_map, reference_articles_map, criteria_map)
            except Exception as e:
                logger.warning(f"Skipping id={task_id}: failed to build prompt: {e}")
                continue
            line = {
                "key": str(task_id),
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "generation_config": {"temperature": 0}
                }
            }
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
            num_written += 1

    logger.info(f"Wrote {num_written} batch requests to {requests_path}")

    client = _get_genai_client()
    from google.genai import types

    uploaded_file = client.files.upload(
        file=requests_path,
        config=types.UploadFileConfig(display_name=f"{args.target_model}-race-batch", mime_type='jsonl')
    )

    batch_job = client.batches.create(
        model=args.model_name,
        src=uploaded_file.name,
        config=types.CreateBatchJobConfig(display_name=f"race-{args.target_model}")
    )

    job_meta = {
        "job_name": batch_job.name,
        "model_name": args.model_name,
        "target_model": args.target_model,
        "cleaned_data_dir": args.cleaned_data_dir,
        "query_file": args.query_file,
        "only_zh": args.only_zh,
        "only_en": args.only_en,
        "limit": args.limit,
        "num_requests": num_written,
        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    job_meta_path = os.path.join(args.output_dir, "batch_job.json")
    with open(job_meta_path, 'w', encoding='utf-8') as f:
        json.dump(job_meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Submitted batch job: {batch_job.name}  (state: {batch_job.state.name})")
    logger.info(f"Job metadata saved to {job_meta_path}")
    logger.info(f"Later, run: python -m utils.race_batch fetch {args.target_model} --output_dir {args.output_dir}")


def _extract_response_text(obj):
    """obj is one parsed JSONL line from the downloaded result file. Try the
    documented {"key": ..., "response": {"candidates": [...]}} shape, and a
    couple of plausible fallbacks, since Google's public docs don't nail down
    the exact schema."""
    if 'error' in obj and obj['error']:
        raise ValueError(str(obj['error']))

    resp = obj.get('response', obj)
    candidates = resp.get('candidates') if isinstance(resp, dict) else None
    if not candidates:
        raise ValueError(f"No candidates in response: {str(resp)[:300]}")

    parts = candidates[0].get('content', {}).get('parts', [])
    for part in parts:
        if 'text' in part:
            return part['text']
    raise ValueError(f"No text part in response candidate: {str(candidates[0])[:300]}")


def cmd_fetch(args):
    job_meta_path = os.path.join(args.output_dir, "batch_job.json")
    if not os.path.exists(job_meta_path):
        logger.error(f"No batch_job.json found in {args.output_dir}. Run `submit` first.")
        return
    with open(job_meta_path, encoding='utf-8') as f:
        job_meta = json.load(f)

    if args.target_model != job_meta.get("target_model"):
        logger.warning(f"target_model argument ({args.target_model}) does not match the "
                        f"submitted job's target_model ({job_meta.get('target_model')}); using the submitted one.")

    client = _get_genai_client()

    batch_job = client.batches.get(name=job_meta["job_name"])
    state = batch_job.state.name
    logger.info(f"Job {job_meta['job_name']} state: {state}")

    if state not in TERMINAL_STATES:
        logger.info("Job is still running. Nothing to fetch yet, try again later.")
        return

    if state not in SUCCESS_STATES:
        logger.error(f"Batch job did not succeed: {state}. Error: {getattr(batch_job, 'error', None)}")
        return

    result_file_name = getattr(batch_job.dest, 'file_name', None) if batch_job.dest else None
    if not result_file_name:
        logger.error("Batch job finished but no output file was found on the job (dest.file_name is empty).")
        return

    file_content = client.files.download(file=result_file_name)
    text = file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content

    key_to_text = {}
    key_to_error = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse batch result line: {line[:200]}")
            continue
        key = obj.get('key')
        if key is None:
            continue
        try:
            key_to_text[key] = _extract_response_text(obj)
        except ValueError as e:
            key_to_error[key] = str(e)

    tasks_to_process, criteria_map, target_articles_map, reference_articles_map = load_tasks(
        job_meta["target_model"], job_meta["cleaned_data_dir"], job_meta["query_file"],
        job_meta["only_zh"], job_meta["only_en"], job_meta["limit"]
    )

    all_results = []
    for task in tasks_to_process:
        task_id = task['id']
        key = str(task_id)
        prompt = task['prompt']
        language = task.get('language', 'en')

        if key in key_to_error:
            all_results.append({"id": task_id, "prompt": prompt, "error": key_to_error[key]})
            continue
        if key not in key_to_text:
            all_results.append({"id": task_id, "prompt": prompt, "error": "No batch result for this task"})
            continue

        llm_response_str = key_to_text[key]
        try:
            json_str_extracted = extract_json_from_markdown(llm_response_str)
            if not json_str_extracted:
                raise ValueError("Failed to extract JSON from LLM response")
            llm_output_json = json.loads(json_str_extracted)
            if not all(dim in llm_output_json for dim in EXPECTED_DIMS):
                missing = [d for d in EXPECTED_DIMS if d not in llm_output_json]
                raise ValueError(f"Missing expected dimensions: {missing}")

            criteria_data = criteria_map[prompt]
            scores = calculate_weighted_scores(llm_output_json, criteria_data, language)
            target_total = scores["target"]["total"]
            reference_total = scores["reference"]["total"]
            overall_score = (target_total / (target_total + reference_total)
                              if (target_total + reference_total) > 0 else 0)

            normalized_dims = {}
            for dim in EXPECTED_DIMS:
                dim_key = f"{dim}_weighted_avg"
                t = scores["target"]["dims"].get(dim_key, 0)
                r = scores["reference"]["dims"].get(dim_key, 0)
                normalized_dims[dim] = t / (t + r) if (t + r) > 0 else 0

            all_results.append({
                "id": task_id,
                "prompt": prompt,
                "comprehensiveness": normalized_dims.get("comprehensiveness", 0),
                "insight": normalized_dims.get("insight", 0),
                "instruction_following": normalized_dims.get("instruction_following", 0),
                "readability": normalized_dims.get("readability", 0),
                "overall_score": overall_score
            })
        except Exception as e:
            all_results.append({
                "id": task_id,
                "prompt": prompt,
                "error": f"Error processing batch result: {str(e)}",
                "model_output": llm_response_str[:500] if llm_response_str else "No response"
            })

    all_results.sort(key=lambda x: x.get('id', float('inf')))

    output_file = os.path.join(args.output_dir, "raw_results.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    successful = [r for r in all_results if "error" not in r]
    result_file = os.path.join(args.output_dir, "race_result.txt")
    if successful:
        comp = sum(r["comprehensiveness"] for r in successful) / len(successful)
        ins = sum(r["insight"] for r in successful) / len(successful)
        instf = sum(r["instruction_following"] for r in successful) / len(successful)
        read = sum(r["readability"] for r in successful) / len(successful)
        overall = sum(r["overall_score"] for r in successful) / len(successful)
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"Comprehensiveness: {comp:.4f}\n")
            f.write(f"Insight: {ins:.4f}\n")
            f.write(f"Instruction Following: {instf:.4f}\n")
            f.write(f"Readability: {read:.4f}\n")
            f.write(f"Overall Score: {overall:.4f}\n")
        logger.info(f"Scored {len(successful)}/{len(all_results)} tasks successfully.")
        logger.info(f"Overall Score: {overall:.4f}")
    else:
        logger.warning("No successful results; race_result.txt not written.")

    logger.info(f"Results written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Submit/fetch RACE scoring via the Gemini Batch API.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_p = subparsers.add_parser("submit", help="Build scoring prompts and submit a batch job.")
    submit_p.add_argument("target_model", type=str)
    submit_p.add_argument("--output_dir", type=str, required=True)
    submit_p.add_argument("--raw_data_dir", type=str, default="data/test_data/raw_data")
    submit_p.add_argument("--cleaned_data_dir", type=str, default="data/test_data/cleaned_data")
    submit_p.add_argument("--query_file", type=str, default="data/prompt_data/query.jsonl")
    submit_p.add_argument("--only_zh", action="store_true")
    submit_p.add_argument("--only_en", action="store_true")
    submit_p.add_argument("--limit", type=int, default=None)
    submit_p.add_argument("--skip_cleaning", action="store_true")
    submit_p.add_argument("--max_workers", type=int, default=5, help="Used for the (synchronous) cleaning step only.")
    submit_p.add_argument("--model_name", type=str, default=DEFAULT_GEMINI_MODEL,
                           help="Gemini model to use for both cleaning and batch scoring (must support Batch mode).")

    fetch_p = subparsers.add_parser("fetch", help="Check job status and, if done, write raw_results.jsonl / race_result.txt.")
    fetch_p.add_argument("target_model", type=str)
    fetch_p.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    if args.command == "submit":
        cmd_submit(args)
    else:
        cmd_fetch(args)


if __name__ == "__main__":
    main()
