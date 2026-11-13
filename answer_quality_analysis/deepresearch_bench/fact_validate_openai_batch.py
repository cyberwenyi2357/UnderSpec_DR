"""
Run FACT's `validate` stage with GPT-5.4-mini via OpenAI's Batch API, on the
`scraped.jsonl` that utils.extract -> utils.deduplicate -> utils.scrape left
in each FACT working directory (one directory per model x variant).

Writes `validated_gpt.jsonl` / `fact_result_gpt.txt` into each directory, next
to the `validated.jsonl` / `fact_result.txt` of the synchronous utils.validate
(Gemini) path, so the two judges don't overwrite each other.

Usage (submit/fetch):

    python fact_validate_openai_batch.py submit DIR [DIR ...] [--limit N] [--force]
    python fact_validate_openai_batch.py fetch
"""

import argparse
import json
import logging
import os
import time

from utils.io_utils import load_jsonl
from utils.json_extractor import extract_json_from_markdown
import utils.validate as validate_mod

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL = "gpt-5.4-mini"
QUERY_DATA_PATH = "data/prompt_data/query.jsonl"


def _get_openai_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)


def _load_id_to_lang_map():
    query_data = load_jsonl(QUERY_DATA_PATH)
    m = {item['id']: item.get('language') for item in query_data if 'id' in item and 'language' in item}
    if not m:
        raise ValueError("No valid language information found in query data")
    return m


def cmd_submit(args):
    os.makedirs(args.bookkeeping_dir, exist_ok=True)
    id_to_lang_map = _load_id_to_lang_map()

    requests = []
    index_rows = []
    next_key = 0
    n_skipped_no_ref = 0
    n_skipped_lang = 0

    for name in args.dirs:
        model_dir = name
        scraped_path = os.path.join(model_dir, "scraped.jsonl")
        output_path = os.path.join(model_dir, "validated_gpt.jsonl")

        processed = set()
        if os.path.exists(output_path) and not args.force:
            processed = {d['id'] for d in load_jsonl(output_path)}

        raw_data = load_jsonl(scraped_path)
        data_to_process = [d for d in raw_data if d['id'] not in processed]

        n_this_model = 0
        for d in data_to_process:
            article_id = d.get('id')
            if not article_id:
                continue
            for url, ref in d.get('citations_deduped', {}).items():
                if ref.get('url_content') is None:
                    n_skipped_no_ref += 1
                    continue
                if article_id not in id_to_lang_map or id_to_lang_map[article_id] not in ("zh", "en"):
                    n_skipped_lang += 1
                    continue
                lang = id_to_lang_map[article_id]
                facts = ref['facts']
                facts_str = '\n'.join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])
                template = validate_mod.prompt_template if lang == "zh" else validate_mod.prompt_template_en
                user_prompt = template.format(reference=ref['url_content'], statements=facts_str)

                key = str(next_key)
                next_key += 1
                requests.append({
                    "custom_id": key,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": user_prompt}],
                        "temperature": 0,
                    }
                })
                index_rows.append({"key": key, "model_dir": name, "id": article_id, "url": url, "num_facts": len(facts)})
                n_this_model += 1
                if args.limit is not None and n_this_model >= args.limit:
                    break
            if args.limit is not None and n_this_model >= args.limit:
                break

        logger.info(f"{name}: queued {n_this_model} citation-validation requests (skipped {len(processed)} already-done articles)")

    if n_skipped_no_ref:
        logger.info(f"{n_skipped_no_ref} citations have no reference content; will be marked 'no reference' directly at fetch time.")
    if n_skipped_lang:
        logger.warning(f"Skipped {n_skipped_lang} citations with no/unsupported article language.")

    if not requests:
        logger.error("Nothing to submit.")
        return

    index_path = os.path.join(args.bookkeeping_dir, "request_index.jsonl")
    with open(index_path, 'w', encoding='utf-8') as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # OpenAI batch input files have a size cap; reference-page content makes
    # these requests large, so split into chunks well under that limit
    # instead of uploading everything as one file.
    chunk_size = args.chunk_size
    chunks = [requests[i:i + chunk_size] for i in range(0, len(requests), chunk_size)]

    client = _get_openai_client()
    batches = []
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(args.bookkeeping_dir, f"batch_requests_{i}.jsonl")
        with open(chunk_path, 'w', encoding='utf-8') as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        uploaded = client.files.create(file=open(chunk_path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batches.append({"batch_id": batch.id, "num_requests": len(chunk)})
        logger.info(f"Submitted chunk {i+1}/{len(chunks)}: batch {batch.id} with {len(chunk)} requests (status: {batch.status}).")

    job_meta = {
        "batches": batches,
        "dirs": args.dirs,
        "num_requests": len(requests),
        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.bookkeeping_dir, "batch_job.json"), 'w', encoding='utf-8') as f:
        json.dump(job_meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Submitted {len(batches)} batch job(s) covering {len(requests)} total requests.")
    logger.info(f"Later, run: python fact_validate_openai_batch.py fetch --bookkeeping_dir {args.bookkeeping_dir}")


TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _write_stat(output_path, result_path):
    recs = load_jsonl(output_path)
    total_citations = 0
    total_valid_citations = 0
    total_num = 0
    for d in recs:
        if not d.get('citations'):
            continue
        for c in d.get('citations_deduped', {}).values():
            if c.get('validate_error') is not None:
                continue
            for r in c.get('validate_res', []):
                result = r.get('result')
                if result is None:
                    # a handful of GPT-5.4-mini responses used a different key
                    # for one entry (e.g. {"idx":.., "supported": "supported"}
                    # instead of {"idx":.., "result": "supported"}); treat
                    # those like 'unknown' rather than crashing the whole run.
                    logger.warning(f"id={d.get('id')}: validate_res entry missing 'result' key: {r}")
                    continue
                if result != 'unknown':
                    total_citations += 1
                    if result == 'supported':
                        total_valid_citations += 1
        total_num += 1
    if total_num == 0:
        logger.warning(f"No records with citations in {output_path}; skipping stat.")
        return
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f'total_citations: {total_citations/total_num}\n')
        f.write(f'total_valid_citations: {total_valid_citations/total_num}\n')
        if total_citations:
            f.write(f'valid_rate: {total_valid_citations/total_citations}\n')


def cmd_fetch(args):
    job_meta_path = os.path.join(args.bookkeeping_dir, "batch_job.json")
    index_path = os.path.join(args.bookkeeping_dir, "request_index.jsonl")
    if not os.path.exists(job_meta_path):
        logger.error(f"No batch_job.json found in {args.bookkeeping_dir}. Run `submit` first.")
        return
    with open(job_meta_path, encoding='utf-8') as f:
        job_meta = json.load(f)

    client = _get_openai_client()
    batches = [client.batches.retrieve(b["batch_id"]) for b in job_meta["batches"]]
    for batch in batches:
        logger.info(f"Batch {batch.id} status: {batch.status} "
                    f"(completed={batch.request_counts.completed if batch.request_counts else '?'}, "
                    f"failed={batch.request_counts.failed if batch.request_counts else '?'}, "
                    f"total={batch.request_counts.total if batch.request_counts else '?'})")

    not_done = [b for b in batches if b.status not in TERMINAL_STATUSES]
    if not_done:
        logger.info(f"{len(not_done)}/{len(batches)} chunk(s) still running. "
                     f"Waiting for ALL chunks before writing anything (to avoid marking articles "
                     f"'done' while some of their citations are still pending). Try again later.")
        return

    failed = [b for b in batches if b.status != "completed"]
    if failed:
        for b in failed:
            logger.error(f"Chunk batch {b.id} did not complete successfully: {b.status}. Error file id: {b.error_file_id}")
        # still proceed with whatever completed chunks produced, rather than
        # losing all progress because one chunk failed.

    key_to_text = {}
    for batch in batches:
        if not batch.output_file_id:
            continue
        content = client.files.content(batch.output_file_id).text
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("custom_id")
            if key is None or obj.get("error"):
                continue
            body = (obj.get("response") or {}).get("body") or {}
            try:
                key_to_text[key] = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                continue

    index_rows = load_jsonl(index_path)
    route = {}  # (model_dir, id) -> {url: (key, num_facts)}
    for row in index_rows:
        route.setdefault((row['model_dir'], row['id']), {})[row['url']] = (row['key'], row['num_facts'])

    for name in job_meta["dirs"]:
        model_dir = name
        scraped_path = os.path.join(model_dir, "scraped.jsonl")
        output_path = os.path.join(model_dir, "validated_gpt.jsonl")
        result_path = os.path.join(model_dir, "fact_result_gpt.txt")

        processed = set()
        if os.path.exists(output_path):
            processed = {d['id'] for d in load_jsonl(output_path)}

        raw_data = load_jsonl(scraped_path)
        data_to_process = [d for d in raw_data if d['id'] not in processed]
        if not data_to_process:
            continue

        n_written = 0
        with open(output_path, "a", encoding="utf-8") as out_f:
            for d in data_to_process:
                article_id = d.get('id')
                if not article_id:
                    continue
                for url, ref in d.get('citations_deduped', {}).items():
                    if ref.get('url_content') is None:
                        ref['validate_res'] = []
                        ref['validate_error'] = "no reference"
                        continue

                    key_info = route.get((name, article_id), {}).get(url)
                    if key_info is None:
                        ref['validate_res'] = []
                        ref['validate_error'] = "no batch request was built for this citation"
                        continue
                    key, num_facts = key_info

                    response = key_to_text.get(key)
                    if response is None:
                        ref['validate_res'] = []
                        ref['validate_error'] = "no batch result for this citation (batch-level error)"
                        continue

                    try:
                        cleaned = response.replace("```json", "").replace("```", "")
                        extracted = extract_json_from_markdown(cleaned) or cleaned
                        validate_res = json.loads(extracted)
                        for v in validate_res:
                            v['idx'] -= 1
                        assert len(validate_res) == num_facts
                        ref['validate_res'] = validate_res
                        ref['validate_error'] = None
                    except Exception as e:
                        ref['validate_res'] = []
                        ref['validate_error'] = str(e)

                out_f.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_written += 1

        logger.info(f"{name}: wrote {n_written} records -> {output_path}")
        _write_stat(output_path, result_path)
        logger.info(f"{name}: stat -> {result_path}")

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit")
    p.add_argument("dirs", nargs="+", help="FACT working directories (one per model x variant), each holding scraped.jsonl.")
    p.add_argument("--limit", type=int, default=None, help="Limit number of NEW citation requests queued per model dir (for testing).")
    p.add_argument("--force", action="store_true", help="Re-validate even if the article id is already in validated_gpt.jsonl.")
    p.add_argument("--chunk_size", type=int, default=1500, help="Max requests per batch input file (OpenAI batch files have a size cap; reference-page content makes each request large).")
    p.add_argument("--bookkeeping_dir", default="fact_gpt_validate_batch", help="Where the batch requests, request index and job ids are kept.")

    p = sub.add_parser("fetch")
    p.add_argument("--bookkeeping_dir", default="fact_gpt_validate_batch")

    args = parser.parse_args()
    if args.command == "submit":
        cmd_submit(args)
    else:
        cmd_fetch(args)


if __name__ == "__main__":
    main()
