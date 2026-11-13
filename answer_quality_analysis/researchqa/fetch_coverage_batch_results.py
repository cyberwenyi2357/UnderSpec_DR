"""
Poll a Gemini Batch API job created by scoring/submit_coverage_batch.py, and once
it reaches a terminal state, reassemble the per-rubric scores into the exact same
{id: {"coverage_score": ..., "rubric_judges": [...]}} format compute_coverage.py
produces. Supports the same incremental-resume behavior: item ids already present
in --output (from a previous fetch, or from compute_coverage.py itself) are
skipped, so this can be re-run safely and mixed with real-time runs on the same
output file.

Usage:
    # One-shot status check; writes/updates --output if the job has finished.
    python scoring/fetch_coverage_batch_results.py --meta <output>.batch_meta.json

    # Block and poll every 30s until the job finishes, then write automatically.
    python scoring/fetch_coverage_batch_results.py --meta <output>.batch_meta.json --wait
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compute_coverage import compute_coverage_score

from google import genai
from dotenv import load_dotenv
load_dotenv()

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED",
}


def main():
    parser = argparse.ArgumentParser(description="Fetch and reassemble Gemini Batch API rubric-coverage results")
    parser.add_argument("--meta", type=str, required=True, help="Path to the .batch_meta.json file from submit_coverage_batch.py")
    parser.add_argument("--wait", action="store_true", help="Block and poll until the job reaches a terminal state")
    parser.add_argument("--poll_interval", type=int, default=30, help="Seconds between polls when --wait is set")
    args = parser.parse_args()

    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    job = client.batches.get(name=meta["job_name"])
    print(f"Batch job {job.name} state: {job.state.name}")

    if args.wait:
        while job.state.name not in TERMINAL_STATES:
            time.sleep(args.poll_interval)
            job = client.batches.get(name=meta["job_name"])
            print(f"Batch job {job.name} state: {job.state.name}")

    if job.state.name not in TERMINAL_STATES:
        print(f"Job not finished yet (state: {job.state.name}). Re-run this command later to check again.")
        return

    if job.state.name == "JOB_STATE_FAILED":
        print(f"Job failed: {job.error}")
        return

    if not job.dest or not job.dest.inlined_responses:
        print("Job finished but no inlined responses were found.")
        return

    inlined_responses = job.dest.inlined_responses
    # Older google-genai versions (<2.x, the last ones that still support Python <3.10)
    # don't have InlinedResponse.metadata at all - fall back to positional matching
    # against the request order recorded at submit time in that case.
    has_metadata = any(getattr(r, "metadata", None) for r in inlined_responses)

    responses_by_tag = {}
    if has_metadata:
        for r in inlined_responses:
            tag = (getattr(r, "metadata", None) or {}).get("tag")
            if r.error is not None:
                print(f"Warning: request {tag} failed: {r.error}")
                continue
            responses_by_tag[tag] = r.response.text if r.response else ""
    else:
        print("No response metadata available (older google-genai SDK) - matching by submission order instead.")
        if len(inlined_responses) != len(meta["requests"]):
            print(f"Warning: got {len(inlined_responses)} response(s) but submitted {len(meta['requests'])} request(s) - positional matching would be unreliable. Aborting.")
            return
        for r, req in zip(inlined_responses, meta["requests"]):
            if r.error is not None:
                print(f"Warning: request {req['tag']} failed: {r.error}")
                continue
            responses_by_tag[req["tag"]] = r.response.text if r.response else ""

    chunks_by_item = {}
    for req in meta["requests"]:
        chunks_by_item.setdefault(req["item_id"], []).append(req)
    for chunks in chunks_by_item.values():
        chunks.sort(key=lambda c: c["chunk_index"])

    output_path = meta["output"]
    results = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        if results:
            print(f"Resuming from existing output: {len(results)} item(s) already scored will be skipped.")

    n_failed_items = 0
    for item_id, chunks in chunks_by_item.items():
        if item_id in results:
            continue

        rubric_judges = []
        item_failed = False
        for chunk in chunks:
            text = responses_by_tag.get(chunk["tag"])
            if text is None:
                item_failed = True
                break
            scores = [s.strip() for s in text.split("\n") if s.strip()]
            if len(scores) != len(chunk["rubric_batch"]):
                print(f"Warning: item {item_id} chunk {chunk['chunk_index']} expected {len(chunk['rubric_batch'])} scores, got {len(scores)}. Skipping item.")
                item_failed = True
                break
            for rubric, score in zip(chunk["rubric_batch"], scores):
                rubric_judges.append({"rubric": rubric, "score": score})

        if item_failed:
            n_failed_items += 1
            continue

        try:
            coverage_score = compute_coverage_score(rubric_judges)
        except KeyError as e:
            print(f"Warning: item {item_id} had an unrecognized score label {e}. Skipping item.")
            n_failed_items += 1
            continue

        results[item_id] = {"coverage_score": coverage_score, "rubric_judges": rubric_judges}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} item(s) to {output_path} ({n_failed_items} item(s) failed/skipped)")
    if results:
        avg = sum(r["coverage_score"] for r in results.values()) / len(results)
        print(f"Average rubric coverage: {avg:.3f}")


if __name__ == "__main__":
    main()
