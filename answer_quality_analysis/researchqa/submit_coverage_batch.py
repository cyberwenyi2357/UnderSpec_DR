"""
Submit rubric-coverage scoring for a response file as a Gemini Batch API job
(client.batches) instead of calling generate_content synchronously per batch.
The Batch API is priced at a discount vs. real-time calls, at the cost of not
getting results back immediately.

Reuses build_prompt() from compute_coverage.py so the judging prompt is byte-for-byte
identical to the real-time path. Writes a <output>.batch_meta.json file recording the
batch job name and, for every submitted request, which item_id/rubric chunk it
corresponds to - scoring/fetch_coverage_batch_results.py needs this to reassemble
results once the job finishes.

Note: this submits all requests inline in one call (types.InlinedRequest). Gemini's
Batch API also supports much larger jobs via a GCS-backed source file, which this
script does not implement - if you hit an inline-size error on a very large run,
that's the path to add.

Usage:
    python scoring/submit_coverage_batch.py \
        --data data/query/researchqa_rubrics.json \
        --responses data/answers/gpt-5.4/query.jsonl \
        --output scores_query_gpt-5.4.json \
        --batch_size 8 \
        --model gemini-pro-latest
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compute_coverage import build_prompt, load_response_map
from researchqa_loader import load_researchqa_data

from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Submit rubric-coverage scoring as a Gemini Batch API job")
    parser.add_argument("--data", type=str, required=True, help="Path to ResearchQA JSON file")
    parser.add_argument("--responses", type=str, required=True, help="Path to response map JSON file (id -> response dict)")
    parser.add_argument("--output", type=str, required=True, help="Path the final results will be written to by fetch_coverage_batch_results.py")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for rubric items")
    parser.add_argument("--model", type=str, default="gemini-pro-latest", help="Gemini model name")
    args = parser.parse_args()

    items = load_researchqa_data(args.data)
    response_map = load_response_map(args.responses)

    inline_requests = []
    request_meta = []
    skipped = 0
    for item in items:
        item_id = item.id
        if item_id not in response_map:
            skipped += 1
            continue
        answer = response_map[item_id]["answer"]
        rubric_items = [r.rubric_item for r in item.rubric]
        for i in range(0, len(rubric_items), args.batch_size):
            batch = rubric_items[i:i + args.batch_size]
            prompt = build_prompt(answer, batch)
            tag = f"{item_id}::{i}"
            inline_requests.append(types.InlinedRequest(
                contents=prompt,
                metadata={"tag": tag},
                config=types.GenerateContentConfig(temperature=0),
            ))
            request_meta.append({"tag": tag, "item_id": item_id, "chunk_index": i, "rubric_batch": batch})

    print(f"Skipped {skipped} response(s) with no matching item in {args.data}")
    n_items = len(set(m["item_id"] for m in request_meta))
    print(f"Submitting {len(inline_requests)} request(s) covering {n_items} item(s)")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    job = client.batches.create(model=args.model, src=inline_requests)
    print(f"Created batch job: {job.name} (state: {job.state.name})")

    meta_path = args.output + ".batch_meta.json"
    meta = {
        "job_name": job.name,
        "model": args.model,
        "output": args.output,
        "requests": request_meta,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved batch metadata to {meta_path}")


if __name__ == "__main__":
    main()
