"""Ambiguity-awareness judge: did the agent recognize that the query it was given was ambiguous, and how did it react?

gpt-5.4-mini reads the query the agent was actually given (the variant text) and the agent's full answer, and
classifies the answer into one of:
    asked_clarifying_question         asked the user to clarify, before or instead of answering
    stated_assumption_and_proceeded   noticed the ambiguity, stated assumptions, then answered in full
    no_recognition                    answered as if the query were fully specified
with a short evidence quote (or, for no_recognition, a one-sentence explanation).

The prompt, model and parsing are those of the original judge_ambiguity_awareness.py (identical for ResearchQA,
DeepResearch Bench and ScholarQA-Bench). Input here is the per benchmark x model files of rated_answer/
({"questions": [{"id", "variants": {<variant>: {"text", "answer", ...}}}]}), so any of them can be judged.

It runs on the OpenAI Batch API (minutes to ~24h), so it is submit / fetch:

    python answer_quality_analysis/ambiguity_awareness_judge.py submit rated_answer/*.json [--limit N] [--force]
    python answer_quality_analysis/ambiguity_awareness_judge.py fetch

submit queues every (question, variant) with both a query text and an answer that is not judged yet (custom_id
"<file stem>::<subset/>id::<variant>") and writes bookkeeping to <out-dir>/batch_job.json. fetch downloads the
results when the batch is done and appends {"id", "subset", "variant", "category", "evidence", "error"} rows to
<out-dir>/<file stem>.jsonl (resumable: run submit again to pick up anything still missing).
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

MODEL = "gpt-5.4-mini"
CATEGORIES = ["asked_clarifying_question", "stated_assumption_and_proceeded", "no_recognition"]
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

JUDGE_PROMPT = """You are evaluating how a deep research AI agent handled a user's query.

Below is the ORIGINAL USER QUERY the agent was given, followed by the AGENT'S FULL RESPONSE.

Your task: determine whether the agent recognized that the user's query was ambiguous, underspecified, or missing information needed to answer it well, AND how it reacted to that.

Classify the agent's behavior into exactly ONE of these three categories:
- "asked_clarifying_question": the agent explicitly asked the user one or more questions to clarify the request, before or instead of giving a full answer.
- "stated_assumption_and_proceeded": the agent recognized the ambiguity, proposed one or more explicit assumptions/interpretations to resolve it, and then went ahead and produced a full (typically long) answer on that basis.
- "no_recognition": the agent showed no sign of recognizing any ambiguity in the query and just answered as if the query were fully specified.

Also provide:
- "evidence": a short (at most 2 sentences) quote or close paraphrase from the response that supports your classification, in the original language of the response. If the category is "no_recognition", instead give a one-sentence explanation of why there is no sign of recognition.

Respond with ONLY a single JSON object, no other text, no markdown code fences:
{{"category": "asked_clarifying_question" | "stated_assumption_and_proceeded" | "no_recognition", "evidence": "..."}}

<original_user_query>
{query}
</original_user_query>

<agent_response>
{response}
</agent_response>
"""


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def openai_client():
    """OpenAI client from OPENAI_API_KEY (environment, or a .env file when python-dotenv is installed)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def extract_json_from_markdown(text):
    """Strip ```json ... ``` / ``` ... ``` fences around a JSON object, if present."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def parse_reply(text):
    """Model reply -> (category, evidence); raises when there is no parsable JSON with a category."""
    extracted = extract_json_from_markdown(text)
    parsed = json.loads(extracted) if extracted else json.loads(text)
    if "category" not in parsed:
        raise ValueError(f"missing 'category' in parsed response: {parsed}")
    return parsed["category"], parsed.get("evidence")


def item_key(subset, qid, variant):
    return f"{subset + '/' if subset else ''}{qid}::{variant}"


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_requests(path, out_dir, force=False, limit=None):
    stem = Path(path).stem
    done = set()
    if not force:
        done = {item_key(r.get("subset"), r["id"], r["variant"]) for r in read_jsonl(out_dir / f"{stem}.jsonl")
                if r.get("category") is not None}
    requests, n_missing = [], 0
    for q in load_json(path)["questions"]:
        for variant, e in q["variants"].items():
            key = item_key(q.get("subset"), q["id"], variant)
            if key in done:
                continue
            if not (e.get("text") and e.get("answer")):
                n_missing += 1
                continue
            requests.append({
                "custom_id": f"{stem}::{key}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": MODEL, "temperature": 0,
                         "messages": [{"role": "user",
                                       "content": JUDGE_PROMPT.format(query=e["text"], response=e["answer"])}]},
            })
            if limit is not None and len(requests) >= limit:
                return requests, n_missing, len(done)
    return requests, n_missing, len(done)


def cmd_submit(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    for path in args.files:
        reqs, n_missing, n_done = build_requests(path, out_dir, args.force, args.limit)
        print(f"{Path(path).stem}: queued {len(reqs)} (already judged {n_done}, no query text or answer {n_missing})")
        requests += reqs
    if not requests:
        print("Nothing to submit.")
        return
    req_path = out_dir / "batch_requests.jsonl"
    with open(req_path, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    client = openai_client()
    uploaded = client.files.create(file=open(req_path, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h")
    meta = {"batch_id": batch.id, "num_requests": len(requests), "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "batch_job.json").write_text(json.dumps(meta, indent=2))
    print(f"Submitted batch {batch.id} with {len(requests)} requests (status: {batch.status}). Run `fetch` later.")


def result_row(obj):
    stem, key = obj["custom_id"].split("::", 1)
    item, variant = key.rsplit("::", 1)
    subset, _, qid = item.rpartition("/")
    row = {"id": qid, "subset": subset or None, "variant": variant}
    if obj.get("error"):
        return stem, {**row, "category": None, "evidence": None, "error": str(obj["error"])}
    try:
        text = obj["response"]["body"]["choices"][0]["message"]["content"]
        category, evidence = parse_reply(text)
        return stem, {**row, "category": category, "evidence": evidence, "error": None}
    except Exception as e:  # noqa: BLE001
        return stem, {**row, "category": None, "evidence": None, "error": f"failed to parse response: {e}"}


def cmd_fetch(args):
    out_dir = Path(args.out_dir)
    meta_path = out_dir / "batch_job.json"
    if not meta_path.exists():
        print(f"No batch_job.json in {out_dir}. Run `submit` first.")
        return
    client = openai_client()
    batch = client.batches.retrieve(json.loads(meta_path.read_text())["batch_id"])
    rc = batch.request_counts
    print(f"Batch {batch.id}: {batch.status} (completed {rc.completed if rc else '?'}, "
          f"failed {rc.failed if rc else '?'}, total {rc.total if rc else '?'})")
    if batch.status not in TERMINAL_STATUSES:
        print("Still running, try again later.")
        return
    if batch.status != "completed" and batch.error_file_id:
        print("Error file head:", client.files.content(batch.error_file_id).text[:2000])
    if not batch.output_file_id:
        return
    by_file = {}
    for line in client.files.content(batch.output_file_id).text.splitlines():
        if line.strip():
            stem, row = result_row(json.loads(line))
            by_file.setdefault(stem, []).append(row)
    for stem, rows in by_file.items():
        with open(out_dir / f"{stem}.jsonl", "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{stem}: wrote {len(rows)} results -> {out_dir / (stem + '.jsonl')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", default="results/ambiguity_judge")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("submit")
    p.add_argument("files", nargs="+", help="rated_answer/{benchmark}_{model}.json files")
    p.add_argument("--limit", type=int, default=None, help="max NEW requests per file (for testing)")
    p.add_argument("--force", action="store_true", help="re-judge items already in the output file")
    sub.add_parser("fetch")
    args = parser.parse_args()
    cmd_submit(args) if args.command == "submit" else cmd_fetch(args)


if __name__ == "__main__":
    main()
