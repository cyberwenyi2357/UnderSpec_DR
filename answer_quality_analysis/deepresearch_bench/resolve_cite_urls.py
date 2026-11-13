"""Build FACT input from an agent's raw output: resolve inline <cite id="..."> tags to the URLs they cite.

FACT's extract step only recognizes citations that carry a URL (numbered references or markdown links). The
GPT-5.4 and DR Tulu answers cite with <cite id="<call_id>-<n>">text</cite> instead, where <call_id> is a tool
call and <n> the index of a document it returned. This resolves each tag through the trajectory's
query_full_traces.tool_calls[].documents[] and rewrites it as [text](url), so extract can find it.

- A tag citing several ids ("a-1,b-0") takes the first id that resolves.
- Unresolvable tags (e.g. short "S12"/"W3" ids some GPT-5.4 answers use) are replaced by their plain text.
- The answers stored in this dataset keep the raw tags and have no trajectories, so the input here is the
  agent's original output: one JSON object per line with id, query_answer and query_full_traces.

Usage:
    python resolve_cite_urls.py --input answer_deepresearch_bench_query_5.4.jsonl \
        --output data/fact_input/gpt5.4_query.jsonl \
        [--exclude_judge ../../data/deepresearch_bench/ambiguity_judge/gpt-5.4/query.jsonl]
"""

import argparse
import json
import re
from pathlib import Path

QUERY_FILE = Path(__file__).resolve().parent / "data" / "prompt_data" / "query.jsonl"
CITE_RE = re.compile(r'<cite id="([^"]+)">(.*?)</cite>', re.DOTALL)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_url_map(query_full_traces):
    m = {}
    for t in (query_full_traces.get("tool_calls") or []):
        cid = t.get("call_id")
        if not cid:
            continue
        for i, doc in enumerate(t.get("documents") or []):
            if doc.get("url"):
                m[f"{cid}-{i}"] = doc["url"]
    return m


def resolve(article, url_map, counters):
    def repl(m):
        counters[0] += 1
        ids = [s.strip() for s in m.group(1).split(",")]
        text = m.group(2)
        url = next((url_map[i] for i in ids if i in url_map), None)
        if url:
            counters[1] += 1
            return f"[{text.replace('[', '(').replace(']', ')')}]({url})"
        return text
    return CITE_RE.sub(repl, article)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="agent output jsonl (id, query_answer, query_full_traces)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude_judge", help="ambiguity-judge jsonl; ids judged asked_clarifying_question are dropped")
    args = parser.parse_args()

    prompt_by_id = {r["id"]: r["prompt"] for r in read_jsonl(QUERY_FILE)}
    exclude = set()
    if args.exclude_judge:
        exclude = {r["id"] for r in read_jsonl(args.exclude_judge) if r.get("category") == "asked_clarifying_question"}

    counters = [0, 0]  # cite tags seen, tags resolved to a URL
    out = []
    for d in read_jsonl(args.input):
        if d["id"] in exclude:
            continue
        url_map = build_url_map(d.get("query_full_traces") or {})
        out.append({"id": d["id"], "prompt": prompt_by_id.get(d["id"], d.get("query", "")),
                    "article": resolve(d.get("query_answer", "") or "", url_map, counters)})
    out.sort(key=lambda r: r["id"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rate = counters[1] / counters[0] * 100 if counters[0] else 0
    print(f"{len(out)} answers ({len(exclude)} excluded), {counters[1]}/{counters[0]} cite tags resolved ({rate:.1f}%)")


if __name__ == "__main__":
    main()
