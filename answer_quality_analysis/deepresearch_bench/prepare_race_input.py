"""Turn one of this dataset's answer files into RACE input.

RACE joins an answer to its scoring criteria and reference report by the task's prompt text in
DeepResearch Bench's query.jsonl, not by id, so every row gets that canonical prompt (for every variant: all
variants are scored against the original task's criteria and reference). The answer keeps its raw
<cite id="..."> tags, as when the dataset's RACE scores were produced.

Usage:
    python prepare_race_input.py --answers ../../data/deepresearch_bench/answers/gpt-5.4/query.jsonl \
        --name gpt5.4_query \
        [--exclude_judge ../../data/deepresearch_bench/ambiguity_judge/gpt-5.4/query.jsonl]

writes data/test_data/raw_data/<name>.jsonl, the file deepresearch_bench_race.py / utils.race_batch read
for target model <name>.
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUERY_FILE = HERE / "data" / "prompt_data" / "query.jsonl"
RAW_DIR = HERE / "data" / "test_data" / "raw_data"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, help="dataset answers jsonl ({id, answer})")
    parser.add_argument("--name", required=True, help="target model name for RACE (output file stem)")
    parser.add_argument("--exclude_judge", help="ambiguity-judge jsonl; ids judged asked_clarifying_question are dropped")
    args = parser.parse_args()

    prompt_by_id = {r["id"]: r["prompt"] for r in read_jsonl(QUERY_FILE)}
    exclude = set()
    if args.exclude_judge:
        exclude = {r["id"] for r in read_jsonl(args.exclude_judge) if r.get("category") == "asked_clarifying_question"}

    rows = [{"id": r["id"], "prompt": prompt_by_id[r["id"]], "article": r["answer"]}
            for r in read_jsonl(args.answers) if r["id"] not in exclude]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{args.name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} answers ({len(exclude)} excluded) -> {out}")


if __name__ == "__main__":
    main()
