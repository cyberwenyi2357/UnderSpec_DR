"""Convert ../query/{benchmark}.jsonl into the eval_data/*.jsonl files read by the queue_runs*.sh scripts.

Tongyi DeepResearch reads one {"question", "answer"} object per line, so every query variant becomes its own file:
    eval_data/{benchmark}[_{subset}]_{variant}.jsonl
e.g. eval_data/researchqa_without_term.jsonl, eval_data/scholarqa_bench_cs_2_query.jsonl.

    python make_eval_data.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUERY_DIR = HERE.parent / "query"
OUT_DIR = HERE / "eval_data"
VARIANTS = ["query", "fully_underspecified", "without_scope", "without_term", "without_preference"]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for path in sorted(QUERY_DIR.glob("*.jsonl")):
        files = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prefix = path.stem + (f"_{row['subset']}" if row.get("subset") else "")
                for v in VARIANTS:
                    if row.get(v):
                        files.setdefault(f"{prefix}_{v}", []).append({"question": row[v], "answer": ""})
        for name, rows in files.items():
            with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{name}: {len(rows)} questions")


if __name__ == "__main__":
    main()
