"""Collect ResearchQA queries, query variants, deep research agent answers and ambiguity-judge results
from the ResearchQA working repo into data/.

Usage:
    python build.py --src /path/to/ResearchQA
"""

import argparse
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"

VARIANTS = ["query", "fully_underspecified", "without_scope", "without_term", "without_preference"]
# output name -> suffix used in the ResearchQA file names
MODELS = {"gpt-5.4": "5.4", "dr-tulu": "qwen", "tongyi": "tongyi"}

# variant -> [(file, field holding the variant text)]; ids are unique across the two samples.
QUERY_FILES = {
    "query": [("researchQA_sample450_query.jsonl", "query"), ("researchQA_sample50_query.jsonl", "question")],
    "fully_underspecified": [("researchQA_sample450_fully_underspecified.jsonl", "query_fully_underspecified"),
                             ("researchQA_sample50_fully_underspecified.jsonl", "question")],
    "without_scope": [("researchQA_sample450_without_scope.jsonl", "query_without_scope_limitation"),
                      ("researchQA_sample50_without_scope.jsonl", "question")],
    "without_term": [("researchQA_sample450_without_term.jsonl", "query_without_term"),
                     ("researchQA_sample50_without_term.jsonl", "question")],
    "without_preference": [("researchQA_sample450_without_preference.jsonl", "query_without_preference")],
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="path to the ResearchQA working repo")
    src = Path(parser.parse_args().src)

    # Queries: one row per item with the original query and every variant it has (null if none).
    texts = {v: {} for v in VARIANTS}
    for variant, files in QUERY_FILES.items():
        for name, field in files:
            for r in read_jsonl(src / name):
                texts[variant][r["id"]] = r[field]
    meta = {it["id"]: it for it in json.loads((src / "test.json").read_text(encoding="utf-8"))}
    ids = sorted(texts["query"])
    queries = [{"id": i, "general_domain": meta[i]["general_domain"], "subdomain": meta[i]["subdomain"],
                "field": meta[i]["field"], **{v: texts[v].get(i) for v in VARIANTS}} for i in ids]
    write_jsonl(OUT / "queries.jsonl", queries)

    # Answers and judge results, one file per model x variant, plus a combined long-format file.
    combined = []
    stats = Counter()
    for model, suffix in MODELS.items():
        for variant in VARIANTS:
            answers = json.loads((src / f"answer_researchQA_sample500_{variant}_{suffix}_extracted.json")
                                 .read_text(encoding="utf-8"))
            write_jsonl(OUT / "answers" / model / f"{variant}.jsonl",
                        [{"id": i, "answer": a["answer"]} for i, a in sorted(answers.items())])
            judged = {r["id"]: r for r in read_jsonl(src / "results" / "ambiguity_judge" / f"{variant}_{suffix}.jsonl")}
            write_jsonl(OUT / "ambiguity_judge" / model / f"{variant}.jsonl",
                        [{"id": i, "category": r["category"], "evidence": r["evidence"], "error": r.get("error")}
                         for i, r in sorted(judged.items())])
            for i, a in sorted(answers.items()):
                j = judged.get(i, {})
                combined.append({"id": i, "model": model, "variant": variant, "query": texts[variant].get(i),
                                 "original_query": texts["query"].get(i), "answer": a["answer"],
                                 "ambiguity_category": j.get("category"), "ambiguity_evidence": j.get("evidence")})
                stats[(model, variant, "judged" if j.get("category") else "not judged")] += 1
    write_jsonl(OUT / "combined.jsonl", combined)

    print(f"queries: {len(queries)}")
    for v in VARIANTS:
        print(f"  {v:22s} {sum(q[v] is not None for q in queries)}")
    for (model, variant, status), n in sorted(stats.items()):
        print(f"{model:8s} {variant:22s} {status:10s} {n}")
    print(f"combined rows: {len(combined)}")


if __name__ == "__main__":
    main()
