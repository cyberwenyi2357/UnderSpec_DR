"""Write data/query/researchqa_rubrics.json: the ResearchQA test-split items (with their rubrics) for the
500 questions in data/query/researchqa.jsonl, in the format compute_coverage.py reads.

Usage:
    python scoring/make_rubric_file.py --test_json /path/to/ResearchQA/test.json
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", required=True, help="ResearchQA test.json (download_researchqa.py --split test.json)")
    args = parser.parse_args()
    ids = {json.loads(line)["id"] for line in open(REPO / "data" / "query" / "researchqa.jsonl", encoding="utf-8")}
    items = [it for it in json.load(open(args.test_json, encoding="utf-8")) if it["id"] in ids]
    out = REPO / "data" / "query" / "researchqa_rubrics.json"
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {out} ({len(items)} questions, {sum(len(it['rubric']) for it in items)} rubric items)")


if __name__ == "__main__":
    main()
