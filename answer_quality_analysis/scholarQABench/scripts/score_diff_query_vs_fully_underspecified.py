"""Compute per-question final_score differences: query minus fully_underspecified.

The ``per_response`` keys in ``outputs/final_scores.json`` are 0-based indices
into each variant's prediction file, not the original question numbers. Because
``fully_underspecified`` drops a few duplicate variants, index N in one variant
is not necessarily the same question as index N in the other. This script maps
each response back to its original question id (via the numeric suffix of the
prediction ``id``) and then subtracts scores question by question.

Usage:
    python scripts/score_diff_query_vs_fully_underspecified.py
"""
import argparse
import json
import re


def load_prediction_ids(path):
    """Return {result_index(str): original_question_number(int)}."""
    with open(path) as f:
        preds = json.load(f)
    mapping = {}
    for idx, row in enumerate(preds):
        m = re.search(r"(\d+)$", str(row.get("id", "")))
        if m:
            mapping[str(idx)] = int(m.group(1))
    return mapping


def index_by_original(per_response, index_to_original):
    """Return {original_question_number: response_dict}."""
    by_original = {}
    for result_idx, entry in per_response.items():
        original = index_to_original.get(result_idx)
        if original is not None:
            by_original[original] = entry
    return by_original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-scores", default="outputs/final_scores.json")
    parser.add_argument(
        "--query-pred",
        default="data/scholarqa_multi/openScholar_multi_predictions.json",
    )
    parser.add_argument(
        "--fully-underspecified-pred",
        default="data/scholarqa_multi/openScholar_multi_fully_underspecified_predictions.json",
    )
    parser.add_argument("--out", default="outputs/score_diff_query_vs_fully_underspecified.json")
    args = parser.parse_args()

    final_scores = json.load(open(args.final_scores))
    query = final_scores["query"]["per_response"]
    fully = final_scores["fully_underspecified"]["per_response"]

    query_by_orig = index_by_original(query, load_prediction_ids(args.query_pred))
    fully_by_orig = index_by_original(fully, load_prediction_ids(args.fully_underspecified_pred))

    fields = ("organization", "coverage", "relevance", "final_score")

    per_question = {}
    diffs = []
    for original in sorted(set(query_by_orig) & set(fully_by_orig)):
        q_entry = query_by_orig[original]
        f_entry = fully_by_orig[original]

        field_diffs = {}
        for field in fields:
            q_val = q_entry.get(field)
            f_val = f_entry.get(field)
            if q_val is None or f_val is None:
                field_diffs[field] = None
            else:
                field_diffs[field] = round(q_val - f_val, 4)

        per_question[f"original_{original}"] = {
            "query": {f: q_entry.get(f) for f in fields},
            "fully_underspecified": {f: f_entry.get(f) for f in fields},
            "diff": field_diffs,
        }
        if field_diffs["final_score"] is not None:
            diffs.append(field_diffs["final_score"])

    only_query = sorted(set(query_by_orig) - set(fully_by_orig))
    only_fully = sorted(set(fully_by_orig) - set(query_by_orig))

    report = {
        "description": "final_score and dimension differences (query - fully_underspecified), aligned by original question id",
        "num_compared": len(per_question),
        "mean_final_score_diff": round(sum(diffs) / len(diffs), 4) if diffs else None,
        "query_higher_count": sum(1 for d in diffs if d > 0),
        "fully_underspecified_higher_count": sum(1 for d in diffs if d < 0),
        "equal_count": sum(1 for d in diffs if d == 0),
        "only_in_query": only_query,
        "only_in_fully_underspecified": only_fully,
        "per_question": per_question,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Compared {len(per_question)} questions.")
    print(f"Mean final_score diff (query - fully_underspecified): {report['mean_final_score_diff']}")
    print(f"query higher: {report['query_higher_count']}, "
          f"fully_underspecified higher: {report['fully_underspecified_higher_count']}, "
          f"equal: {report['equal_count']}")
    if only_query:
        print(f"Only in query (no fully_underspecified counterpart): {only_query}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
