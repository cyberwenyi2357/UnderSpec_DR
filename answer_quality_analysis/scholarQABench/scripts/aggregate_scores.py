"""Aggregate Prometheus per-response scores into a single final score.

For every response in a ScholarQA-Multi variant, this averages the three
evaluation dimensions -- organization, coverage and relevance -- into one
final score (the mean of the available dimensions).

Score sources per variant:
  - outputs/prometheus_org_cov_<variant>/results.json  -> organization, coverage
  - outputs/prometheus_relevance_<variant>/results.json -> relevance
    (relevance is scored with a separate model, per the repo README, so it is
     read from the dedicated relevance run when available)

Results are keyed by the response's position in the evaluated input file, so
the org_cov and relevance runs of a variant must use the same input file.
With --input-template, positions are mapped back to the input's `id` field.

Usage:
    python scripts/aggregate_scores.py
    python scripts/aggregate_scores.py --outputs-dir outputs --out outputs/final_scores.json
    # folders named org_cov_<variant> / relevance_<variant>:
    python scripts/aggregate_scores.py --outputs-dir outputs/prometheus_csAndMulti_gpt5.4 --dir-prefix "" \
        --input-template data/scholarqa_csAndMulti/prometheus_openScholar_multi_{variant}_gpt5.4.json \
        --out outputs/prometheus_csAndMulti_gpt5.4/final_scores.json
"""
import argparse
import glob
import json
import os
import re


def extract_score(entry):
    """Return an int score in 1..5 from a result entry, or None if unpar.able.

    Handles clean values ("5"), values with trailing feedback, and cases where
    the score field accidentally holds the whole feedback text ("...[RESULT] 5").
    """
    raw = entry.get("score")
    if raw is None:
        return None
    s = str(raw).strip()

    if s in ("1", "2", "3", "4", "5"):
        return int(s)

    head = s.split("\n\n")[0].strip()
    if head in ("1", "2", "3", "4", "5"):
        return int(head)

    # Fall back to the last "[RESULT] N" marker in the score or feedback text.
    for text in (s, str(entry.get("feedback", ""))):
        matches = re.findall(r"\[RESULT\]\s*([1-5])", text)
        if matches:
            return int(matches[-1])

    m = re.match(r"([1-5])\b", s)
    if m:
        return int(m.group(1))
    return None


def load_dimension(results_path, prefix):
    """Load {response_id: score} for the criteria key starting with `prefix`."""
    if not os.path.exists(results_path):
        return {}
    with open(results_path) as f:
        data = json.load(f)
    key = next((k for k in data if k != "summary" and k.startswith(prefix)), None)
    if key is None:
        return {}
    scores = {}
    for resp_id, entry in data[key].items():
        val = extract_score(entry)
        if val is not None:
            scores[resp_id] = val
    return scores


def load_input_ids(input_template, variant):
    """Return the `id` of each response in the evaluated input file, by position."""
    with open(input_template.format(variant=variant)) as f:
        data = json.load(f)
    responses = data["data"] if isinstance(data, dict) and "data" in data else data
    return [str(r.get("id", i)) for i, r in enumerate(responses)]


def aggregate_variant(outputs_dir, variant, dir_prefix="prometheus_", input_ids=None):
    org_cov_path = os.path.join(outputs_dir, f"{dir_prefix}org_cov_{variant}", "results.json")
    rel_path = os.path.join(outputs_dir, f"{dir_prefix}relevance_{variant}", "results.json")

    organization = load_dimension(org_cov_path, "Organization")
    coverage = load_dimension(org_cov_path, "Coverage")
    # Prefer the dedicated relevance run; fall back to the copy in org_cov.
    relevance = load_dimension(rel_path, "Relevance") or load_dimension(org_cov_path, "Relevance")

    org_cov_ids, rel_ids = set(organization) | set(coverage), set(relevance)
    if org_cov_ids and rel_ids and org_cov_ids != rel_ids:
        print(f"WARNING [{variant}]: org_cov has {len(org_cov_ids)} responses, relevance has {len(rel_ids)}; "
              f"{len(org_cov_ids ^ rel_ids)} only scored in one of them.")

    all_ids = sorted(
        set(organization) | set(coverage) | set(relevance),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    per_response = {}
    for rid in all_ids:
        dims = {
            "organization": organization.get(rid),
            "coverage": coverage.get(rid),
            "relevance": relevance.get(rid),
        }
        present = [v for v in dims.values() if v is not None]
        final = round(sum(present) / len(present), 4) if present else None
        if input_ids is not None:
            rid = input_ids[int(rid)]
        per_response[rid] = {**dims, "final_score": final}

    finals = [r["final_score"] for r in per_response.values() if r["final_score"] is not None]
    variant_average = round(sum(finals) / len(finals), 4) if finals else None
    return per_response, variant_average


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", default="outputs", help="Directory holding the prometheus_* result folders.")
    parser.add_argument("--out", default="outputs/final_scores.json", help="Path to write the aggregated JSON.")
    parser.add_argument("--dir-prefix", default="prometheus_",
                        help='Prefix of the result folders, i.e. <prefix>org_cov_<variant>. Use "" for org_cov_<variant>.')
    parser.add_argument("--input-template", default=None,
                        help="Evaluated input file with a {variant} placeholder; maps response positions to their `id`.")
    args = parser.parse_args()

    org_cov_glob = f"{args.dir_prefix}org_cov_"
    variants = []
    for path in sorted(glob.glob(os.path.join(args.outputs_dir, org_cov_glob + "*"))):
        name = os.path.basename(path)
        variants.append(name[len(org_cov_glob):])

    report = {}
    for variant in variants:
        input_ids = load_input_ids(args.input_template, variant) if args.input_template else None
        per_response, variant_average = aggregate_variant(args.outputs_dir, variant, args.dir_prefix, input_ids)
        if input_ids is not None and len(per_response) != len(input_ids):
            print(f"WARNING [{variant}]: {len(per_response)} scored responses vs {len(input_ids)} in the input file.")
        report[variant] = {
            "num_responses": len(per_response),
            "variant_average_final_score": variant_average,
            "per_response": per_response,
        }
        print(f"{variant:>22}: {len(per_response):>3} responses | mean final score = {variant_average}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote aggregated scores to {args.out}")


if __name__ == "__main__":
    main()
