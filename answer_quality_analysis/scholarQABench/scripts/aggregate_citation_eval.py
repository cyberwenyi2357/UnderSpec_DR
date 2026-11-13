"""Aggregate ScholarQA-CS citation_eval results for all openScholar_cs query variants into one JSON file.

Reads, per (variant, model), the per-sample files written by
`citation_correctness_eval.py --citations [--cited_sents_only]`:
  - <eval_dir>/answer_openScholar_cs_<variant>_<model>.jsonl.per_sample.jsonl            (mode=post_fix)
  - <eval_dir>/answer_openScholar_cs_<variant>_<model>.jsonl.cited_only.per_sample.jsonl (mode=cited_only)

Also writes every per-sample row (all models, variants and modes) to one JSONL file
(--per_sample_out), tagged with model/variant/mode and per-sample rec/prec/f1.

Summary output is nested as result[model][variant][mode][subset]:
  - subset=all:             every answered sample (empty outputs are skipped by the eval script);
                            reproduces the .score_post_fix / .score_cited_only numbers
  - subset=excl_clarifying: additionally drops samples the ambiguity judge labeled
                            `asked_clarifying_question`

Scores (rec, prec, f1) are in [0, 1]; citation_f1 is the harmonic mean of the averaged
citation_rec and citation_prec (as in OpenScholar).

`paired_with_query` compares each variant against the full `query` variant on the shared
question ids (same model, mode and subset), since without_* variants cover subsets of questions.

Usage:
    python scripts/aggregate_citation_eval.py
    python scripts/aggregate_citation_eval.py --out data/scholarqa_cs/citation_eval/citation_eval_summary.json
"""
import argparse
import collections
import glob
import json
import os
import re

VARIANTS = ["query", "fully_underspecified", "without_preference", "without_scope", "without_term"]
MODES = {"post_fix": "", "cited_only": ".cited_only"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_clarifying(path):
    clar = collections.defaultdict(set)
    if not os.path.exists(path):
        return clar
    for r in load_jsonl(path):
        if r.get("category") == "asked_clarifying_question":
            clar[(r["model"], r["setting"])].add(r["id"])
    return clar


def sample_scores(row):
    rec = row["n_supported_sents"] / row["n_scored_sents"] if row["n_scored_sents"] else 0
    prec = row["n_valid_citations"] / row["n_total_citations"] if row["n_total_citations"] else 0
    return rec, prec


def count_cited_papers(text):
    """Distinct cited ids, matching extract_citations() in citation_correctness_eval.py."""
    ids = set()
    for match in re.findall(r"\[(\d+(?:,\s*\d+)*)\]", text):
        ids.update(int(n.strip()) for n in match.split(","))
    return len(ids)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def f1(rec, prec):
    """Citation F1 as reported in OpenScholar: harmonic mean of the averaged rec and prec."""
    if rec is None or prec is None:
        return None
    return 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0


def fmt(x):
    return None if x is None else round(x, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="data/scholarqa_cs/citation_eval")
    parser.add_argument("--ambiguity_file",
                        default="results/ambiguity_judge_scholarqa_cs/merged_ambiguity_judge_scholarqa_cs.jsonl")
    parser.add_argument("--out", default="data/scholarqa_cs/citation_eval/citation_eval_summary.json")
    parser.add_argument("--per_sample_out", default="data/scholarqa_cs/citation_eval/citation_eval_per_sample.jsonl")
    args = parser.parse_args()

    clar = load_clarifying(args.ambiguity_file)

    models = sorted({
        re.match(r"answer_openScholar_cs_(?:%s)_(.+)\.jsonl\.per_sample\.jsonl$" % "|".join(VARIANTS),
                 os.path.basename(p)).group(1)
        for p in glob.glob(os.path.join(args.eval_dir, "answer_openScholar_cs_*.jsonl.per_sample.jsonl"))
    })

    # (model, variant, mode, subset) -> {id: (rec, prec)}
    per_id = {}
    result = {}
    per_sample_rows = []
    for model in models:
        for variant in VARIANTS:
            base = os.path.join(args.eval_dir, f"answer_openScholar_cs_{variant}_{model}.jsonl")
            for mode, suffix in MODES.items():
                path = f"{base}{suffix}.per_sample.jsonl"
                if not os.path.exists(path):
                    continue
                samples = load_jsonl(path)
                clar_ids = clar[(model, variant)]
                answered = [s for s in samples if not s["skipped"]]
                answers = {a.get("id"): a for a in load_jsonl(base)} if os.path.exists(base) else {}
                for s in samples:
                    rec, prec = (None, None) if s["skipped"] else sample_scores(s)
                    per_sample_rows.append({
                        "id": s["id"],
                        "model": model,
                        "variant": variant,
                        "mode": mode,
                        "input": answers.get(s["id"], {}).get("input"),
                        "asked_clarifying_question": s["id"] in clar_ids,
                        **s,
                        "citation_rec": fmt(rec),
                        "citation_prec": fmt(prec),
                        "citation_f1": fmt(f1(rec, prec)),
                    })
                for subset in ["all", "excl_clarifying"]:
                    kept = answered if subset == "all" else [s for s in answered if s["id"] not in clar_ids]
                    scores = {s["id"]: sample_scores(s) for s in kept}
                    per_id[(model, variant, mode, subset)] = scores
                    rec = mean([r for r, _ in scores.values()])
                    prec = mean([p for _, p in scores.values()])
                    n_papers = [count_cited_papers(answers[s["id"]]["output"]) for s in kept if s["id"] in answers]
                    result.setdefault(model, {}).setdefault(variant, {}).setdefault(mode, {})[subset] = {
                        "n_questions": len(samples),
                        "n_empty_output": len(samples) - len(answered),
                        "n_clarifying": len(clar_ids & {s["id"] for s in answered}),
                        "n_scored": len(kept),
                        "citation_rec": fmt(rec),
                        "citation_prec": fmt(prec),
                        "citation_f1": fmt(f1(rec, prec)),
                        "avg_cited_papers": fmt(sum(n_papers) / len(n_papers)) if n_papers else None,
                        "avg_scored_sents": fmt(sum(s["n_scored_sents"] for s in kept) / len(kept)) if kept else None,
                        "paired_with_query": None,
                    }

    # Paired comparison against the `query` variant on shared ids.
    for (model, variant, mode, subset), cur in per_id.items():
        ref = per_id.get((model, "query", mode, subset))
        if variant == "query" or ref is None:
            continue
        ids = sorted(cur.keys() & ref.keys())
        if not ids:
            continue
        q_rec, v_rec = mean([ref[i][0] for i in ids]), mean([cur[i][0] for i in ids])
        q_prec, v_prec = mean([ref[i][1] for i in ids]), mean([cur[i][1] for i in ids])
        result[model][variant][mode][subset]["paired_with_query"] = {
            "n_paired": len(ids),
            "query_rec": fmt(q_rec),
            "variant_rec": fmt(v_rec),
            "query_prec": fmt(q_prec),
            "variant_prec": fmt(v_prec),
            "query_f1": fmt(f1(q_rec, q_prec)),
            "variant_f1": fmt(f1(v_rec, v_prec)),
            "rec_better": sum(cur[i][0] > ref[i][0] for i in ids),
            "rec_worse": sum(cur[i][0] < ref[i][0] for i in ids),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    with open(args.per_sample_out, "w") as f:
        for row in per_sample_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(per_sample_rows)} per-sample rows to {args.per_sample_out}")
    print(f"Wrote citation_eval summary for models {models} to {args.out}")


if __name__ == "__main__":
    main()
