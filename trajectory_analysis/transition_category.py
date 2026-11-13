"""Search query transition category: how each search step changes the previous one, judged by gpt-5.4-mini.

Categories: Specialization (more specific), Generalization (more general), Exploration (another aspect of the
same topic), Repetition (paraphrase, same intent).

* Pairs are consecutive search calls of a session (google_search / snippet_search for gpt-5.4 and dr-tulu,
  browse_webpage excluded; search / google_scholar for tongyi).
* gpt-5.4 / dr-tulu: one label per (previous query -> next query) pair, SINGLE_TEMPLATE.
* tongyi: one label per (previous call -> next call) pair, where each side is the whole batch of queries
  issued in one call, BATCH_TEMPLATE (same text plus a note that each side is a batch).
* Pairs where either side has no query are not sent; their category is null. Unparseable replies are null.
* temperature 0; the first category name found in the reply is used.

Adds to every question object:
    consecutive_query_pairs: [{"pair_index", "query_1", "query_2", "search_transition_category"}, ...]
    search_transition_category_counts / search_transition_category_ratios (null categories excluded)

Usage:
    python -m trajectory_analysis.transition_category --model dr-tulu in.json [out.json] [--workers 8]
"""

import argparse
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import add_io_args, dump_json, has_query, is_batch_schema, load_json, openai_client, search_calls, \
    tool_query

JUDGE_MODEL = "gpt-5.4-mini"
CATEGORIES = ["Specialization", "Generalization", "Exploration", "Repetition"]
SYSTEM_PROMPT = "You are an expert search behavior analyst."

CATEGORY_BLOCK = """Analyze the search behavior evolution from Query 1 to Query 2 for an autonomous agent.
Classify the transition into exactly ONE of these four categories:

1. Specialization (Vertical Deepening): Query 2 is MORE specific than Query 1 by adding constraints/details (q2 \\subset q1). (e.g., 'apple' -> 'green apple nutritional value').
2. Generalization (Vertical Broadening): Query 2 is MORE general than Query 1 by removing constraints/abstracting (q2 \\supset q1). (e.g., 'green apple nutritional value' -> 'benefits of fruits').
3. Exploration (Horizontal Expansion within the same domain/task): Query 2 is NOT simply more specific or more general. It shifts to a different aspect / subtopic / related entity but still within the same overall topic/domain. (e.g., 'green apple nutritional value' -> 'green apple recipes', or 'MRI Scans' -> 'CT Scans').
4. Repetition (Stationary): Query 2 is semantically equivalent to Query 1. It is a paraphrase, reformatting, or synonym replacement with NO significant change in intent (e.g., 'green apple value' -> 'nutritional value of green apple').

Output ONLY the category name (Specialization, Generalization, Exploration, or Repetition)."""

SINGLE_TEMPLATE = "Query 1 (Previous): {q1}\nQuery 2 (Current): {q2}\n\n" + CATEGORY_BLOCK
BATCH_TEMPLATE = """Query 1 (Previous): {q1}
Query 2 (Current): {q2}

Note: each of Query 1 and Query 2 may contain more than one query string — these are
the batch of queries the agent issued together in a single search call. Treat each
side as a whole batch representing that step's overall search intent, not as
individual isolated queries.

""" + CATEGORY_BLOCK


def consecutive_pairs(obj):
    queries = [tool_query(tc) for tc in search_calls(obj)]
    return [{"pair_index": i + 1, "query_1": queries[i], "query_2": queries[i + 1]} for i in range(len(queries) - 1)]


def parse_category(text):
    low = (text or "").lower()
    for cat in CATEGORIES:
        if cat.lower() in low:
            return cat
    return None


def classify(client, q1, q2, batch, retries=3):
    prompt = (BATCH_TEMPLATE if batch else SINGLE_TEMPLATE).format(q1=q1, q2=q2)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
            return parse_category(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            print(f"  [retry {attempt + 1}] {e}")
            time.sleep(1.5 * (attempt + 1))
    return None


def category_distribution(pairs):
    counts = Counter(p["search_transition_category"] for p in pairs if p.get("search_transition_category") in CATEGORIES)
    total = sum(counts.values())
    return dict(counts), ({c: n / total for c, n in counts.items()} if total else {})


def compute(data, model, client=None, workers=8):
    batch = is_batch_schema(model)
    client = client or openai_client()
    jobs = []
    for oi, obj in enumerate(data):
        obj["consecutive_query_pairs"] = consecutive_pairs(obj)
        for pi, pair in enumerate(obj["consecutive_query_pairs"]):
            pair["search_transition_category"] = None
            if has_query(pair["query_1"]) and has_query(pair["query_2"]):
                jobs.append((oi, pi, pair["query_1"], pair["query_2"]))
    print(f"{len(jobs)} pairs to classify")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(classify, client, q1, q2, batch): (oi, pi) for oi, pi, q1, q2 in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            oi, pi = futs[fut]
            data[oi]["consecutive_query_pairs"][pi]["search_transition_category"] = fut.result()
            if n % 100 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}")
    for obj in data:
        obj["search_transition_category_counts"], obj["search_transition_category_ratios"] = \
            category_distribution(obj["consecutive_query_pairs"])
    return data


def main():
    p = add_io_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    data = compute(load_json(a.input), a.model, workers=a.workers)
    dump_json(data, a.output or a.input)
    counts, ratios = category_distribution([pr for o in data for pr in o["consecutive_query_pairs"]])
    print({c: f"{counts.get(c, 0)} ({ratios.get(c, 0):.1%})" for c in CATEGORIES})


if __name__ == "__main__":
    main()
