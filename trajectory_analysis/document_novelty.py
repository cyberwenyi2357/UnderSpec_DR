"""Document novelty: what share of the documents a search returns were not seen earlier in the session.

For each search call (google_search / snippet_search for gpt-5.4 and dr-tulu, browse_webpage excluded;
search / google_scholar for tongyi), in order:
    new_documents_count = returned URLs not returned by any earlier search call of the same session
    new_documents_ratio = new_documents_count / number of returned documents
Documents are identified by URL (document ids are regenerated per request). Calls that returned nothing get
no value. gpt-5.4 / dr-tulu read URLs from ``documents[*].url``; tongyi from ``arguments.urls``, and its
results are written into ``arguments``.

Object level:
* avg_new_documents_ratio: mean over the session's search calls.
* avg_new_documents_ratio_excl_first: the same without the first call, whose ratio is always 1; this is what
  the query-vs-variant comparison used, so short sessions are not pushed up. Null with fewer than 2 calls.
* max_low_novelty_streak / has_sustained_low_novelty: longest run of consecutive searches with ratio < 0.5,
  and whether it reaches 2.

Usage:
    python -m trajectory_analysis.document_novelty --model gpt-5.4 in.json [out.json]
"""

import argparse

from .common import add_io_args, dump_json, is_batch_schema, load_json, mean, search_calls, tongyi_args

LOW_THRESHOLD = 0.5
MIN_STREAK = 2


def call_urls(tc, batch):
    """(urls, number of returned documents) of a search call."""
    if batch:
        urls = tongyi_args(tc).get("urls") or []
        return urls, len(urls)
    docs = tc.get("documents") or []
    return [d.get("url") for d in docs if d.get("url")], len(docs)


def max_streak(ratios, threshold=LOW_THRESHOLD):
    best = cur = 0
    for r in ratios:
        cur = cur + 1 if r < threshold else 0
        best = max(best, cur)
    return best


def process(obj, batch):
    seen, ratios = set(), []
    for tc in search_calls(obj):
        target = tongyi_args(tc) if batch else tc
        urls, n_docs = call_urls(tc, batch)
        if not n_docs:
            target.pop("new_documents_count", None)
            target.pop("new_documents_ratio", None)
            continue
        new = [u for u in urls if u not in seen]
        target["new_documents_count"] = len(new)
        target["new_documents_ratio"] = len(new) / n_docs
        ratios.append(target["new_documents_ratio"])
        seen.update(urls)
    obj["avg_new_documents_ratio"] = mean(ratios)
    obj["avg_new_documents_ratio_excl_first"] = mean(ratios[1:])
    obj["max_low_novelty_streak"] = max_streak(ratios)
    obj["has_sustained_low_novelty"] = obj["max_low_novelty_streak"] >= MIN_STREAK


def compute(data, model):
    batch = is_batch_schema(model)
    for obj in data:
        process(obj, batch)
    return data


def main():
    a = add_io_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0])).parse_args()
    data = compute(load_json(a.input), a.model)
    dump_json(data, a.output or a.input)
    m = mean(o["avg_new_documents_ratio_excl_first"] for o in data)
    print(f"{len(data)} objects, mean avg_new_documents_ratio_excl_first = {m:.4f}" if m is not None
          else f"{len(data)} objects, no session with 2+ searches")


if __name__ == "__main__":
    main()
