"""Search query drift: how far a session's last search query moved away from its first one.

    drift = 1 - cos(emb(first search query), emb(last search query))

* Search calls: google_search / snippet_search (gpt-5.4, dr-tulu; browse_webpage excluded) and
  search / google_scholar (tongyi). Calls with an empty query (malformed, nothing searched) are skipped;
  how many were skipped is kept in ``n_empty_query_search_calls``.
* Embeddings: OpenAI text-embedding-3-small, cosine on unit-normalised vectors.
* tongyi sends a batch of queries per call. Best match: for each query of the last batch take its max cosine
  over the queries of the first batch, then average. With one query per side this is plain cosine.
* Sessions with fewer than 2 (non-empty) search calls get ``first_last_similarity = drift = null``
  (first and last would be the same call, a constant 1 similarity).

Adds to every question object: n_search_calls, n_empty_query_search_calls, first_search, last_search
({"tool", "query"} or null), first_last_similarity, query_drift (both rounded to 4 decimals).

Embeddings are cached in a JSONL file ({"query", "embedding"} per line) so reruns don't call the API again.

Usage:
    python -m trajectory_analysis.query_drift --model tongyi in.json [out.json] [--cache emb.jsonl]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import add_io_args, dump_json, has_query, load_json, openai_client, query_list, search_calls, tool_name, \
    tool_query

EMBED_MODEL = "text-embedding-3-small"
BATCH = 100


def first_last(obj):
    all_searches = search_calls(obj)
    searches = [tc for tc in all_searches if has_query(tool_query(tc))]
    pick = lambda tc: {"tool": tool_name(tc), "query": tool_query(tc)}
    return {
        "n_search_calls": len(searches),
        "n_empty_query_search_calls": len(all_searches) - len(searches),
        "first_search": pick(searches[0]) if searches else None,
        "last_search": pick(searches[-1]) if searches else None,
    }


class EmbeddingCache:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.emb = {}
        if self.path and self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        self.emb[d["query"]] = d["embedding"]

    def ensure(self, queries, client=None):
        missing = sorted(set(queries) - set(self.emb))
        if not missing:
            return
        client = client or openai_client()
        out = open(self.path, "a", encoding="utf-8") if self.path else None
        try:
            for i in range(0, len(missing), BATCH):
                batch = missing[i:i + BATCH]
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                for q, item in zip(batch, resp.data):
                    self.emb[q] = item.embedding
                    if out:
                        out.write(json.dumps({"query": q, "embedding": item.embedding}, ensure_ascii=False) + "\n")
                print(f"  embedded {min(i + BATCH, len(missing))}/{len(missing)}")
                time.sleep(0.1)
        finally:
            if out:
                out.close()

    def unit(self, q):
        v = np.asarray(self.emb[q], dtype=np.float64)
        return v / np.linalg.norm(v)


def similarity(first_q, last_q, cache):
    a = [cache.unit(q) for q in query_list(first_q) if q in cache.emb]
    b = [cache.unit(q) for q in query_list(last_q) if q in cache.emb]
    if not a or not b:
        return None
    return float((np.array(b) @ np.array(a).T).max(axis=1).mean())


def compute(data, cache, client=None):
    rows = [first_last(o) for o in data]
    needed = set()
    for r in rows:
        if r["n_search_calls"] >= 2:
            needed.update(query_list(r["first_search"]["query"]) + query_list(r["last_search"]["query"]))
    cache.ensure(needed, client)
    for obj, r in zip(data, rows):
        sim = similarity(r["first_search"]["query"], r["last_search"]["query"], cache) \
            if r["n_search_calls"] >= 2 else None
        obj.update(r)
        obj["first_last_similarity"] = round(sim, 4) if sim is not None else None
        obj["query_drift"] = round(1 - sim, 4) if sim is not None else None
    return data


def main():
    p = add_io_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--cache", default="query_embeddings.jsonl", help="embedding cache (JSONL)")
    a = p.parse_args()
    data = compute(load_json(a.input), EmbeddingCache(a.cache))
    dump_json(data, a.output or a.input)
    vals = [o["query_drift"] for o in data if o["query_drift"] is not None]
    print(f"{len(data)} objects, {len(vals)} with >=2 searches, mean drift "
          f"{np.mean(vals):.4f}" if vals else f"{len(data)} objects, none with >=2 searches")


if __name__ == "__main__":
    main()
