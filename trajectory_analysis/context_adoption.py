"""Context-driven term adoption rate: how many of the words a search query adds come from retrieved content.

For each search call, in order:
    new_words             = tokens(current query) - tokens(previous search query)   (all tokens on the first)
    context_adopted_words = new_words that appear in content retrieved by any earlier call
    context_adoption_rate = |context_adopted_words| / |new_words|                    (null if no new words)

gpt-5.4 / dr-tulu (single-query schema; results on the tool call):
* Search calls are google_search / snippet_search with a non-empty query.
* Retrieved content = documents' title + snippet and browse_webpage ``output`` of every earlier call
  (browse calls are not scored but feed the context).

tongyi (batch schema; results in tool_call["arguments"]):
* Search calls are search / google_scholar; a query batch is tokenised as one set.
* Retrieved content = search result text of earlier calls plus the visit pages read after each of them.

Tokens: [a-z0-9]+ on lowercased text, English stopwords removed, no stemming. --tokenizer jieba (DeepResearch
Bench) uses jieba mixed Chinese/English instead; as in the original analysis gpt-5.4 / dr-tulu then stem
pure-English tokens and tongyi does not, and tongyi's context is the search result text only.

Object level:
* avg_context_adoption_rate: mean over the session's non-null call rates.
* avg_context_adoption_rate_excl_first: the same without the first scored call, whose rate is always 0 (nothing
  retrieved yet); this is what the query-vs-variant comparison used.

Usage:
    python -m trajectory_analysis.context_adoption --model dr-tulu in.json [out.json] [--tokenizer jieba]
"""

import argparse

from .common import SEARCH_TOOLS, add_io_args, dump_json, is_batch_schema, load_json, mean, query_list, \
    tokens_jieba, tokens_raw, tongyi_search_steps


def set_rate(target, cur, prev, prior):
    new = cur - prev
    adopted = new & prior
    target["new_words"], target["context_adopted_words"] = sorted(new), sorted(adopted)
    target["context_adoption_rate"] = len(adopted) / len(new) if new else None
    return target["context_adoption_rate"]


def process_single(obj, model, tok):
    prior, prev, rates = set(), set(), []
    for tc in obj.get("tool_calls", []):
        q = tc.get("query")
        if tc.get("tool_name") in SEARCH_TOOLS[model] and q:
            cur = tok(q) if isinstance(q, str) else set()
            rates.append(set_rate(tc, cur, prev, prior))
            prev = cur
        for doc in tc.get("documents") or []:
            prior |= tok(doc.get("title")) | tok(doc.get("snippet"))
        prior |= tok(tc.get("output"))
    return rates


def process_batch(obj, tok, include_visits):
    prior, prev, rates = set(), set(), []
    for args, visits in tongyi_search_steps(obj):
        cur = set().union(*(tok(q) for q in query_list(args.get("query"))), set())
        rates.append(set_rate(args, cur, prev, prior))
        prev = cur
        for text in [args.get("tool_response")] + (visits if include_visits else []):
            prior |= tok(text)
    return rates


def compute(data, model, tokenizer="en"):
    batch = is_batch_schema(model)
    tok = (lambda t: tokens_jieba(t, stem_english=not batch)) if tokenizer == "jieba" else tokens_raw
    for obj in data:
        rates = process_batch(obj, tok, tokenizer != "jieba") if batch else process_single(obj, model, tok)
        rates = [r for r in rates if r is not None]
        obj["avg_context_adoption_rate"] = mean(rates)
        obj["avg_context_adoption_rate_excl_first"] = mean(rates[1:])
    return data


def main():
    p = add_io_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--tokenizer", choices=["en", "jieba"], default="en",
                   help="jieba for DeepResearch Bench (Chinese + English), en otherwise")
    a = p.parse_args()
    data = compute(load_json(a.input), a.model, a.tokenizer)
    dump_json(data, a.output or a.input)
    m = mean(o["avg_context_adoption_rate_excl_first"] for o in data)
    print(f"{len(data)} objects, mean avg_context_adoption_rate_excl_first = {m:.4f}" if m is not None
          else f"{len(data)} objects, no session with 2+ scored searches")


if __name__ == "__main__":
    main()
