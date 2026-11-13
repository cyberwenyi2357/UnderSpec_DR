"""Search query provenance: where the tokens of each search query come from.

For every search call, the query's token set is traced in order, each step working on what is left over:
    1. tokens_in_user_query    tokens that appear in the user question
    2. tokens_in_prior_queries ... in any earlier query of the session
    3. tokens_in_prior_snippets / tokens_in_prior_responses ... in any earlier retrieved content
    4. untraced_tokens         the rest
``*_ratio`` = share of each class in the call's tokens (null when the query has no tokens); the object-level
``avg_*_ratio`` averages the non-null call ratios.

gpt-5.4 / dr-tulu (single-query schema; results on the tool call):
* All tool calls are traced in order, browse_webpage included (its query is empty so its ratios are null,
  but its documents' snippets still join the prior context).
* Prior context = snippets of the documents returned by earlier calls.
* Tokens: question and query [a-zA-Z]+, snippets [a-zA-Z0-9]+; lowercased, English stopwords removed,
  Porter-stemmed. Field: tokens_in_prior_snippets.

tongyi (batch schema; results in tool_call["arguments"]):
* Only search / google_scholar calls are traced; a call's query batch is tokenised as one set.
* Prior context = search result text of earlier calls plus the visit pages read after each of them
  (tool_response_tokens, also written out).
* Tokens: question [a-zA-Z]+ stemmed; queries and responses [a-z0-9]+ stemmed. Field: tokens_in_prior_responses.

--tokenizer jieba (used for DeepResearch Bench, all tasks, since half are Chinese) switches every tokenizer to
jieba mixed Chinese/English with Chinese + English stopwords. As in the original analysis:
* gpt-5.4 / dr-tulu stem pure-English jieba tokens; tongyi does not.
* tongyi's prior context is the search result text only (visit pages are not added).

Usage:
    python -m trajectory_analysis.query_provenance --model tongyi in.json [out.json] [--tokenizer jieba]
"""

import argparse

from .common import ALNUM_RE, ALPHA_RE, add_io_args, dump_json, is_batch_schema, load_json, mean, query_list, \
    question, stem, tokens_jieba, tokens_raw, tokens_stemmed, tongyi_search_steps

SINGLE_FIELDS = ["tokens_in_user_query", "tokens_in_prior_queries", "tokens_in_prior_snippets", "untraced_tokens"]
BATCH_FIELDS = ["tokens_in_user_query", "tokens_in_prior_queries", "tokens_in_prior_responses", "untraced_tokens"]


def tokenizers(model, tokenizer):
    """(question, query, context) tokenizers, each text -> set."""
    batch = is_batch_schema(model)
    if tokenizer == "jieba":
        f = lambda t: tokens_jieba(t, stem_english=not batch)
        return f, f, f
    if batch:  # context tokens are stemmed when merged into the prior context (see process_batch)
        return (lambda t: tokens_stemmed(t, ALPHA_RE)), (lambda t: {stem(w) for w in tokens_raw(t)}), tokens_raw
    return (lambda t: tokens_stemmed(t, ALPHA_RE)), (lambda t: tokens_stemmed(t, ALPHA_RE)), \
        (lambda t: tokens_stemmed(t, ALNUM_RE))


def trace(cur, user, prior_q, prior_ctx):
    a = cur & user
    r1 = cur - a
    b = r1 & prior_q
    r2 = r1 - b
    c = r2 & prior_ctx
    return [sorted(a), sorted(b), sorted(c), sorted(r2 - c)]


def write_trace(target, fields, parts):
    total = sum(len(p) for p in parts)
    for f, p in zip(fields, parts):
        target[f] = p
        target[f"{f}_ratio"] = len(p) / total if total else None


def process_single(obj, tok_q, tok_query, tok_ctx):
    user = tok_q(question(obj))
    obj["user_query_tokens"] = sorted(user)
    prior_q, prior_ctx = set(), set()
    for tc in obj.get("tool_calls", []):
        q = tc.get("query")
        cur = tok_query(q if isinstance(q, str) else None)
        tc["query_tokens"] = sorted(cur)
        write_trace(tc, SINGLE_FIELDS, trace(cur, user, prior_q, prior_ctx))
        prior_q |= cur
        for doc in tc.get("documents") or []:
            prior_ctx |= tok_ctx(doc.get("snippet"))
    for f in SINGLE_FIELDS:
        obj[f"avg_{f}_ratio"] = mean(tc.get(f"{f}_ratio") for tc in obj.get("tool_calls", []))


def process_batch(obj, tok_q, tok_query, tok_ctx, include_visits, ctx_stem):
    user = tok_q(question(obj))
    obj["user_query_tokens"] = sorted(user)
    prior_q, prior_ctx = set(), set()
    steps = tongyi_search_steps(obj)
    calls = [args for args, _ in steps]
    for args, visits in steps:
        texts = [args.get("tool_response")] + (visits if include_visits else [])
        ctx = set().union(*(tok_ctx(t) for t in texts))
        args["tool_response_tokens"] = sorted(ctx)  # stored unstemmed, as context_adoption uses them
        cur = set().union(*(tok_query(q) for q in query_list(args.get("query"))), set())
        write_trace(args, BATCH_FIELDS, trace(cur, user, prior_q, prior_ctx))
        prior_q |= cur
        prior_ctx |= {stem(t) for t in ctx} if ctx_stem else ctx
    for f in BATCH_FIELDS:
        obj[f"avg_{f}_ratio"] = mean(args.get(f"{f}_ratio") for args in calls)


def compute(data, model, tokenizer="en"):
    toks = tokenizers(model, tokenizer)
    for obj in data:
        if is_batch_schema(model):
            process_batch(obj, *toks, include_visits=tokenizer != "jieba", ctx_stem=tokenizer != "jieba")
        else:
            process_single(obj, *toks)
    return data


def main():
    p = add_io_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    p.add_argument("--tokenizer", choices=["en", "jieba"], default="en",
                   help="jieba for DeepResearch Bench (Chinese + English), en otherwise")
    a = p.parse_args()
    data = compute(load_json(a.input), a.model, a.tokenizer)
    dump_json(data, a.output or a.input)
    fields = BATCH_FIELDS if is_batch_schema(a.model) else SINGLE_FIELDS
    print(f"{len(data)} objects; mean of avg ratios: " +
          ", ".join(f"{f}={m:.3f}" for f in fields if (m := mean(o[f'avg_{f}_ratio'] for o in data)) is not None))


if __name__ == "__main__":
    main()
