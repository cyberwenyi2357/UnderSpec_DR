#!/usr/bin/env python3
"""
Convert OpenScholar-style answers with ``<cite id="...">span</cite>`` tags into
the ``{input, output, ctxs}`` format expected by
``citation_correctness_eval.py --citations``.

Cite ids have the form ``{call_id}-{doc_index}`` and point at
``query_full_traces.tool_calls[*].documents[doc_index]`` of the tool call with
that ``call_id``. For each row:

- cited documents become ``ctxs`` (0-indexed, in order of first citation);
- each ``<cite id="a,b">span</cite>`` becomes ``span [i][j]``; when the span
  covers several sentences, the markers are appended to every sentence so the
  NLI eval attributes each sentence to the cited documents;
- ids that cannot be resolved (no such tool call / doc index out of range /
  no retrieval at all) are mapped to an out-of-range index, which the eval
  scores as unsupported (``--invalid drop`` removes them instead);
- pre-existing ``[n]`` strings in the answer are rewritten to ``(n)`` so they
  are not mistaken for citations.

Rows without ``query_full_traces`` (e.g. Tongyi DeepResearch outputs) have no
inline citations; they are written with empty ``ctxs`` and score 0.

Example::

    python scripts/convert_cite_tags_for_citation_eval.py \\
        --input data/scholarqa_cs/answer_openScholar_cs_query_qwen.jsonl \\
        --output data/scholarqa_cs/citation_eval/answer_openScholar_cs_query_qwen.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from nltk import sent_tokenize

CITE_RE = re.compile(r'<cite\s+id="([^"]*)"\s*>(.*?)</cite>', re.S)
# Anything the eval's ``\[\d+`` regex would read as a citation, e.g. [2], [1-3].
BRACKET_NUM_RE = re.compile(r"\[(\d[^\]\n]{0,50})\]")
STRAY_CITE_TAG_RE = re.compile(r"</?cite\b[^>]*>")
TRAILING_PUNCT_RE = re.compile(r"([\s.!?;:,]*)$")
# browse_webpage documents carry the full page; the agent only saw a truncated
# prefix in the tool output, so keep a comparable prefix for the NLI model.
MAX_DOC_WORDS = 512
# Retrieved documents as rendered to the agent inside <tool_output>.
TRACE_DOC_RE = re.compile(
    r'<(snippet|webpage)\s+id=["\']?([^\s>"\']+)["\']?\s*>(.*?)</\1>', re.S)
TRACE_FIELD_RE = re.compile(r"^\s*(Title|URL|Snippet):\s*(.*?)(?=^\s*(?:Title|URL|Snippet):|\Z)", re.S | re.M)


def _doc_to_ctx(doc: dict) -> dict:
    text = ""
    for key in ("snippet", "text", "summary"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
    words = text.split()
    if len(words) > MAX_DOC_WORDS:
        text = " ".join(words[:MAX_DOC_WORDS])
    ctx = {"title": (doc.get("title") or "Untitled").strip(), "text": text}
    for key in ("url", "id"):
        if doc.get(key):
            ctx[key] = doc[key]
    return ctx


def _trace_docs(row: dict) -> dict[str, dict]:
    """Documents shown to the agent in ``generated_text``, keyed by snippet id."""
    traces = row.get("query_full_traces") or {}
    docs = {}
    for _, doc_id, body in TRACE_DOC_RE.findall(traces.get("generated_text") or ""):
        fields = {k.lower(): v.strip() for k, v in TRACE_FIELD_RE.findall(body)}
        docs.setdefault(doc_id, {
            "title": fields.get("title"),
            "url": fields.get("url"),
            "snippet": fields.get("snippet") or body.strip(),
        })
    return docs


def _doc_lookup(row: dict) -> dict[str, dict]:
    # Some tool outputs (from malformed tool-call tags) only appear in
    # generated_text, so fall back to it for ids missing from tool_calls.
    lookup = _trace_docs(row)
    traces = row.get("query_full_traces") or {}
    for call in traces.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        for i, doc in enumerate(call.get("documents") or []):
            if isinstance(doc, dict):
                lookup[f"{call.get('call_id')}-{i}"] = doc
    return lookup


def _attach_markers(span: str, markers: str) -> str:
    """Append markers to every sentence of span, before trailing punctuation."""
    if not markers:
        return span
    out = []
    for sent in sent_tokenize(span) or [span]:
        punct = TRAILING_PUNCT_RE.search(sent).group(1)
        body = sent[: len(sent) - len(punct)] if punct else sent
        out.append(f"{body} {markers}{punct.rstrip()}")
    # Preserve leading/trailing whitespace of the original span.
    lead = span[: len(span) - len(span.lstrip())]
    trail = span[len(span.rstrip()):]
    return lead + " ".join(out) + trail


def convert_row(row: dict, invalid: str, stats: Counter) -> dict:
    answer = row.get("answer_text") or ""
    lookup = _doc_lookup(row)
    n_retrieved = len(_trace_docs(row))
    ctxs: list[dict] = []
    id2idx: dict[str, int] = {}
    INVALID = -1  # placeholder, resolved once len(ctxs) is known

    answer = BRACKET_NUM_RE.sub(lambda m: f"({m.group(1)})", answer)
    answer = re.sub(r"\[(?=\d)", "(", answer)

    def _replace(m: re.Match) -> str:
        ids = [x for x in re.split(r"[,\s]+", m.group(1).strip()) if x]
        idxs = []
        for cid in ids:
            stats["cite_ids"] += 1
            if cid in lookup:
                if cid not in id2idx:
                    id2idx[cid] = len(ctxs)
                    ctxs.append(_doc_to_ctx(lookup[cid]))
                idxs.append(id2idx[cid])
            else:
                stats["unresolved_ids"] += 1
                if invalid == "penalize":
                    idxs.append(INVALID)
        markers = "".join(f"[{i}]" if i != INVALID else "[__INVALID__]"
                          for i in dict.fromkeys(idxs))
        return _attach_markers(m.group(2), markers)

    output = CITE_RE.sub(_replace, answer)
    output = output.replace("[__INVALID__]", f"[{len(ctxs)}]")
    # Unclosed / unmatched tags (rare malformed generations) carry no span.
    stats["stray_cite_tags"] += len(STRAY_CITE_TAG_RE.findall(output))
    output = STRAY_CITE_TAG_RE.sub("", output)
    stats["cite_tags"] += len(CITE_RE.findall(answer))
    stats["rows_with_cites"] += int(bool(ctxs) or "<cite" in answer)

    return {
        "id": row.get("case_id") or row.get("id"),
        "input": row.get("query") or row.get("question") or "",
        "output": output,
        "ctxs": ctxs,
        "n_retrieved": n_retrieved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path,
                        help="Output .jsonl (only with a single --input)")
    parser.add_argument("--output-dir", type=Path,
                        help="Write each input to <output-dir>/<input name>")
    parser.add_argument("--invalid", choices=["penalize", "drop"], default="penalize",
                        help="How to handle cite ids that do not resolve to a retrieved document")
    args = parser.parse_args()
    if bool(args.output) == bool(args.output_dir):
        parser.error("pass exactly one of --output / --output-dir")
    if args.output and len(args.input) > 1:
        parser.error("--output only works with a single --input")

    for in_path in args.input:
        out_path = args.output or args.output_dir / in_path.name
        stats: Counter = Counter()
        with in_path.open(encoding="utf-8") as fh:
            rows = [convert_row(json.loads(line), args.invalid, stats) for line in fh if line.strip()]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{in_path.name}: rows={len(rows)} rows_with_cites={stats['rows_with_cites']} "
              f"cite_tags={stats['cite_tags']} cite_ids={stats['cite_ids']} "
              f"unresolved={stats['unresolved_ids']} stray_tags={stats['stray_cite_tags']} -> {out_path}")


if __name__ == "__main__":
    main()
