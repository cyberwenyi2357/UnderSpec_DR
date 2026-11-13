#!/usr/bin/env python3
"""
Read a JSONL file (one object per line) or a JSON array file, send each row's
configured text field to GPT via the OpenAI Batch API, and write three
query-analysis/rewrite results as JSONL.

Requires OpenAI credentials in the environment (e.g. ``agent/.env``):

    OPENAI_API_KEY=...

Runs in two batch rounds:
  1. Analysis batch: for every selected row, ask whether the query has a
     preference / term / scope limitation.
  2. Rewrite batch: for every row, submit only the rewrite tasks whose
     analysis flag was "yes" (``query_fully_underspecified`` is always
     requested).

Edit **``SYSTEM_PROMPT``** / ``ANALYSIS_TASKS`` in this file. Optional CLI
``--system`` overrides **``SYSTEM_PROMPT``** for a single run.

By default, only a fixed set of **1-based line numbers** in the input JSONL is
processed (see ``_SYNTHETIC_DATA_SUBSET_1BASED``). Pass ``--all-lines`` to run
on every row (e.g. other input files).

Example::

    python scripts/query_rewrite.py \\
    --input scripts/researchQA_sample50.json \\
    --input-field query \\
    --output researchQA_sample50_underspecified.jsonl \\
    --all-lines

Resume after interruption (skip ``id`` values already present in ``--output``)::

    python scripts/query_rewrite.py -i synthetic_data.jsonl -o out.jsonl --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_AGENT_DIR = Path(__file__).resolve().parent.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from dr_agent.openai_batch import (  # noqa: E402
    download_batch_results,
    extract_chat_completion_body,
    poll_batch_until_done,
    submit_chat_completions_batch,
)

_OUTPUT_FIELDS = (
    "id",
    "query",
    "has_preference",
    "has_term",
    "has_scope_limitation",
    "query_without_preference",
    "query_without_term",
    "query_without_scope_limitation",
    "query_fully_underspecified",
)

# ---------------------------------------------------------------------------
# Edit here: system instructions for processing each row's query text.
# If this string is empty (after strip), only the user message is sent.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You analyze research queries.

Return exactly one JSON object and nothing else.
The JSON object must have these keys:
- "has_preference"
- "has_term"
- "has_scope_limitation"

For each key, the value must be exactly "yes" or "no".
"""

ANALYSIS_INSTRUCTION = """Analyze the query and answer these three questions:

1. "has_preference": Does the query explicitly contain user preference information, such as desired format, depth, target audience, or evaluation criteria?
2. "has_term": Does the query contain any proper nouns, such as named systems, algorithms, frameworks, organizations, or specific named entities?
3. "has_scope_limitation": Does the query explicitly restrict the problem to a specific scenario, domain, time period, or location?
"""

REWRITE_SYSTEM_PROMPT = """You need to rewrite user queries.

You will be given:
1. The original query
2. One category of information to simplify

Return only the rewritten query. Do not explain the changes.
"""

REWRITE_TASKS = (
    (
        "has_preference",
        "query_without_preference",
        "Remove any user preference information in the original query(e.g. format, depth, audience, criteria).",
    ),
    (
        "has_term",
        "query_without_term",
        "Simplify any proper nouns (e.g. named technology categories, algorithms, frameworks, organizations) to a brief generic mention. If the query contains the abbreviation of a proper noun with the full name or explanation, just keep the abbreviation and remove the full name/explanation. If the query only has a proper noun without any explanation, replace it with a generic description.",
    ),
    (
        "has_scope_limitation",
        "query_without_scope_limitation",
        "Remove any explicit scope limitations (e.g. scenario, domain, time period, location).",
    ),
    (
        "has_any_underspecification",
        "query_fully_underspecified",
        "Remove all the important details in the user query, such as user preference information (e.g. format, depth, audience, criteria), proper nouns (e.g. named technology categories, algorithms, frameworks, organizations), and scope limitations (e.g. scenario, domain, time period, location). Replace named entities with generic descriptions where needed. If a proper noun appears with both an abbreviation and its full name, keep only the abbreviation and remove the full name.",
    ),
)

# 1-based line index in the input JSONL (第 n 行 = 第 n 个 object). Only these
# lines are sent to GPT unless you pass ``--all-lines`` on the CLI.
_SYNTHETIC_DATA_SUBSET_1BASED = frozenset(
    {2, 8, 9, 17, 18, 22, 29, 31, 34, 37, 42}
    | set(range(54, 77))  # 54–76
    | set(range(78, 88))  # 78–87
    | set(range(89, 100))  # 89–99
)


def _load_dotenv() -> None:
    load_dotenv(_AGENT_DIR / ".env")
    load_dotenv(_AGENT_DIR.parent / ".env")


def _resolve_input_path(path_s: str) -> Path:
    path = Path(path_s)
    if path.is_absolute():
        return path
    if path.is_file():
        return path
    return _AGENT_DIR / path


def _resolve_output_path(path_s: str) -> Path:
    path = Path(path_s)
    if path.is_absolute():
        return path
    # Outputs are usually meant relative to the caller's current directory.
    return path


def _existing_ids(path: Path) -> set:
    if not path.is_file():
        return set()
    seen: set = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "id" in obj:
                seen.add(obj["id"])
    return seen


def _output_row(
    obj: dict,
    responses: dict[str, str | None] | None = None,
    gpt_error: str | None = None,
    query_text: str | None = None,
) -> dict:
    responses = responses or {}
    row = {
        "id": obj.get("id") or obj.get("prompt_id"),
        "query": query_text
        if query_text is not None
        else (obj.get("query") or _get_field_value(obj, "prompt.content")),
        "has_preference": responses.get("has_preference"),
        "has_term": responses.get("has_term"),
        "has_scope_limitation": responses.get("has_scope_limitation"),
        "query_without_preference": responses.get("query_without_preference"),
        "query_without_term": responses.get("query_without_term"),
        "query_without_scope_limitation": responses.get("query_without_scope_limitation"),
        "query_fully_underspecified": responses.get("query_fully_underspecified"),
    }
    if gpt_error is not None:
        row["gpt_error"] = gpt_error
    return row


def _get_field_value(obj: dict, field: str) -> str | None:
    """Extract a field value, supporting dot-notation for nested dicts/lists.

    E.g. ``prompt.content`` on ``{"prompt": [{"role": "user", "content": "..."}]}``
    concatenates all ``content`` values from the list with newlines.
    """
    if "." not in field:
        val = obj.get(field)
    else:
        key, subkey = field.split(".", 1)
        val = obj.get(key)
        if isinstance(val, list):
            parts = [item.get(subkey, "") for item in val if isinstance(item, dict)]
            return "\n".join(p for p in parts if p) or None
        if isinstance(val, dict):
            return val.get(subkey)
        return None

    if isinstance(val, list):
        parts = [
            item.get("content", "") if isinstance(item, dict) else str(item)
            for item in val
        ]
        return "\n".join(p for p in parts if p) or None
    return val


def _iter_input_records(path: Path):
    """Yield ``(record_number_1based, object)`` from JSONL or JSON array input."""
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                yield line_no, json.loads(line)
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array or JSONL file")
    for i, obj in enumerate(data, start=1):
        yield i, obj


def _parse_analysis_json(content: str) -> dict[str, str | None]:
    raw = (content or "").strip()
    empty = {
        "has_preference": None,
        "has_term": None,
        "has_scope_limitation": None,
    }
    if not raw:
        return empty
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {**empty, "has_preference": raw}
    if not isinstance(obj, dict):
        return {**empty, "has_preference": str(obj)}
    return {
        "has_preference": obj.get("has_preference"),
        "has_term": obj.get("has_term"),
        "has_scope_limitation": obj.get("has_scope_limitation"),
    }


def _build_user_text(base_query: str, instruction: str, user_prefix: str) -> str:
    return (
        f"{instruction}\n\n"
        f"{user_prefix}"
        f"Query:\n{base_query.strip()}"
    )


def _build_rewrite_user_text(base_query: str, instruction: str) -> str:
    return f"{instruction}\n\nQuery:\n{base_query.strip()}"


def _custom_id(row_key: str, tag: str) -> str:
    return hashlib.sha256(f"{row_key}|{tag}".encode()).hexdigest()[:48]


def _chat_body(
    model: str, system: str | None, user_text: str, temperature: float
) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    return {"model": model, "messages": messages, "temperature": temperature}


def _message_content(result_line: dict) -> str:
    body = extract_chat_completion_body(result_line)
    return (body["choices"][0]["message"]["content"] or "").strip()


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="synthetic_data.jsonl",
        help="Input JSON/JSONL (relative to agent/ if not absolute)",
    )
    parser.add_argument(
        "--input-field",
        type=str,
        default="query",
        help="Field to send as the user query text (default: query)",
    )
    parser.add_argument(
        "--user-prefix",
        type=str,
        default="",
        help="Optional extra text inserted before the original query in each rewrite request",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output JSONL path (one extended object per line)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-mini-2026-03-17",
        help="OpenAI model name",
    )
    parser.add_argument(
        "--system",
        "-s",
        type=str,
        default=None,
        metavar="TEXT",
        help="Override SYSTEM_PROMPT defined in this script for one run.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows whose id already appears in --output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N rows (0 = all)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based input object index to start processing from (default: 1)",
    )
    parser.add_argument(
        "--all-lines",
        action="store_true",
        help="Process every non-empty line in the input (ignore the built-in line subset)",
    )
    parser.add_argument(
        "--batch-work-dir",
        type=str,
        default=".openai_batch_jobs/query_rewrite",
        help="Directory for batch input/output artifacts",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between batch status polls",
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        help="OpenAI Batch API completion window",
    )
    args = parser.parse_args()

    in_path = _resolve_input_path(args.input)
    out_path = _resolve_output_path(args.output)

    if not in_path.is_file():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY. Set it in agent/.env.", file=sys.stderr)
        sys.exit(1)

    skip_ids = _existing_ids(out_path) if args.resume else set()

    line_subset: frozenset[int] | None
    if args.all_lines:
        line_subset = None
    else:
        line_subset = _SYNTHETIC_DATA_SUBSET_1BASED

    system_src = args.system if args.system is not None else SYSTEM_PROMPT
    system_prompt = system_src.strip() or None
    rewrite_system = REWRITE_SYSTEM_PROMPT.strip() or None

    work_dir = Path(args.batch_work_dir)
    if not work_dir.is_absolute():
        work_dir = Path.cwd() / work_dir

    # ------------------------------------------------------------------
    # Pass 1: select rows, split off immediate errors (missing input field)
    # ------------------------------------------------------------------
    valid_rows: list[dict] = []  # {"key": str, "obj": dict, "prompt_text": str}
    error_rows: list[dict] = []
    skipped = 0
    skipped_subset = 0

    for line_no, obj in _iter_input_records(in_path):
        if line_no < args.start_index:
            skipped_subset += 1
            continue
        if line_subset is not None and line_no not in line_subset:
            skipped_subset += 1
            continue
        if not isinstance(obj, dict):
            continue

        oid = obj.get("id")
        if args.resume and oid in skip_ids:
            skipped += 1
            continue

        prompt_text = _get_field_value(obj, args.input_field)

        if obj.get("type") == "vague":
            clarified = obj.get("clarified_query")
            if clarified:
                prompt_text = clarified
                obj = {**obj, "query": clarified}
        elif obj.get("type") == "clear":
            skipped_subset += 1
            continue

        if prompt_text is None:
            print(
                f"[warn] row id={oid!r} has no {args.input_field!r} field",
                file=sys.stderr,
            )
            error_rows.append(
                _output_row(obj, None, gpt_error=f"missing_{args.input_field}")
            )
            continue

        if not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)

        row_key = str(oid) if oid is not None else f"__line{line_no}"
        valid_rows.append({"key": row_key, "obj": obj, "prompt_text": prompt_text})

    if args.limit:
        valid_rows = valid_rows[: args.limit]

    print(
        f"Selected {len(valid_rows)} row(s) for batch processing "
        f"(skipped(resume)={skipped}, skipped(not in line subset)={skipped_subset}, "
        f"immediate_errors={len(error_rows)})",
        file=sys.stderr,
    )

    responses_by_key: dict[str, dict] = {row["key"]: {} for row in valid_rows}
    row_errors: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Batch round 1: ambiguity analysis
    # ------------------------------------------------------------------
    if valid_rows:
        analysis_requests = [
            (
                _custom_id(row["key"], "analysis"),
                _chat_body(
                    args.model,
                    system_prompt,
                    _build_user_text(
                        row["prompt_text"], ANALYSIS_INSTRUCTION, args.user_prefix
                    ),
                    args.temperature,
                ),
            )
            for row in valid_rows
        ]
        print(f"[batch 1/2] submitting {len(analysis_requests)} analysis request(s)...", file=sys.stderr)
        batch_id = submit_chat_completions_batch(
            analysis_requests,
            work_dir=work_dir,
            completion_window=args.completion_window,
            batch_label="query_rewrite_analysis",
        )
        batch = poll_batch_until_done(
            batch_id, poll_interval_sec=args.poll_interval, verbose=True
        )
        results = download_batch_results(batch) if batch.status == "completed" else {}
        if batch.status != "completed":
            print(f"[error] analysis batch ended with status={batch.status}", file=sys.stderr)

        for row in valid_rows:
            cid = _custom_id(row["key"], "analysis")
            result_line = results.get(cid)
            if result_line is None:
                row_errors[row["key"]] = f"analysis batch missing result (status={batch.status})"
                continue
            try:
                content = _message_content(result_line)
            except Exception as e:
                row_errors[row["key"]] = f"analysis error: {e}"
                continue
            flags = _parse_analysis_json(content)
            flags["has_any_underspecification"] = "yes"
            responses_by_key[row["key"]] = flags

    # ------------------------------------------------------------------
    # Batch round 2: conditional rewrites
    # ------------------------------------------------------------------
    rewrite_requests: list[tuple[str, dict]] = []
    rewrite_lookup: dict[str, tuple[str, str]] = {}  # custom_id -> (row_key, output_key)

    for row in valid_rows:
        key = row["key"]
        if key in row_errors:
            continue
        flags = responses_by_key[key]
        for analysis_key, output_key, removal_instruction in REWRITE_TASKS:
            if str(flags.get(analysis_key) or "").strip().lower() == "yes":
                cid = _custom_id(key, output_key)
                rewrite_requests.append(
                    (
                        cid,
                        _chat_body(
                            args.model,
                            rewrite_system,
                            _build_rewrite_user_text(row["prompt_text"], removal_instruction),
                            args.temperature,
                        ),
                    )
                )
                rewrite_lookup[cid] = (key, output_key)

    if rewrite_requests:
        print(f"[batch 2/2] submitting {len(rewrite_requests)} rewrite request(s)...", file=sys.stderr)
        batch_id = submit_chat_completions_batch(
            rewrite_requests,
            work_dir=work_dir,
            completion_window=args.completion_window,
            batch_label="query_rewrite_rewrites",
        )
        batch = poll_batch_until_done(
            batch_id, poll_interval_sec=args.poll_interval, verbose=True
        )
        results = download_batch_results(batch) if batch.status == "completed" else {}
        if batch.status != "completed":
            print(f"[error] rewrite batch ended with status={batch.status}", file=sys.stderr)

        for cid, (key, output_key) in rewrite_lookup.items():
            result_line = results.get(cid)
            if result_line is None:
                continue
            try:
                responses_by_key[key][output_key] = _message_content(result_line)
            except Exception as e:
                print(f"[warn] rewrite {key}/{output_key} failed: {e}", file=sys.stderr)
                responses_by_key[key][output_key] = None

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    processed = 0
    errors = len(error_rows)
    with out_path.open("a", encoding="utf-8") as fout:
        for row in error_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in valid_rows:
            key = row["key"]
            if key in row_errors:
                out_obj = _output_row(
                    row["obj"], None, gpt_error=row_errors[key], query_text=row["prompt_text"]
                )
                errors += 1
            else:
                out_obj = _output_row(
                    row["obj"], responses_by_key[key], query_text=row["prompt_text"]
                )
                processed += 1
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(
        f"Done. wrote={processed} skipped(resume)={skipped} "
        f"skipped(not in line subset)={skipped_subset} input_errors={errors} -> {out_path}"
    )


if __name__ == "__main__":
    main()
