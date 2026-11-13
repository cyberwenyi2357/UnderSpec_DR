#!/usr/bin/env python3
"""
Run ``AutoReasonSearchWorkflow`` once per record in a query template file, writing one
JSON line per run with the full trajectory payload.

**Standard mode** (default): read ``--query-file``, append rows to ``--output``.

**Merge mode**: pass ``--interactions-json`` + ``--merge-trajectories``. For each
interaction object ``{record_index, query, query_type?}``, run the workflow on ``query``
without ``clarified_query``, then set ``without_clarification_answer`` on the trajectory
row whose ``index`` equals ``record_index`` (rewrites the trajectories file in place).

Usage (from ``drtulu_inference/``)::

    python scripts/eval_query_template_batch.py --config workflows/auto_search_sft.yaml        # DR-Tulu-8B (vLLM)
    python scripts/eval_query_template_batch.py --config workflows/auto_search_sft-gpt5.4.yaml # GPT-5.4 (OpenAI API)

    python scripts/eval_query_template_batch.py \\
        --interactions-json realdata_interactions.json \\
        --merge-trajectories realdata_eval_trajectories.jsonl

    # Synthetic dataset (same merge logic; shorthand flag):
    python scripts/eval_query_template_batch.py --synthetic-merge

Requires MCP/search credentials as for the normal workflow; uses ``skip_service_check=True``
so start Serper/MCP yourself before running.

**OpenAI Batch API** (``--openai-batch``): LLM steps use OpenAI's Batch API (cheaper, async).
Tool calls (search/browse/ask_user) still run locally between batch rounds.
``ask_user`` uses a stub reply (no UserSimulator). Batch jobs are written under
``--batch-work-dir``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import tiktoken

_AGENT_DIR = Path(__file__).resolve().parent.parent

if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from dr_agent.workflow import _parse_overrides

_tokenizer = tiktoken.get_encoding("cl100k_base")


def _count_tokens(obj: Any) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return len(_tokenizer.encode(text))


def _load_workflow_module():
    path = _AGENT_DIR / "workflows" / "auto_search_sft.py"
    spec = importlib.util.spec_from_file_location("auto_search_sft_eval", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _jsonable_traces(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _jsonable_traces(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable_traces(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


# Field name (as written by scripts/query_rewrite.py) -> short variant label
# used for the output keys ("<label>_answer" / "<label>_full_traces").
QUERY_VARIANT_FIELDS: list[tuple[str, str]] = [
    ("query", "query"),
    ("input", "query"),
    ("query_without_scope_limitation", "without_scope_limitation"),
    ("query_fully_underspecified", "fully_underspecified"),
    ("query_without_term", "without_term"),
    ("query_without_preference", "without_preference"),
    # Field names used by ../query/*.jsonl
    ("without_scope", "without_scope_limitation"),
    ("fully_underspecified", "fully_underspecified"),
    ("without_term", "without_term"),
    ("without_preference", "without_preference"),
]


def _load_trajectory_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        # A single JSON array (possibly pretty-printed across multiple lines),
        # e.g. output of json.dump(..., indent=2).
        data = json.loads(text)
        return [obj for obj in data if isinstance(obj, dict)]

    rows: list[dict[str, Any]] = []
    # Split on "\n" only: str.splitlines() also breaks on characters like U+0085 / U+2028,
    # which json.dumps(ensure_ascii=False) leaves unescaped inside strings.
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _write_trajectory_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for obj in rows:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _row_index_by_trajectory_index(rows: list[dict[str, Any]]) -> dict[int, int]:
    m: dict[int, int] = {}
    for pos, obj in enumerate(rows):
        i = obj.get("index")
        if isinstance(i, int):
            m[i] = pos
    return m


async def _run_interactions_merge(
    *,
    interactions_path: Path,
    trajectories_path: Path,
    cfg_path: Path,
    config_overrides: dict[str, Any],
    limit: int | None,
    resume: bool,
    verbose: bool,
) -> None:
    interactions_raw = json.loads(interactions_path.read_text(encoding="utf-8"))
    if not isinstance(interactions_raw, list):
        print("interactions file must be a JSON array", file=sys.stderr)
        sys.exit(1)

    rows = _load_trajectory_rows(trajectories_path)
    idx_map = _row_index_by_trajectory_index(rows)

    mod = _load_workflow_module()
    wf = mod.AutoReasonSearchWorkflow(
        configuration=str(cfg_path),
        skip_service_check=True,
        **config_overrides,
    )

    processed = 0
    for entry in interactions_raw:
        if limit is not None and processed >= limit:
            break
        if not isinstance(entry, dict):
            continue
        rid = entry.get("record_index")
        if not isinstance(rid, int):
            continue
        query = (entry.get("query") or "").strip()
        if not query:
            print(f"[skip] record_index={rid}: empty query", file=sys.stderr)
            continue

        pos = idx_map.get(rid)
        if pos is None:
            print(
                f"[warn] record_index={rid}: no trajectory with index={rid}",
                file=sys.stderr,
            )
            continue

        if resume and rows[pos].get("without_clarification_answer") is not None:
            print(
                f"[resume] skip index={rid} (without_clarification_answer already set)",
                flush=True,
            )
            continue

        template_record: dict[str, Any] = {"query": query}
        qt = entry.get("query_type")
        if qt is not None:
            template_record["type"] = qt

        try:
            result = await wf(
                problem=query,
                template_record=template_record,
                verbose=verbose,
            )
            rows[pos]["without_clarification_answer"] = result.get("final_response")
            rows[pos]["without_clarification_ask_user_ease_rating"] = result.get(
                "ask_user_ease_rating"
            )
            rows[pos]["without_clarification_ask_user_ease_ratings"] = result.get(
                "ask_user_ease_ratings"
            )
        except Exception as e:
            print(f"[error] index={rid}: {e}", file=sys.stderr)
            rows[pos]["without_clarification_answer"] = None

        _write_trajectory_rows(trajectories_path, rows)
        processed += 1
        preview = query[:72] + ("…" if len(query) > 72 else "")
        print(f"[merge] {processed} record_index={rid} query={preview!r}", flush=True)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(_AGENT_DIR / "workflows" / "auto_search_sft.yaml"),
        help="Workflow YAML (relative paths resolve under agent/ when not absolute)",
    )
    parser.add_argument(
        "--config-overrides",
        type=str,
        default=None,
        help="Override workflow config, e.g. 'search_agent_base_url=http://localhost:30002/v1'",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default="synthetic_testing_set.jsonl",
        help="Template / eval set path (ignored when --interactions-json is set)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_eval_trajectories.jsonl",
        help="Output JSONL path (ignored when --interactions-json is set)",
    )
    parser.add_argument(
        "--interactions-json",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "JSON array of {record_index, query, query_type?}; merge mode — updates "
            "--merge-trajectories in place with without_clarification_answer"
        ),
    )
    parser.add_argument(
        "--merge-trajectories",
        type=str,
        default="realdata_eval_trajectories.jsonl",
        help="Trajectories JSONL to update (merge mode only; under agent/ if relative)",
    )
    parser.add_argument(
        "--synthetic-merge",
        action="store_true",
        help=(
            "Merge mode: use agent/synthetic_interactions.json and "
            "agent/synthetic_eval_trajectories.jsonl (overrides --interactions-json / "
            "--merge-trajectories for this run)"
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--record-limit", type=int, default=None, help="Max number of records (objects) to process")
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        metavar="ID[,ID,...]",
        help="Comma-separated list of record ids to run (matches the 'id' field in the query file)",
    )
    parser.add_argument(
        "--update-result",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Existing result JSONL to update in place. For each processed record, overwrite "
            "the answer/traces fields of the matching row (by 'id') instead of appending to --output."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Merge mode: skip rows that already have without_clarification_answer",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--openai-batch",
        action="store_true",
        help=(
            "Use OpenAI Batch API for LLM calls (multi-round: submit completions, run tools, "
            "repeat). ask_user uses a stub reply (no UserSimulator)."
        ),
    )
    parser.add_argument(
        "--batch-work-dir",
        type=str,
        default=".openai_batch_jobs",
        help="Directory for batch input/output artifacts (default: .openai_batch_jobs)",
    )
    parser.add_argument(
        "--batch-poll-interval",
        type=float,
        default=10.0,
        help="Seconds between batch status polls (default: 10)",
    )
    args = parser.parse_args()

    if args.synthetic_merge:
        args.interactions_json = str(_AGENT_DIR / "synthetic_interactions.json")
        args.merge_trajectories = "synthetic_eval_trajectories.jsonl"

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _AGENT_DIR / cfg_path

    config_overrides = _parse_overrides(args.config_overrides)
    if config_overrides:
        print(f"Config overrides: {config_overrides}", flush=True)

    if args.openai_batch and args.interactions_json:
        print(
            "--openai-batch is not supported with --interactions-json merge mode yet.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.interactions_json:
        ipath = Path(args.interactions_json)
        if not ipath.is_absolute():
            ipath = _AGENT_DIR / ipath
        tpath = Path(args.merge_trajectories)
        if not tpath.is_absolute():
            tpath = _AGENT_DIR / tpath
        if not ipath.is_file():
            print(f"Missing: {ipath}", file=sys.stderr)
            sys.exit(1)
        if not tpath.is_file():
            print(f"Missing: {tpath}", file=sys.stderr)
            sys.exit(1)
        await _run_interactions_merge(
            interactions_path=ipath,
            trajectories_path=tpath,
            cfg_path=cfg_path,
            config_overrides=config_overrides,
            limit=args.limit,
            resume=args.resume,
            verbose=args.verbose,
        )
        return

    mod = _load_workflow_module()
    AutoReasonSearchWorkflow = mod.AutoReasonSearchWorkflow

    qpath = Path(args.query_file)
    if not qpath.is_absolute():
        qpath = _AGENT_DIR / qpath

    records = _load_trajectory_rows(qpath)
    if not records:
        print(f"No records loaded from {qpath}", file=sys.stderr)
        sys.exit(1)

    wf = AutoReasonSearchWorkflow(
        configuration=str(cfg_path),
        skip_service_check=True,
        **config_overrides,
    )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path

    target_ids: set[str] | None = None
    if args.ids:
        target_ids = {x.strip() for x in args.ids.split(",")}

    # Standard mode appends to --output, so re-running the same command would otherwise
    # duplicate rows already written. Skip ids already present in --output by default.
    # (--update-result has its own upsert-by-id semantics, so it's exempt from this.)
    existing_ids: set[str] | None = None
    if not args.update_result and out_path.is_file():
        existing_ids = {
            str(r.get("id")) for r in _load_trajectory_rows(out_path) if r.get("id") is not None
        }
        print(f"[skip-existing] {len(existing_ids)} id(s) already in {out_path}", flush=True)

    written = 0
    records_processed = 0
    for i, row in enumerate(records):
        if args.record_limit is not None and records_processed >= args.record_limit:
            break
        if args.limit is not None and written >= args.limit:
            break
        if target_ids is not None and str(row.get("id")) not in target_ids:
            continue
        if existing_ids is not None and str(row.get("id")) in existing_ids:
            continue
        variants: list[tuple[str, str, dict]] = []
        for field_name, variant_type in QUERY_VARIANT_FIELDS:
            variant_q = (row.get(field_name) or "").strip()
            if variant_q:
                variants.append((variant_q, variant_type, {**row}))

        if not variants:
            fields = ", ".join(f for f, _ in QUERY_VARIANT_FIELDS)
            print(f"[skip] index={i} id={row.get('id')}: no non-empty query field found (checked: {fields})", flush=True)
            continue

        line_obj: dict[str, Any] = {"id": row.get("id")}
        for variant_q, variant_type, _ in variants:
            query_key = "query" if variant_type == "query" else f"{variant_type}_query"
            line_obj[query_key] = variant_q
            line_obj[f"{variant_type}_full_traces"] = None

        if args.openai_batch:
            from dr_agent.batch_workflow_eval import make_batch_job, run_batch_eval_jobs

            batch_work_dir = Path(args.batch_work_dir)
            if not batch_work_dir.is_absolute():
                batch_work_dir = Path.cwd() / batch_work_dir

            jobs = []
            variant_meta: list[tuple[str, str, str]] = []
            for variant_q, variant_type, template_record in variants:
                jobs.append(
                    make_batch_job(
                        wf,
                        problem=variant_q or "(empty)",
                        template_record=template_record,
                        record_id=row.get("id"),
                        variant_type=variant_type,
                    )
                )
                variant_meta.append((variant_type, f"{variant_type}_full_traces"))

            print(
                f"[openai-batch] index={i} id={row.get('id')} variants={len(jobs)}",
                flush=True,
            )
            try:
                results_by_id = await run_batch_eval_jobs(
                    wf,
                    jobs,
                    work_dir=batch_work_dir / f"record_{row.get('id')}",
                    poll_interval_sec=args.batch_poll_interval,
                    verbose=args.verbose,
                )
                for job, (variant_type, traces_key) in zip(jobs, variant_meta):
                    result = results_by_id.get(job.custom_id) or {}
                    if result.get("error"):
                        line_obj["error"] = result["error"]
                        print(
                            f"[error] index={i} [{variant_type}]: {result['error']}",
                            file=sys.stderr,
                        )
                    else:
                        line_obj[traces_key] = _jsonable_traces(result.get("full_traces"))
            except Exception as e:
                line_obj["error"] = f"{type(e).__name__}: {e}"
                print(f"[error] index={i} batch: {e}", file=sys.stderr)
        else:
            for variant_q, variant_type, template_record in variants:
                traces_key = f"{variant_type}_full_traces"
                preview = variant_q[:72] + ("…" if len(variant_q) > 72 else "")
                print(f"[run] index={i} [{variant_type}] query={preview!r}", flush=True)
                try:
                    result = await wf(
                        problem=variant_q or "(empty)",
                        template_record=template_record,
                        verbose=args.verbose,
                    )
                    line_obj[traces_key] = _jsonable_traces(result.get("full_traces"))
                except Exception as e:
                    line_obj["error"] = f"{type(e).__name__}: {e}"
                    print(f"[error] index={i} [{variant_type}]: {e}", file=sys.stderr)

        if args.update_result:
            upath = Path(args.update_result)
            if not upath.is_absolute():
                upath = _AGENT_DIR / upath
            existing = _load_trajectory_rows(upath)
            updated = False
            for er in existing:
                if er.get("id") == line_obj["id"]:
                    for k, v in line_obj.items():
                        if k != "id" and v is not None:
                            er[k] = v
                    updated = True
                    break
            if not updated:
                existing.append(line_obj)
            _write_trajectory_rows(upath, existing)
            print(f"[update-result] id={line_obj['id']} written to {upath}", flush=True)
        else:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
        written += 1
        records_processed += 1
        print(f"[eval] {written} index={i} id={row.get('id')} done", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
