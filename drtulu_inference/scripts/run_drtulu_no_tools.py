#!/usr/bin/env python3
"""
Run DR-Tulu-8B (RL checkpoint) on a query file as a plain chat model: no system prompt,
no tool descriptions, one generation per query, no tool execution.

Each output line: {id, query, query_answer, query_raw_output, attempted_tool_call}
  - query_answer:       text after </think>, with <answer> tags stripped if present
  - query_raw_output:   full generation (including <think>...</think>)
  - attempted_tool_call: True if the model still emitted <call_tool> (it was not executed)
  - truncated:          True if generation stopped at --max-tokens

Usage (from drtulu_inference/, with vllm installed):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_drtulu_no_tools.py \\
        --query-file queries.jsonl \\
        --output answers_drtulu_no_tools.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams

_AGENT_DIR = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _AGENT_DIR / path


def _extract_answer(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    return (m.group(1) if m else text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="rl-research/DR-Tulu-8B")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=40960)
    # Defaults follow the checkpoint's generation_config.json.
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    # 1.0 = off; values > 1 penalize tokens already in the prompt/output (e.g. 1.05-1.1).
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()

    qpath, out_path = _resolve(args.query_file), _resolve(args.output)
    records = [
        json.loads(l) for l in qpath.read_text(encoding="utf-8").split("\n") if l.strip()
    ]

    done: set[str] = set()
    if out_path.is_file():
        done = {
            str(json.loads(l)["id"])
            for l in out_path.read_text(encoding="utf-8").split("\n")
            if l.strip()
        }
    todo = [r for r in records if str(r.get("id")) not in done]
    print(f"{len(records)} queries, {len(done)} already in output, {len(todo)} to run", flush=True)
    if not todo:
        return

    llm = LLM(model=args.model, max_model_len=args.max_model_len, seed=args.seed)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    conversations = [[{"role": "user", "content": r["query"]}] for r in todo]
    outputs = llm.chat(conversations, sampling)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for rec, out in zip(todo, outputs):
            text = out.outputs[0].text
            fh.write(
                json.dumps(
                    {
                        "id": rec.get("id"),
                        "query": rec["query"],
                        "query_answer": _extract_answer(text),
                        "query_raw_output": text,
                        "attempted_tool_call": "<call_tool" in text,
                        "truncated": out.outputs[0].finish_reason == "length",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(todo)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
