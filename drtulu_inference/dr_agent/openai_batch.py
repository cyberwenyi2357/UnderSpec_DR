"""OpenAI Batch API helpers for chat completions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

CHAT_COMPLETIONS_URL = "/v1/chat/completions"
TERMINAL_BATCH_STATUSES = frozenset(
    {"completed", "failed", "expired", "cancelled", "cancelling"}
)


def _client(api_key: Optional[str] = None) -> OpenAI:
    return OpenAI(api_key=api_key) if api_key else OpenAI()


def write_batch_input_jsonl(
    path: Path,
    requests: List[Tuple[str, Dict[str, Any]]],
) -> None:
    """Write batch input file. Each item is (custom_id, chat-completions body)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for custom_id, body in requests:
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": CHAT_COMPLETIONS_URL,
                "body": body,
            }
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def submit_chat_completions_batch(
    requests: List[Tuple[str, Dict[str, Any]]],
    *,
    work_dir: Path,
    api_key: Optional[str] = None,
    completion_window: str = "24h",
    batch_label: Optional[str] = None,
) -> str:
    """Upload JSONL and create a batch job. Returns batch id."""
    if not requests:
        raise ValueError("submit_chat_completions_batch: empty requests")

    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / f"batch_input_{int(time.time())}.jsonl"
    write_batch_input_jsonl(input_path, requests)

    client = _client(api_key)
    with input_path.open("rb") as fh:
        input_file = client.files.create(file=fh, purpose="batch")

    metadata = {}
    if batch_label:
        metadata["label"] = batch_label[:512]

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=CHAT_COMPLETIONS_URL,
        completion_window=completion_window,
        metadata=metadata or None,
    )
    return batch.id


def poll_batch_until_done(
    batch_id: str,
    *,
    api_key: Optional[str] = None,
    poll_interval_sec: float = 10.0,
    timeout_sec: Optional[float] = None,
    verbose: bool = False,
) -> Any:
    """Poll batch status until terminal state. Returns final batch object."""
    client = _client(api_key)
    started = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        if verbose:
            print(
                f"[batch] id={batch_id} status={batch.status} "
                f"completed={getattr(counts, 'completed', None)} "
                f"failed={getattr(counts, 'failed', None)} "
                f"total={getattr(counts, 'total', None)}",
                flush=True,
            )
        if batch.status in TERMINAL_BATCH_STATUSES:
            return batch
        if timeout_sec is not None and (time.time() - started) > timeout_sec:
            raise TimeoutError(
                f"Batch {batch_id} did not finish within {timeout_sec}s (status={batch.status})"
            )
        time.sleep(poll_interval_sec)


def download_batch_results(
    batch: Any,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Download batch output JSONL.

    Returns mapping custom_id -> parsed line dict (with ``response`` or ``error``).
    """
    client = _client(api_key)
    if batch.status != "completed":
        raise RuntimeError(
            f"Batch {batch.id} not completed (status={batch.status}); "
            f"errors_file_id={getattr(batch, 'error_file_id', None)}"
        )
    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch.id} completed but has no output_file_id")

    raw = client.files.content(batch.output_file_id).text
    results: Dict[str, Dict[str, Any]] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id")
        if cid:
            results[cid] = obj
    return results


def extract_chat_completion_body(result_line: Dict[str, Any]) -> Dict[str, Any]:
    """Return chat completion JSON body from a batch output line."""
    if result_line.get("error"):
        err = result_line["error"]
        raise RuntimeError(f"Batch request failed: {err}")
    response = result_line.get("response") or {}
    status = response.get("status_code")
    if status != 200:
        raise RuntimeError(f"Batch response status {status}: {response.get('body')}")
    body = response.get("body")
    if isinstance(body, str):
        body = json.loads(body)
    if not isinstance(body, dict):
        raise RuntimeError(f"Unexpected batch response body type: {type(body)}")
    return body
