import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from scholarqacs_rubric import PendingJudgeCall

LOGGER = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}


def _to_inlined_request(call: PendingJudgeCall, temperature: float, thinking_level: str) -> types.InlinedRequest:
    return types.InlinedRequest(
        contents=call.user_prompt,
        metadata={"key": call.key},
        config=types.GenerateContentConfig(
            system_instruction=call.system_prompt,
            temperature=temperature,
            max_output_tokens=call.max_tokens,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level, include_thoughts=False),
        ),
    )


def run_gemini_batch(
    model: str,
    calls: List[PendingJudgeCall],
    temperature: float = 0,
    thinking_level: str = "low",
    poll_interval: int = 20,
    chunk_size: int = 500,
) -> Dict[str, Optional[str]]:
    """
    Submit ``calls`` (each with a globally-unique ``key``) to the Gemini Batch API, block until every
    submitted job reaches a terminal state, and return a dict mapping each call's key to the raw text
    response (or None if that call failed or is missing from the results).

    ``model`` must be the bare Gemini model id (e.g. "gemini-3.1-pro-preview"), not the "gemini/"-prefixed
    litellm identifier.
    """
    if model.startswith("gemini/"):
        model = model[len("gemini/"):]

    client = genai.Client()

    jobs = []
    for chunk_start in range(0, len(calls), chunk_size):
        chunk = calls[chunk_start: chunk_start + chunk_size]
        inlined = [_to_inlined_request(c, temperature, thinking_level) for c in chunk]
        job = client.batches.create(model=model, src=inlined)
        LOGGER.info(f"Submitted Gemini batch job {job.name} with {len(chunk)} requests "
                    f"({chunk_start + len(chunk)}/{len(calls)})")
        jobs.append(job)

    pending = {job.name: job for job in jobs}
    finished = {}
    while pending:
        time.sleep(poll_interval)
        for name in list(pending):
            job = client.batches.get(name=name)
            if job.state.name in TERMINAL_STATES:
                LOGGER.info(f"Batch job {name} finished: {job.state.name}")
                finished[name] = job
                del pending[name]
            else:
                pending[name] = job
        if pending:
            LOGGER.info(f"Waiting on {len(pending)}/{len(jobs)} batch job(s)...")

    results: Dict[str, Optional[str]] = {}
    for job in finished.values():
        if job.state.name == "JOB_STATE_FAILED":
            LOGGER.warning(f"Batch job {job.name} failed: {job.error}")
            continue
        if not job.dest or not job.dest.inlined_responses:
            LOGGER.warning(f"Batch job {job.name} ({job.state.name}) has no inlined responses")
            continue
        for inlined_resp in job.dest.inlined_responses:
            key = (inlined_resp.metadata or {}).get("key")
            if key is None:
                continue
            if inlined_resp.error is not None:
                LOGGER.warning(f"Batch request {key} failed: {inlined_resp.error}")
                results[key] = None
                continue
            try:
                results[key] = inlined_resp.response.text
            except Exception as e:
                LOGGER.warning(f"Could not extract text for {key}: {e}")
                results[key] = None

    missing = [c.key for c in calls if c.key not in results]
    if missing:
        LOGGER.warning(f"{len(missing)} of {len(calls)} batch requests are missing from the results "
                        f"(counted as failed): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        for key in missing:
            results[key] = None

    return results
