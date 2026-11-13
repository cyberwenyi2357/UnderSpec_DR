"""
Run AutoReasonSearchWorkflow-style eval using OpenAI Batch API for LLM steps.

Tool execution (search, browse, ask_user) still runs locally in real time between
batch rounds. ``ask_user`` uses a stub reply (no UserSimulator / extra LLM calls).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dr_agent.client import GenerateWithToolsOutput, LLMToolClient, ToolOutput
from dr_agent.openai_batch import (
    download_batch_results,
    extract_chat_completion_body,
    poll_batch_until_done,
    submit_chat_completions_batch,
)
from dr_agent.tool_interface.chained_tool import ChainedTool


@dataclass
class BatchEvalJob:
    custom_id: str
    record_id: Any
    variant_type: str
    problem: str
    template_record: dict
    messages: List[Dict[str, str]]
    original_message_count: int
    tool_calls_made: List[ToolOutput] = field(default_factory=list)
    tool_call_count: int = 0
    phase: str = "search"  # search | answer | done
    used_answer_agent: bool = False
    search_history_text: str = ""
    error: Optional[str] = None

    def generated_text(self) -> str:
        parts = []
        for msg in self.messages[self.original_message_count :]:
            parts.append(msg.get("content") or "")
        return "".join(parts)


def _custom_id(record_id: Any, variant_type: str) -> str:
    raw = f"{record_id}|{variant_type}"
    return hashlib.sha256(raw.encode()).hexdigest()[:48]


def _extract_ask_user_ease_rating(text: Any) -> Optional[int]:
    if not isinstance(text, str):
        return None
    m = re.search(r"ease[_\s-]*rating[\"']?\s*[:=]\s*(\d+)", text, re.I)
    return int(m.group(1)) if m else None


def _disable_ask_user_simulator(workflow: Any) -> None:
    """Do not wire UserSimulator; ask_user returns the AskUserTool stub reply."""
    workflow.ask_user_tool.user_simulator = None
    workflow.ask_user_tool.user_env = None


def _prepare_job_browse(workflow: Any, problem: str) -> None:
    if isinstance(workflow.composed_browse_tool, ChainedTool):
        browse_tool = workflow.composed_browse_tool.tools[0]
        browse_tool.bm25_query = problem
        browse_agent = workflow.composed_browse_tool.tools[-1]
        browse_agent.agent.question = problem
    else:
        workflow.composed_browse_tool.bm25_query = problem


def _collect_link_stats(workflow: Any, job: BatchEvalJob) -> tuple:
    browsed_links: List[str] = []
    searched_links: List[str] = []
    ask_user_ease_rating = None
    failed_tool_calls = 0
    for tool_output in job.tool_calls_made:
        if tool_output.error:
            failed_tool_calls += 1
        if tool_output.tool_name == "ask_user":
            rating = _extract_ask_user_ease_rating(tool_output.output)
            if ask_user_ease_rating is None:
                ask_user_ease_rating = rating
        if tool_output.tool_name in ["snippet_search", "google_search"]:
            searched_links.extend([d.url for d in tool_output.documents])
        if tool_output.tool_name == "browse_webpage":
            if isinstance(workflow.composed_browse_tool, ChainedTool):
                if tool_output.raw_output and (
                    chained := tool_output.raw_output.get("tool_outputs")
                ):
                    for document in chained[0].documents:
                        if document.url:
                            browsed_links.append(document.url)
            elif getattr(tool_output, "documents", None):
                for document in tool_output.documents:
                    if document.url:
                        browsed_links.append(document.url)
    return (
        list(set(browsed_links)),
        list(set(searched_links)),
        ask_user_ease_rating,
        failed_tool_calls,
    )


async def _commercial_batch_round(
    wf: Any,
    client: LLMToolClient,
    jobs: List[BatchEvalJob],
    *,
    work_dir: Path,
    api_key: Optional[str],
    max_tokens: int,
    temperature: Optional[float],
    max_tool_calls: int,
    poll_interval_sec: float,
    verbose: bool,
    round_label: str,
) -> None:
    pending = [j for j in jobs if j.phase != "done" and j.error is None]
    if not pending:
        return

    stop_sequences = client._get_all_stop_sequences()
    requests = []
    for job in pending:
        dynamic_max = client._calculate_dynamic_max_tokens_messages(
            job.messages, max_tokens
        )
        body = client.build_commercial_chat_completion_body(
            job.messages,
            temperature=temperature,
            max_tokens=dynamic_max,
            stop_sequences=stop_sequences,
        )
        requests.append((job.custom_id, body))

    batch_id = submit_chat_completions_batch(
        requests,
        work_dir=work_dir,
        api_key=api_key,
        batch_label=round_label,
    )
    if verbose:
        print(f"[openai-batch] submitted {round_label} n={len(requests)} id={batch_id}")

    batch = poll_batch_until_done(
        batch_id,
        api_key=api_key,
        poll_interval_sec=poll_interval_sec,
        verbose=verbose,
    )
    if batch.status != "completed":
        err = f"Batch {batch_id} ended with status={batch.status}"
        for job in pending:
            job.error = err
        return

    results = download_batch_results(batch, api_key=api_key)

    for job in pending:
        line = results.get(job.custom_id)
        if line is None:
            job.error = f"Missing batch result for custom_id={job.custom_id}"
            continue
        try:
            completion_body = extract_chat_completion_body(line)
            content = client.postprocess_commercial_chat_completion(
                completion_body,
                stop_sequences=stop_sequences,
                verbose=verbose,
            )
        except Exception as e:
            job.error = f"{type(e).__name__}: {e}"
            continue

        job.messages.append({"role": "assistant", "content": content})

        if job.phase == "answer":
            job.phase = "done"
            continue

        tool_match = client._find_first_tool_call(content)
        if not tool_match:
            job.search_history_text = job.generated_text()
            if "<answer>" in job.search_history_text:
                job.phase = "done"
            else:
                job.phase = "answer"
            continue

        if job.tool_call_count >= max_tool_calls:
            error_output = tool_match._create_error_output(
                error_msg="Exceed allowed tool call requests",
                call_id="",
                runtime=0,
                output="Exceed allowed tool call requests.",
            )
            job.tool_calls_made.append(error_output)
            if error_output.output:
                job.messages.append(
                    {"role": "user", "content": tool_match.format_result(error_output)}
                )
            job.search_history_text = job.generated_text()
            job.phase = (
                "answer" if "<answer>" not in job.search_history_text else "done"
            )
            continue

        try:
            _prepare_job_browse(wf, job.problem)
            tool_output = await tool_match(content)
        except Exception as e:
            job.error = f"Tool {tool_match.name}: {type(e).__name__}: {e}"
            continue

        job.tool_call_count += 1
        job.tool_calls_made.append(tool_output)
        if tool_output.called:
            job.messages.append(
                {"role": "user", "content": tool_match.format_result(tool_output)}
            )

        job.search_history_text = job.generated_text()
        if "<answer>" in job.search_history_text:
            job.phase = "done"


async def _run_search_phase(
    wf: Any,
    jobs: List[BatchEvalJob],
    *,
    work_dir: Path,
    api_key: Optional[str],
    max_tokens: int,
    temperature: Optional[float],
    max_tool_calls: int,
    poll_interval_sec: float,
    verbose: bool,
) -> None:
    client: LLMToolClient = wf.search_agent.client
    if not client.is_commercial_api_model:
        raise ValueError(
            "OpenAI Batch mode requires a commercial API model (e.g. gpt-*). "
            f"Got: {client.model_name}"
        )

    round_num = 0
    while True:
        active = [
            j
            for j in jobs
            if j.phase == "search" and j.error is None and j.tool_call_count < max_tool_calls
        ]
        if not active:
            break
        round_num += 1
        await _commercial_batch_round(
            wf,
            client,
            active,
            work_dir=work_dir,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            max_tool_calls=max_tool_calls,
            poll_interval_sec=poll_interval_sec,
            verbose=verbose,
            round_label=f"search_r{round_num}",
        )

    for job in jobs:
        if job.phase == "search" and job.error is None:
            job.search_history_text = job.generated_text()
            if "<answer>" in job.search_history_text:
                job.phase = "done"
            else:
                job.phase = "answer"


async def _run_answer_phase(
    wf: Any,
    jobs: List[BatchEvalJob],
    *,
    work_dir: Path,
    api_key: Optional[str],
    max_tokens: int,
    temperature: Optional[float],
    poll_interval_sec: float,
    verbose: bool,
    dataset_name: str,
) -> None:
    answer_jobs = [j for j in jobs if j.phase == "answer" and j.error is None]
    if not answer_jobs:
        return

    client: LLMToolClient = wf.answer_agent.client
    for job in answer_jobs:
        job.used_answer_agent = True
        job.messages = wf.answer_agent.prompt(
            question=job.problem,
            history=job.search_history_text,
            dataset_name=dataset_name,
        )
        if job.messages and job.messages[-1]["role"] == "user":
            job.messages[-1]["content"] += "<answer>"
        job.original_message_count = len(job.messages)
        job.phase = "answer"

    await _commercial_batch_round(
        wf,
        client,
        answer_jobs,
        work_dir=work_dir,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        max_tool_calls=0,
        poll_interval_sec=poll_interval_sec,
        verbose=verbose,
        round_label="answer",
    )


def job_to_workflow_result(wf: Any, job: BatchEvalJob) -> Dict[str, Any]:
    if job.error:
        raise RuntimeError(job.error)

    generated = job.generated_text()
    traces = GenerateWithToolsOutput(
        tool_calls=job.tool_calls_made,
        generated_text=generated,
        total_tokens=wf.search_agent.client._count_tokens_messages(job.messages),
        tool_call_count=job.tool_call_count,
        stopped_reason="natural",
    )

    browsed, searched, ask_rating, failed = _collect_link_stats(wf, job)

    if job.used_answer_agent:
        final = wf.answer_agent.postprocess_output(traces)
    else:
        final = wf.search_agent.postprocess_output(traces)

    return {
        "final_response": final,
        "full_traces": traces,
        "browsed_links": browsed,
        "searched_links": searched,
        "total_tool_calls": job.tool_call_count,
        "total_failed_tool_calls": failed,
        "ask_user_ease_rating": ask_rating,
    }


def make_batch_job(
    wf: Any,
    *,
    problem: str,
    template_record: dict,
    record_id: Any,
    variant_type: str,
    dataset_name: str = "researchqa",
) -> BatchEvalJob:
    _prepare_job_browse(wf, problem)
    messages = wf.search_agent.prompt(question=problem, dataset_name=dataset_name)
    return BatchEvalJob(
        custom_id=_custom_id(record_id, variant_type),
        record_id=record_id,
        variant_type=variant_type,
        problem=problem,
        template_record=template_record,
        messages=list(messages),
        original_message_count=len(messages),
    )


async def run_batch_eval_jobs(
    wf: Any,
    jobs: List[BatchEvalJob],
    *,
    work_dir: Path,
    poll_interval_sec: float = 10.0,
    verbose: bool = False,
    dataset_name: str = "researchqa",
) -> Dict[str, Dict[str, Any]]:
    """
    Run many variant jobs with shared batch rounds.

    Returns ``{custom_id: workflow_result_dict}``.
    """
    cfg = wf.configuration
    work_dir.mkdir(parents=True, exist_ok=True)
    _disable_ask_user_simulator(wf)

    await _run_search_phase(
        wf,
        jobs,
        work_dir=work_dir,
        api_key=cfg.search_agent_api_key,
        max_tokens=cfg.search_agent_max_tokens,
        temperature=cfg.search_agent_temperature,
        max_tool_calls=cfg.search_agent_max_tool_calls,
        poll_interval_sec=poll_interval_sec,
        verbose=verbose,
    )

    await _run_answer_phase(
        wf,
        jobs,
        work_dir=work_dir,
        api_key=cfg.search_agent_api_key,
        max_tokens=cfg.search_agent_max_tokens,
        temperature=cfg.search_agent_temperature,
        poll_interval_sec=poll_interval_sec,
        verbose=verbose,
        dataset_name=dataset_name,
    )

    out: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        if job.error:
            out[job.custom_id] = {"error": job.error}
        else:
            try:
                out[job.custom_id] = job_to_workflow_result(wf, job)
            except Exception as e:
                out[job.custom_id] = {"error": f"{type(e).__name__}: {e}"}
    return out
