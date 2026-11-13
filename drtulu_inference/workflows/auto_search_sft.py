import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dotenv
from dr_agent.agent_interface import BaseAgent
from dr_agent.client import DocumentToolOutput, GenerationConfig, LLMToolClient, ToolOutput
from dr_agent.shared_prompts import UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS
from dr_agent.tool_interface.ask_user_tool import AskUserTool
from dr_agent.tool_interface.chained_tool import ChainedTool
from dr_agent.tool_interface.mcp_tools import (
    BaseTool,
    Crawl4AIBrowseTool,
    JinaBrowseTool,
    SemanticScholarSnippetSearchTool,
    SerperBrowseTool,
    SerperSearchTool,
)
from dr_agent.utils import (
    check_port,
    extract_port_from_url,
    launch_mcp_server,
    launch_vllm_server,
)
from dr_agent.workflow import BaseWorkflow, BaseWorkflowConfiguration
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

# Load env from agent/.env (this file lives in agent/workflows/). Optional repo-root .env
# is loaded first for legacy setups; agent/.env overrides on duplicate keys.
_agent_dir = Path(__file__).resolve().parent.parent
_repo_root = _agent_dir.parent
dotenv.load_dotenv(_repo_root / ".env")
dotenv.load_dotenv(_agent_dir / ".env", override=True)

DEFAULT_ASK_USER_SIMULATOR_PROMPT = (
    "You are a human user chatting with a research assistant. "
    "Answer the assistant's clarifying questions briefly and naturally, in plain language. "
    "If you truly cannot answer, say so in one short sentence. "
    "Do not use markdown or bullet lists. Never ask the assistant a question back."
)


def _extract_query_template_fields_regex(text: str) -> Optional[dict]:
    """
    Best-effort extraction when the file is not strict JSON (e.g. unescaped newlines
    inside ``"final_response": "..."``). Pulls the first ``query``, ``type``, ``clarified_query`` string values.
    """

    def _first_string_field(key: str) -> Optional[str]:
        m = re.search(
            rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"',
            text,
        )
        return m.group(1) if m else None

    clarified = (_first_string_field("clarified_query") or "").strip()
    if not clarified:
        return None
    out: Dict[str, Any] = {"clarified_query": clarified}
    q = _first_string_field("query")
    if q is not None:
        out["query"] = q
    t = _first_string_field("type")
    if t is not None:
        out["type"] = t
    return out


def _extract_query_template_fields_regex_loose(text: str) -> Optional[dict]:
    """
    Like ``_extract_query_template_fields_regex`` but does not require ``clarified_query``.
    Used when parsing a single object chunk whose JSON is invalid (e.g. raw newlines in ``final_response``).
    """

    def _first_string_field(key: str) -> Optional[str]:
        m = re.search(
            rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"',
            text,
        )
        return m.group(1) if m else None

    out: Dict[str, Any] = {}
    q = _first_string_field("query")
    if q is not None:
        out["query"] = q
    clarified = _first_string_field("clarified_query")
    if clarified is not None:
        out["clarified_query"] = clarified
    t = _first_string_field("type")
    if t is not None:
        out["type"] = t
    if not out:
        return None
    return out


def _try_parse_query_template_object(part: str) -> Optional[dict]:
    """Parse one logical JSON object; fall back to regex field extraction."""
    chunk = part.strip()
    if not chunk:
        return None
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    row = _extract_query_template_fields_regex_loose(chunk)
    if row:
        return row
    return _extract_query_template_fields_regex(chunk)


def _load_all_query_template_jsonl(path: Path) -> List[dict]:
    """
    Load every record: either a JSON array of objects, or pretty-printed objects separated
    by a blank line between ``}`` and ``{``. Per-chunk regex fallback when JSON is invalid.
    """
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    parts = re.split(r"(?<=\})\s*\n\s*\n(?=\{)", text)
    out: List[dict] = []
    for part in parts:
        obj = _try_parse_query_template_object(part)
        if obj is not None:
            out.append(obj)
    return out


def _load_query_template_jsonl_first(path: Path) -> Optional[dict]:
    """
    Load the first JSON object from a file (pretty-printed multi-object files are ok).
    Used for ``clarified_query`` / ``type``; no keying by ``query``.
    Falls back to regex field extraction if JSON is invalid (common when ``final_response`` contains raw newlines).
    """
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        idx = end
        if isinstance(obj, dict):
            return obj

    row = _extract_query_template_fields_regex(text)
    if row and os.environ.get("ASK_USER_DEBUG", "").strip():
        print(
            "[ask_user] query_template JSON parse failed; using regex fallback for "
            "query/type/clarified_query",
            flush=True,
        )
    return row


def _user_simulator_prompt_from_query_template_row(user_query: str, row: dict) -> str:
    """Build UserSimulator system prompt using ``clarified_query`` (and optional ``type``)."""
    clarified = (row.get("clarified_query") or row.get("query") or "").strip()
    qtype = row.get("type")
    lines = [
  "You are a human user talking to a research agent.",

    "You originally asked the agent an underspecified question because you did not fully articulate your intent.",

    "Answer the agent's clarifying questions briefly and naturally, in plain language.",

    "",

    "## What you originally asked the assistant",

    user_query.strip(),

    "",

    "## Your hidden intent",

    clarified,

    "",
    "The hidden intent represents what you actually wanted to know.",

    "If the assistant asks clarifying questions, answer in a way that is consistent with the hidden intent.",
    ]
    return "\n".join(lines)


def _user_simulator_prompt_initial_only(initial_query: str) -> str:
    """UserSimulator system prompt when there is no ``clarified_query`` (e.g. clear tasks)."""
    q = (initial_query or "").strip()
    lines = [
        "You are a human user talking to a research agent.",
        "Answer the agent's clarifying questions briefly and naturally, in plain language.",
        "",
        "## What you asked the assistant",
        q,
        "",
        "If the assistant asks clarifying questions, answer in a way consistent with what you asked above.",
    ]
    return "\n".join(lines)


@dataclass
class WebPageReaderAgentV2(BaseAgent):
    question: Optional[str] = None
    prompt = """
We are searching on the internet for the following question:
{question}
Here is some webpage scraped from the internet:
{document}
Can you clean the raw webpage text and convert it into a more readable format? You should remove all the unnecessary information and keep the main content of the page. Please produce the output in the format of "Cleaned webpage text:\n[you text here]".
""".strip()

    def preprocess_input(self, documents: Union[str, Any]) -> Dict[str, str]:
        # Accept either a raw string or a ToolOutput-like object with an `output` attribute
        assert self.question is not None, "Question is not set"

        if isinstance(documents, DocumentToolOutput):
            # print("using DocumentToolOutput")
            doc_str = "\n".join(
                [
                    document.simple_stringify()[: 32000 * 4 // len(documents.documents)]
                    for document in documents.documents
                ]
            )
        elif hasattr(documents, "output"):
            doc_str = documents.output
        else:
            doc_str = documents if isinstance(documents, str) else str(documents)
        input_params = {"question": self.question, "document": doc_str}
        # print(input_params)
        return input_params

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        output_string = result.generated_text
        if "</think>" in output_string:
            output_string = "".join(output_string.split("</think>")[1:]).strip()

        if "Cleaned webpage text:" in output_string:
            output_string = output_string.split("Cleaned webpage text:")[1].strip()

        return output_string


@dataclass
class SearchAgent(BaseAgent):
    prompt_version: str = "v20250907"

    def prompt(
        self,
        question: str,
        dataset_name: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:

        PROMPT = UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS[self.prompt_version]
        system_prompt = PROMPT["system_prompt"]

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if history:
            messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        return messages

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        return result.generated_text


@dataclass
class AnswerAgent(BaseAgent):
    prompt_version: str = "v20250907"

    def prompt(self, question: str, history: str, dataset_name: str) -> str:

        PROMPT = UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS[self.prompt_version]
        _answer_datasets = (
            {
                "2wiki",
                "simpleqa",
                "browsecomp",
                "bc_synthetic_depth_one_v2_verified",
                "bc_synthetic_varied_depth_o3_verified",
                "webwalker",
                "webshaper",
            }
            | {"sqav2", "genetic_diseases_qa"}
            | {"healthbench", "deep_research_bench", "researchqa"}
        )
        if dataset_name not in _answer_datasets:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        return [
            {
                "role": "system",
                "content": PROMPT["system_prompt"],
            },
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": history,
            },
            {
                "role": "user",
                "content": "Now please generate an answer based on the search results by far.",
            },
        ]

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        return result.generated_text


class NoBrowseTool(BaseTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def __call__(self, *args, **kwargs):
        return DocumentToolOutput(
            output="Browse tool is not available at this time. Please try other tools.",
            called=True,
            timeout=False,
            runtime=0.0,
            error=None,
            call_id=self._generate_call_id(),
            raw_output=None,
            documents=[],
            tool_name="no_browse",
        )

    def _format_output(self, output: ToolOutput) -> str:
        return output.output

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL to browse"}},
            "required": ["url"],
        }


class AutoReasonSearchWorkflow(BaseWorkflow):
    _default_configuration_path = os.path.join(
        os.path.dirname(__file__), "auto_search_sft.yaml"
    )

    class Configuration(BaseWorkflowConfiguration):

        tool_parser: str

        search_tool_name: str = "serper"

        # Separate generation client (SFT model)
        search_agent_base_url: Optional[str] = None
        search_agent_model_name: str = "dr-tulu/DR-Tulu-8B"
        search_agent_tokenizer_name: str = "Qwen/Qwen3-8B"
        search_agent_api_key: str = "dummy-key"
        search_agent_max_tokens: int = 32000
        search_agent_temperature: float = 0.7
        search_agent_max_tool_calls: int = 10
        # OpenAI GPT-5.x reasoning.effort: none, low, medium, high, xhigh
        search_agent_reasoning_effort: Optional[str] = None
        # Stop sequence(s) for the search agent's generation, e.g. ["</think>"]
        # to stop right after the first thinking block (no tool call executed).
        search_agent_stop: Optional[List[str]] = None
        # Number of top logprobs to return per generated token (vLLM only).
        # Populates ``full_traces.token_logprobs`` when set.
        search_agent_logprobs: Optional[int] = None

        use_browse_agent: bool = False
        browse_agent_base_url: Optional[str] = None
        browse_agent_model_name: str = "Qwen/Qwen3-8B"
        browse_agent_tokenizer_name: str = "Qwen/Qwen3-8B"
        browse_agent_api_key: str = "dummy-key"
        browse_agent_max_tokens: int = 32000
        browse_agent_temperature: float = 0.3
        browse_agent_reasoning_effort: Optional[str] = None

        # MCP transport configuration
        mcp_transport_type: str = "StreamableHttpTransport"
        mcp_executable: Optional[str] = None
        mcp_port: int = 8000

        # Search configuration
        number_documents_to_search: int = 10
        search_timeout: int = 60

        # Browse configuration
        browse_tool_name: Optional[str] = "crawl4ai"
        browse_timeout: int = 60
        browse_max_pages_to_fetch: int = 10
        browse_context_char_length: int = 6000
        crawl4ai_use_docker_version: bool = False
        crawl4ai_use_ai2_config: bool = False

        prompt_version: str = "v20250907"

        # ask_user → userville.UserSimulator (optional). Requires OPENAI_API_KEY for the simulator LLM.
        ask_user_simulator_prompt: Optional[str] = None
        ask_user_simulator_model: Optional[str] = None
        # When True and ask_user_simulator_prompt is unset, use DEFAULT_ASK_USER_SIMULATOR_PROMPT.
        ask_user_use_default_simulator_prompt: bool = False
        # Optional JSON (multi-object file ok): first object at setup; per-call ``template_record`` overrides for batch eval.
        query_template_jsonl: Optional[str] = None
        # If True, ignore chat/UI ``problem`` and use the template record's ``query`` as the task question (history cleared).
        use_query_from_template: bool = False

    def before_launch_check(self) -> None:
        """Check if MCP server and vLLM servers are running, launch if needed."""
        cfg = self.configuration
        if cfg is None:
            return

        console = Console()

        console.print()
        console.print(Panel.fit("🔍 Service Check", style="bold cyan"))
        console.print()

        # Check MCP server
        mcp_port = getattr(cfg, "mcp_port", 8000)
        if not check_port(mcp_port):
            console.print(
                f"[yellow]⚠[/yellow]  MCP server is not running on port [bold]{mcp_port}[/bold]"
            )
            if Confirm.ask("Launch MCP server?"):
                process = launch_mcp_server(mcp_port, self.logger)
                if process:
                    self._launched_processes.append(process)
                    console.print(
                        f"[green]✓[/green]  MCP server launched on port {mcp_port}"
                    )
                else:
                    console.print(
                        "[red]✗[/red]  Failed to start MCP server", style="bold red"
                    )
                    raise RuntimeError(
                        "Failed to start MCP server. Please launch it manually."
                    )
            else:
                console.print("[red]✗[/red]  MCP server is required", style="bold red")
                raise RuntimeError(
                    "MCP server is required. Please launch it manually or allow automatic launch."
                )
        else:
            console.print(
                f"[green]✓[/green]  MCP server is running on port [bold]{mcp_port}[/bold]"
            )

        # Check search agent vLLM server
        search_base_url = getattr(cfg, "search_agent_base_url", None)
        if search_base_url:
            port = extract_port_from_url(search_base_url)
            if port and not check_port(port):
                console.print(
                    f"[yellow]⚠[/yellow]  Search agent vLLM server is not running on port [bold]{port}[/bold]"
                )
                search_model = getattr(cfg, "search_agent_model_name", None)
                if search_model:
                    if Confirm.ask(
                        f"Launch vLLM server for [cyan]{search_model}[/cyan] on port {port}?"
                    ):
                        process = launch_vllm_server(
                            search_model, port, gpu_id=0, logger=self.logger
                        )
                        if process:
                            self._launched_processes.append(process)
                            console.print(
                                f"[green]✓[/green]  vLLM server launched for {search_model} on port {port}"
                            )
                        else:
                            console.print(
                                f"[yellow]⚠[/yellow]  Failed to start vLLM server. Manual launch command:"
                            )
                            console.print(
                                f"   [dim]CUDA_VISIBLE_DEVICES=0 vllm serve {search_model} --port {port} --dtype auto --max-model-len 40960[/dim]"
                            )
                    else:
                        console.print(f"[blue]💡[/blue]  Manual launch command:")
                        console.print(
                            f"   [dim]CUDA_VISIBLE_DEVICES=0 vllm serve {search_model} --port {port} --dtype auto --max-model-len 40960[/dim]"
                        )
            elif port:
                console.print(
                    f"[green]✓[/green]  Search agent vLLM server is accessible on port [bold]{port}[/bold]"
                )

        # Check browse agent vLLM server if enabled
        use_browse_agent = getattr(cfg, "use_browse_agent", False)
        if use_browse_agent:
            browse_base_url = getattr(cfg, "browse_agent_base_url", None)
            if browse_base_url:
                port = extract_port_from_url(browse_base_url)
                if port and not check_port(port):
                    console.print(
                        f"[yellow]⚠[/yellow]  Browse agent vLLM server is not running on port [bold]{port}[/bold]"
                    )
                    browse_model = getattr(cfg, "browse_agent_model_name", None)
                    if browse_model:
                        if Confirm.ask(
                            f"Launch vLLM server for [cyan]{browse_model}[/cyan] on port {port}?"
                        ):
                            process = launch_vllm_server(
                                browse_model, port, gpu_id=1, logger=self.logger
                            )
                            if process:
                                self._launched_processes.append(process)
                                console.print(
                                    f"[green]✓[/green]  vLLM server launched for {browse_model} on port {port}"
                                )
                            else:
                                console.print(
                                    f"[yellow]⚠[/yellow]  Failed to start vLLM server. Manual launch command:"
                                )
                                console.print(
                                    f"   [dim]CUDA_VISIBLE_DEVICES=1 vllm serve {browse_model} --port {port} --dtype auto --max-model-len 40960[/dim]"
                                )
                        else:
                            console.print(f"[blue]💡[/blue]  Manual launch command:")
                            console.print(
                                f"   [dim]CUDA_VISIBLE_DEVICES=1 vllm serve {browse_model} --port {port} --dtype auto --max-model-len 40960[/dim]"
                            )
                elif port:
                    console.print(
                        f"[green]✓[/green]  Browse agent vLLM server is accessible on port [bold]{port}[/bold]"
                    )

        console.print()
        console.print(Panel.fit("✅ Service Check Complete", style="bold green"))
        console.print()

    def setup_components(
        self,
        mcp_transport_type: Optional[str] = "StreamableHttpTransport",
        mcp_executable: Optional[str] = None,
        mcp_port: Optional[int] = 8000,
    ) -> None:
        cfg = self.configuration
        assert cfg is not None
        # print(cfg)

        # Allow configuration overrides for MCP settings
        if getattr(cfg, "mcp_transport_type", None):
            mcp_transport_type = cfg.mcp_transport_type
        if getattr(cfg, "mcp_executable", None):
            mcp_executable = cfg.mcp_executable
        if getattr(cfg, "mcp_port", None) is not None:
            mcp_port = cfg.mcp_port

        # Search and browse tools (MCP-backed) with unified tool parser
        if cfg.search_tool_name == "serper":
            self.search_tool = SerperSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="snippet_search",  # <- to test this v20250824 model, we need to set the tool name in a hacky way.
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )

            self.search_tool2 = SerperSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="google_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.search_tool_name == "s2":
            self.search_tool = SemanticScholarSnippetSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="snippet_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )

            self.search_tool2 = SerperSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="google_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.search_tool_name == "s2-only":
            self.search_tool = SemanticScholarSnippetSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="snippet_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )

            self.search_tool2 = SemanticScholarSnippetSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="google_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        else:
            raise ValueError(f"Invalid search tool name: {cfg.search_tool_name}")

        if cfg.browse_tool_name == "serper":
            self.browse_tool = SerperBrowseTool(
                tool_parser=cfg.tool_parser,
                max_pages_to_fetch=cfg.browse_max_pages_to_fetch,
                timeout=cfg.browse_timeout,
                name="browse_webpage",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.browse_tool_name == "crawl4ai":
            self.browse_tool = Crawl4AIBrowseTool(
                tool_parser=cfg.tool_parser,
                max_pages_to_fetch=cfg.browse_max_pages_to_fetch,
                timeout=cfg.browse_timeout,
                name="browse_webpage",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
                context_chars=cfg.browse_context_char_length,
                use_docker_version=cfg.crawl4ai_use_docker_version,
                use_ai2_config=cfg.crawl4ai_use_ai2_config,
            )
        elif cfg.browse_tool_name == "jina":
            self.browse_tool = JinaBrowseTool(
                tool_parser=cfg.tool_parser,
                timeout=cfg.browse_timeout,
                name="browse_webpage",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.browse_tool_name is None:
            self.browse_tool = NoBrowseTool(
                tool_parser=cfg.tool_parser,
                name="browse_webpage",
            )
        else:
            raise ValueError(f"Invalid browse tool name: {cfg.browse_tool_name}")
        print("Using browse tool: ", self.browse_tool)

        if cfg.use_browse_agent:
            with LLMToolClient(
                model_name=cfg.browse_agent_model_name,
                tokenizer_name=cfg.browse_agent_tokenizer_name,
                base_url=cfg.browse_agent_base_url,
                api_key=cfg.browse_agent_api_key,
                generation_config=GenerationConfig(
                    temperature=cfg.browse_agent_temperature,
                    max_tokens=cfg.browse_agent_max_tokens,
                    reasoning_effort=cfg.browse_agent_reasoning_effort,
                ),
            ) as client:
                self.browse_agent = WebPageReaderAgentV2(client=client).as_tool(
                    max_tokens=cfg.browse_agent_max_tokens,
                    temperature=cfg.browse_agent_temperature,
                )
                self.composed_browse_tool = ChainedTool(
                    [self.browse_tool, self.browse_agent],
                    name="browse_webpage",
                    tool_parser=cfg.tool_parser,
                    output_formatting="last",
                )
        else:
            self.composed_browse_tool = self.browse_tool

        self._query_template_first_row: Optional[dict] = None
        self._query_template_jsonl_path: Optional[str] = None
        qt_path = getattr(cfg, "query_template_jsonl", None)
        if qt_path:
            p = Path(qt_path)
            if not p.is_absolute():
                p = _agent_dir / p
            self._query_template_jsonl_path = str(p)
            self._query_template_first_row = _load_query_template_jsonl_first(p)

        self.ask_user_tool = AskUserTool(
            tool_parser=cfg.tool_parser,
            name="ask_user",
            user_simulator=None,
        )

        with LLMToolClient(
            model_name=cfg.search_agent_model_name,
            tokenizer_name=cfg.search_agent_tokenizer_name,
            base_url=cfg.search_agent_base_url,
            api_key=cfg.search_agent_api_key,
            generation_config=GenerationConfig(
                temperature=cfg.search_agent_temperature,
                max_tokens=cfg.search_agent_max_tokens,
                reasoning_effort=cfg.search_agent_reasoning_effort,
            ),
        ) as client:
            self.search_agent = SearchAgent(
                client=client,
                tools=[
                    self.search_tool,
                    self.search_tool2,
                    self.composed_browse_tool,
                    self.ask_user_tool,
                ],
                prompt_version=cfg.prompt_version,
            )
            self.answer_agent = AnswerAgent(
                client=client,
                prompt_version=cfg.prompt_version,
            )

    def _sync_ask_user_simulator(
        self, problem: str, template_row: Optional[dict] = None
    ) -> None:
        """Attach a fresh UserSimulator (template row or first row of query_template JSONL or config)."""
        from dr_agent.userville import UserSimulator

        cfg = self.configuration
        assert cfg is not None

        sim_prompt: Optional[str] = None
        row = template_row
        if row is None:
            row = getattr(self, "_query_template_first_row", None)
        if row is not None and (row.get("clarified_query") or "").strip():
            sim_prompt = _user_simulator_prompt_from_query_template_row(problem, row)
        else:
            initial_q = ""
            if isinstance(row, dict):
                initial_q = (row.get("query") or "").strip()
            if not initial_q:
                initial_q = (problem or "").strip()
            if initial_q:
                sim_prompt = _user_simulator_prompt_initial_only(initial_q)

        if sim_prompt is None:
            sim_prompt = cfg.ask_user_simulator_prompt
            if cfg.ask_user_use_default_simulator_prompt and not sim_prompt:
                sim_prompt = DEFAULT_ASK_USER_SIMULATOR_PROMPT

        self.ask_user_tool.user_simulator = (
            UserSimulator(sim_prompt, model=cfg.ask_user_simulator_model)
            if sim_prompt
            else None
        )

        if os.environ.get("ASK_USER_DEBUG", "").strip():
            print(
                "[ask_user] template_file="
                f"{getattr(self, '_query_template_jsonl_path', None)} "
                f"first_row={bool(row)} "
                f"clarified_non_empty={bool(row and (row.get('clarified_query') or '').strip())} "
                f"sim_prompt_built={sim_prompt is not None} "
                f"user_simulator_set={self.ask_user_tool.user_simulator is not None}",
                flush=True,
            )

    async def __call__(
        self,
        problem: str,
        dataset_name: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        verbose: bool = True,
        search_callback: Optional[Any] = None,
        step_callback: Optional[Any] = None,
        template_record: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = self.configuration
        assert cfg is not None

        # :class:`AnswerAgent` requires a known benchmark tag; CLI/batch often omits it.
        resolved_dataset_name = dataset_name or "deep_research_bench"

        # Extract history and problem from messages if provided
        history = []
        if messages:
            # Find the last user message as the problem
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    last_user_idx = i
                    break

            if last_user_idx != -1:
                problem = messages[last_user_idx]["content"]
                history = messages[:last_user_idx]
            else:
                # Fallback if no user message found (shouldn't happen ideally)
                history = messages

        effective_row: Optional[dict] = None
        if template_record is not None:
            effective_row = dict(template_record)
        else:
            effective_row = getattr(self, "_query_template_first_row", None)

        if getattr(cfg, "use_query_from_template", False):
            row = effective_row
            tpl_q = (
                (row.get("query") or "").strip()
                if isinstance(row, dict)
                else ""
            )
            if tpl_q:
                problem = tpl_q
                history = []
            elif os.environ.get("ASK_USER_DEBUG", "").strip():
                print(
                    "[workflow] use_query_from_template=True but no ``query`` in template; "
                    "using problem from request.",
                    flush=True,
                )

        # import litellm
        # litellm._turn_on_debug()

        # Set the question for the browse agent
        # TODO: This is a bit hectic and hacky, but it works for now
        # The problem: it uses a bad way to enable the runtime dynamics
        if isinstance(self.composed_browse_tool, ChainedTool):
            browse_tool = self.composed_browse_tool.tools[0]
            browse_tool.bm25_query = problem
            browse_agent = self.composed_browse_tool.tools[-1]
            browse_agent.agent.question = problem
        else:
            browse_tool = self.composed_browse_tool
            browse_tool.bm25_query = problem

        self._sync_ask_user_simulator(problem, template_row=effective_row)

        search_kwargs: Dict[str, Any] = dict(
            question=problem,
            dataset_name=resolved_dataset_name,
            history=history,
            max_tokens=cfg.search_agent_max_tokens,
            temperature=cfg.search_agent_temperature,
            max_tool_calls=cfg.search_agent_max_tool_calls,
            verbose=verbose,
            on_step_callback=step_callback,
        )
        if cfg.search_agent_stop:
            search_kwargs["stop"] = cfg.search_agent_stop
        if cfg.search_agent_logprobs is not None:
            search_kwargs["logprobs"] = cfg.search_agent_logprobs

        results = await self.search_agent(**search_kwargs)

        if search_callback:
            if asyncio.iscoroutinefunction(search_callback):
                await search_callback(results)
            else:
                search_callback(results)

        browsed_links = []
        searched_links = []
        total_tool_calls = 0
        failed_tool_calls = 0
        for tool_output in results.tool_calls:
            total_tool_calls += 1
            if tool_output.error != "":
                failed_tool_calls += 1

            if tool_output.tool_name in ["snippet_search", "google_search"]:
                searched_links.extend(
                    [document.url for document in tool_output.documents]
                )

            if tool_output.tool_name == "browse_webpage":
                if isinstance(self.composed_browse_tool, ChainedTool):
                    if tool_output.raw_output is None:
                        continue
                    if chained_tool_outputs := tool_output.raw_output.get(
                        "tool_outputs"
                    ):
                        for document in chained_tool_outputs[0].documents:
                            if document.url:
                                browsed_links.append(document.url)
                else:
                    if hasattr(tool_output, "documents"):
                        for document in tool_output.documents:
                            if document.url:
                                browsed_links.append(document.url)
                    else:
                        print(
                            f"Warning: browse_webpage tool output has no documents: {tool_output}"
                        )

        browsed_links = list(set(browsed_links))
        searched_links = list(set(searched_links))

        return {
            "final_response": self.search_agent.postprocess_output(results),
            "full_traces": results,
            "browsed_links": browsed_links,
            "searched_links": searched_links,
            "total_tool_calls": total_tool_calls,
            "total_failed_tool_calls": failed_tool_calls,
        }


if __name__ == "__main__":
    AutoReasonSearchWorkflow.app()
