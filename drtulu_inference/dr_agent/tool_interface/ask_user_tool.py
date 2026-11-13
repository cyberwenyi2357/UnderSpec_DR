"""
Human clarification tool: parses <call_tool name="ask_user">...</call_tool> and returns a user reply.

Can delegate to :class:`dr_agent.userville.UserEnv` (full stats + preference flow) or
:class:`dr_agent.userville.UserSimulator` (LLM-simulated user), or a custom ``respond_fn``.
"""

import inspect
import time
from typing import Any, Awaitable, Callable, Optional, Union

from .base import BaseTool
from .data_types import ToolInput, ToolOutput
from .tool_parsers import ToolCallParser


class AskUserTool(BaseTool):
    """
    Resolve ``ask_user`` tool calls by forwarding the parsed question to a user agent.

    Resolution order (first match wins):

    1. ``user_env`` — ``await user_env._handle_ask_question({"query": question})``,
       return ``observation`` (same path as native ``ask_question`` in RL).
       Requires :meth:`dr_agent.userville.UserEnv.init_env` to have run so
       ``user_simulator`` is initialized.
    2. ``user_simulator`` — ``await user_simulator.execute({"query": question})``.
    3. ``respond_fn`` — custom ``(question) -> str | Awaitable[str]``.
    4. Stub string when nothing is configured.

    You may set ``user_env`` / ``user_simulator`` after construction (e.g. once a
    :class:`~dr_agent.userville.UserEnv` has completed ``init_env``).
    """

    def __init__(
        self,
        tool_parser: Optional[Union[ToolCallParser, str]] = "v20250824",
        name: str = "ask_user",
        user_env: Optional[Any] = None,
        user_simulator: Optional[Any] = None,
        respond_fn: Optional[
            Callable[[str], Union[str, Awaitable[str]]]
        ] = None,
        default_reply: Optional[str] = None,
        timeout: int = 300,
        **kwargs: Any,
    ):
        super().__init__(
            tool_parser=tool_parser, name=name, timeout=timeout, **kwargs
        )
        self.user_env = user_env
        self.user_simulator = user_simulator
        self._respond_fn = respond_fn
        self._default_reply = default_reply

    def _stub_reply(self) -> str:
        if self._default_reply is not None:
            return self._default_reply
        return (
            "User interaction is not wired for ask_user in this run. "
            "Proceed with reasonable assumptions from search results, or finish with <answer>."
        )

    async def _resolve_reply(self, question: str) -> str:
        if self.user_env is not None:
            result = await self.user_env._handle_ask_question({"query": question})
            if isinstance(result, dict) and "observation" in result:
                return str(result["observation"])
            return str(result)

        if self.user_simulator is not None:
            return await self.user_simulator.execute(question)

        if self._respond_fn is not None:
            result = self._respond_fn(question)
            if inspect.isawaitable(result):
                return await result  # type: ignore[no-any-return]
            return result  # type: ignore[no-any-return]

        return self._stub_reply()

    async def __call__(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> ToolOutput:
        start = time.time()
        call_id = self._generate_call_id()

        if not isinstance(tool_input, str):
            return self._create_error_output(
                error_msg="ask_user expects string tool input",
                call_id=call_id,
                runtime=time.time() - start,
                output="",
            )

        call_info = self.parse_call(tool_input)
        if not call_info or not str(call_info.content).strip():
            return ToolOutput(
                tool_name=self.name,
                output="",
                called=False,
                error="Could not parse ask_user call from model output",
                timeout=False,
                runtime=time.time() - start,
                call_id=call_id,
                raw_output=None,
            )

        question = str(call_info.content).strip()
        try:
            reply = await self._resolve_reply(question)
        except Exception as e:
            return ToolOutput(
                tool_name=self.name,
                output="",
                called=True,
                error=str(e),
                timeout=False,
                runtime=time.time() - start,
                call_id=call_id,
                raw_output={"question": question},
            )

        return ToolOutput(
            tool_name=self.name,
            output=reply,
            called=True,
            error=None,
            timeout=False,
            runtime=time.time() - start,
            call_id=call_id,
            raw_output={"question": question},
        )

    def _format_output(self, output: ToolOutput) -> str:
        return output.output

    def _generate_tool_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Clarifying question shown to the user",
                }
            },
            "required": ["query"],
        }
