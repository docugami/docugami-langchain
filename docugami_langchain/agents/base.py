from abc import abstractmethod
from typing import Any, AsyncIterator

from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.tracers.context import collect_runs

from docugami_langchain.base_runnable import BaseRunnable, T, TracedResponse
from docugami_langchain.output_parsers.soft_react_json_single_input import (
    FINAL_ANSWER_ACTION,
)


def chat_history_to_messages(chat_history: list[tuple[str, str]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []

    if chat_history:
        for human, ai in chat_history:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=f"{ai}"))
    return messages


def chat_history_to_str(chat_history: list[tuple[str, str]]) -> str:
    messages: str = ""

    if chat_history:
        for human, ai in chat_history:
            messages += f"Human: {human}\n"
            messages += f"AI: {ai}\n"
    return "\n" + messages


def format_log_to_str(
    intermediate_steps: list[tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = ""
    if intermediate_steps:
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts


def render_text_description(tools: list[BaseTool]) -> str:
    """
    Copied from https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/tools/render.py
    to avoid taking a dependency on the entire langchain library

    Render the tool name and description in plain text.

    Output will be in the format of:

    .. code-block:: markdown

        search: This tool is used for search
        calculator: This tool is used for math
    """
    tool_strings = []
    for tool in tools:
        tool_strings.append(f"- {tool.name}: {tool.description}")
    return "\n".join(tool_strings)


def render_text_description_and_args(tools: list[BaseTool]) -> str:
    """
    Copied from https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/tools/render.py
    to avoid taking a dependency on the entire langchain library

    Render the tool name, description, and args in plain text.

    Output will be in the format of:

    .. code-block:: markdown

        search: This tool is used for search, args: {"query": {"type": "string"}}
        calculator: This tool is used for math, args: {"expression": {"type": "string"}}
    """
    tool_strings = []
    for tool in tools:
        args_schema = str(tool.args)
        tool_strings.append(f"- {tool.name}: {tool.description}, args: {args_schema}")
    return "\n".join(tool_strings)


class BaseDocugamiAgent(BaseRunnable[T]):
    """
    Base class with common functionality for various chains.
    """

    @staticmethod
    @abstractmethod
    def to_human_readable(state: T) -> str: ...

    @abstractmethod
    def create_finish_state(self, content: str) -> T: ...

    @abstractmethod
    async def run_stream(self, **kwargs: Any) -> AsyncIterator[TracedResponse[T]]:  # type: ignore
        config, kwargs_dict = self._prepare_run_args(kwargs)

        with collect_runs() as cb:
            last_response_value = None
            current_step_token_stream = ""
            final_streaming_started = False
            async for output in self.runnable().astream_log(
                input=kwargs_dict,
                config=config,
                include_types=["llm"],
            ):
                for op in output.ops:
                    op_path = op.get("path", "")
                    op_value = op.get("value", "")
                    if not final_streaming_started and op_path == "/streamed_output/-":
                        # restart token stream for each interim step
                        current_step_token_stream = ""
                        if not isinstance(op_value, dict):
                            # agent step-wise streaming yields dictionaries keyed by node name
                            # Ref: https://python.langchain.com/docs/langgraph#streaming-node-output
                            raise Exception(
                                "Expected dictionary output from agent streaming"
                            )

                        if not len(op_value.keys()) == 1:
                            raise Exception(
                                "Expected output from one node at a time in step-wise agent streaming output"
                            )

                        key = list(op_value.keys())[0]
                        last_response_value = op_value[key]
                        yield TracedResponse[T](value=last_response_value)
                    elif op_path.startswith("/logs/") and op_path.endswith(
                        "/streamed_output/-"
                    ):
                        # because we chose to only include LLMs, these are LLM tokens
                        if isinstance(op_value, AIMessageChunk):
                            current_step_token_stream += str(op_value.content)

                            if not final_streaming_started:
                                # set final streaming started once as soon as we see the final
                                # answer action in the token stream
                                final_streaming_started = (
                                    FINAL_ANSWER_ACTION in current_step_token_stream
                                )

                            if final_streaming_started:
                                # start streaming the final answer, we are done with intermediate steps
                                final_answer = (
                                    str(current_step_token_stream)
                                    .split(FINAL_ANSWER_ACTION)[-1]
                                    .strip()
                                )
                                if final_answer:
                                    # start streaming the final answer, no more interim steps
                                    last_response_value = self.create_finish_state(
                                        final_answer
                                    )
                                    yield TracedResponse[T](value=last_response_value)

            # yield the final result with the run_id
            if cb.traced_runs:
                run_id = str(cb.traced_runs[0].id)
                yield TracedResponse[T](
                    run_id=run_id,
                    value=last_response_value,  # type: ignore
                )
