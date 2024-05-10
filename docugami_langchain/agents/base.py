from abc import abstractmethod
from typing import AsyncIterator, Optional

from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.context import collect_runs
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from docugami_langchain.agents.models import (
    AgentState,
    Citation,
    CitedAnswer,
    Invocation,
    StepState,
)
from docugami_langchain.base_runnable import BaseRunnable, TracedResponse
from docugami_langchain.tools.common import BaseDocugamiTool

THINKING = "Thinking..."


class BaseDocugamiAgent(BaseRunnable[AgentState]):
    """
    Base class with common functionality for various chains.
    """

    tools: list[BaseDocugamiTool] = []

    @abstractmethod
    def streamable_node_names(self) -> list[str]:
        """Node names in the graph from which token by token output should be streamed."""
        ...

    @abstractmethod
    def parse_final_answer_from_streamed_output(self, text: str) -> str:
        """Given output stream from a streamable node, parses out the final answer (e.g. past a delimiter)."""
        ...

    def execute_tool(
        self,
        state: AgentState,
        config: Optional[RunnableConfig],
    ) -> AgentState:
        """
        Gets the most recent tool invocation (added by the agent) and execute it.
        """

        inv_model = state.get("tool_invocation")
        if not inv_model:
            raise Exception(f"No tool invocation in model: {state}")

        previous_steps = state.get("intermediate_steps") or []
        if previous_steps and any(
            [s for s in previous_steps if s.invocation == inv_model]
        ):
            tool_output = CitedAnswer(
                source=inv_model.tool_name,
                answer="This tool has been invoked before with identical inputs. Please try different inputs or a different tool, after reconsidering previous thoughts and observations. "
                + "Be careful you don't get stuck in a loop.",
            )
        else:
            tool_executor = ToolExecutor(self.tools)
            tool_output = tool_executor.invoke(
                ToolInvocation(  # LangChain version of the Invocation object
                    tool=inv_model.tool_name,
                    tool_input=inv_model.tool_input,
                ),
                config,
            )
            if not isinstance(tool_output, CitedAnswer):
                raise Exception(f"Invalid output from tool executor: {tool_output}")

        step = StepState(
            invocation=inv_model,
            output=tool_output.answer,
            citations=tool_output.citations,
        )
        return {"intermediate_steps": previous_steps + [step]}

    def invocation_answer(
        self,
        invocation: Invocation,
        answer_source: str,
    ) -> AgentState:
        """
        Builds a human readable interim answer from a tool invocation.
        """

        tool_name = invocation.tool_name
        tool_input = invocation.tool_input
        if tool_name and tool_input:
            busy_text = THINKING
            match = [t for t in self.tools if t.name.lower() == tool_name]
            if match:
                busy_text = match[0].to_human_readable(invocation)

        return {
            "tool_invocation": invocation,
            "cited_answer": CitedAnswer(
                source=answer_source,
                answer=busy_text,  # Show the user interim output.
            ),
        }

    def run(  # type: ignore[override]
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[AgentState]:
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            chat_history=chat_history,
            config=config,
        )

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, list[tuple[str, str]]]],
        config: Optional[RunnableConfig] = None,
    ) -> list[AgentState]:
        return super().run_batch(
            inputs=[
                {
                    "question": i[0],
                    "chat_history": i[1],
                }
                for i in inputs
            ],
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[AgentState]]:
        if not question:
            raise Exception("Input required: question")

        config, kwargs_dict = self._prepare_run_args(
            {
                "question": question,
                "chat_history": chat_history,
            }
        )

        with collect_runs() as cb:
            last_response_value = None
            current_step_token_stream = ""
            final_streaming_started = False
            citations: list[Citation] = []
            async for event in self.runnable().astream_events(
                input=kwargs_dict, config=config, version="v1"
            ):
                event_key = event.get("event")
                event_name = event.get("name")
                event_data = event.get("data")
                if event_data:
                    if event_name in self.streamable_node_names():
                        if event_key == "on_chain_start":
                            # Restart token stream every time a streamable node starts
                            current_step_token_stream = ""
                            final_streaming_started = True
                        elif event_key == "on_chain_end":
                            # Yield the completed output when a streamable node finishes
                            last_response_value = event_data.get("output")
                            if last_response_value:
                                yield TracedResponse[AgentState](
                                    value=last_response_value
                                )
                    elif event_name == "execute_tool":
                        if event_key == "on_chain_end":
                            state: Optional[AgentState] = event_data.get("output")
                            if state:
                                answer = state.get("cited_answer")
                                citations = answer.citations if answer else []
                    elif event_key == "on_chat_model_stream":
                        chunk = event_data.get("chunk")
                        if isinstance(chunk, AIMessageChunk):
                            current_step_token_stream += str(chunk.content)
                            final_answer = self.parse_final_answer_from_streamed_output(
                                current_step_token_stream
                            )

                            if final_answer:
                                # Source the answer from the last step, if any
                                if not final_streaming_started:
                                    # Set final streaming started once as soon as we see the final answer
                                    final_streaming_started = bool(final_answer)
                                else:
                                    # Start streaming the final answer, no more interim steps
                                    last_response_value = AgentState(
                                        chat_history=[],
                                        question="",
                                        tool_invocation=None,
                                        cited_answer=CitedAnswer(
                                            source=self.__class__.__name__,
                                            answer=final_answer,
                                            citations=citations,
                                            is_final=True,
                                        ),
                                    )
                                    yield TracedResponse[AgentState](
                                        value=last_response_value
                                    )

            # Yield the final result with the run_id
            if last_response_value:
                if "cited_answer" in last_response_value:
                    last_response_value["cited_answer"].is_final = True

            if cb.traced_runs:
                run_id = str(cb.traced_runs[0].id)
                yield TracedResponse[AgentState](
                    run_id=run_id,
                    value=last_response_value,  # type: ignore
                )
