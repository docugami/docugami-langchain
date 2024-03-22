from abc import abstractmethod
from typing import AsyncIterator, Optional

from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tracers.context import collect_runs
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from docugami_langchain.agents.models import AgentState, CitedAnswer, StepState
from docugami_langchain.base_runnable import BaseRunnable, TracedResponse

THINKING = "Thinking..."


class BaseDocugamiAgent(BaseRunnable[AgentState]):
    """
    Base class with common functionality for various chains.
    """

    tools: list[BaseTool] = []

    @abstractmethod
    def parse_final_answer(self, text: str) -> str: ...

    def execute_tool(
        self,
        state: AgentState,
        config: Optional[RunnableConfig],
    ) -> AgentState:
        # Get the most recent tool invocation (added by the agent) and execute it
        inv_model = state.get("tool_invocation")
        if not inv_model:
            raise Exception(f"No tool invocation in model: {state}")

        previous_steps = state.get("intermediate_steps")
        if previous_steps and any(
            [s for s in previous_steps if s.invocation == inv_model]
        ):
            output = "This tool has been invoked before with identical inputs. Please refer to the previous invocation results or try different inputs."
        else:
            tool_executor = ToolExecutor(self.tools)
            output = tool_executor.invoke(
                ToolInvocation(  # LangChain version of the Invocation object
                    tool=inv_model.tool_name,
                    tool_input=inv_model.tool_input,
                ),
                config,
            )

        step = StepState(
            invocation=inv_model,
            output=str(output),
        )
        return {"intermediate_steps": [step]}  # appended

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
            async for output in self.runnable().astream_log(
                input=kwargs_dict,
                config=config,
                include_types=["llm"],
            ):
                for op in output.ops:
                    op_path = op.get("path", "")
                    op_value = op.get("value", "")
                    if not final_streaming_started and op_path == "/streamed_output/-":
                        # Restart token stream for each interim step
                        current_step_token_stream = ""
                        if not isinstance(op_value, dict):
                            # Agent step-wise streaming yields dictionaries keyed by node name
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
                        yield TracedResponse[AgentState](value=last_response_value)
                    elif op_path.startswith("/logs/") and op_path.endswith(
                        "/streamed_output/-"
                    ):
                        # Because we chose to only include LLMs, these are LLM tokens
                        if isinstance(op_value, AIMessageChunk):
                            current_step_token_stream += str(op_value.content)

                            final_answer = self.parse_final_answer(
                                current_step_token_stream
                            )

                            if final_answer:
                                if not final_streaming_started:
                                    # Set final streaming started once as soon as we see the final
                                    # answer action in the token stream
                                    final_streaming_started = bool(final_answer)
                                else:
                                    # Start streaming the final answer, no more interim steps
                                    last_response_value = AgentState(
                                        chat_history=[],
                                        question="",
                                        tool_invocation=None,
                                        intermediate_steps=[],
                                        cited_answer=CitedAnswer(
                                            source=self.__class__.__name__,
                                            answer=final_answer,
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
