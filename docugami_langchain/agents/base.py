from abc import abstractmethod
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessageChunk
from langchain_core.tracers.context import collect_runs

from docugami_langchain.base_runnable import BaseRunnable, T, TracedResponse
from docugami_langchain.output_parsers.soft_react_json_single_input import (
    FINAL_ANSWER_ACTION,
)


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
