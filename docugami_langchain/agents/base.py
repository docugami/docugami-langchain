from abc import abstractmethod
from typing import AsyncIterator

from langchain_core.tracers.context import collect_runs
from langgraph.graph import END

from docugami_langchain.base_runnable import BaseRunnable, T, TracedResponse


class BaseDocugamiAgent(BaseRunnable[T]):
    """
    Base class with common functionality for various chains.
    """

    @abstractmethod
    async def run_stream(self, **kwargs) -> AsyncIterator[TracedResponse[T]]:  # type: ignore
        config, kwargs_dict = self._prepare_run_args(kwargs)

        with collect_runs() as cb:
            last_answer = None
            async for output in self.runnable().astream(
                input=kwargs_dict,
                config=config,  # type: ignore
            ):
                if not isinstance(output, dict):
                    # agent step-wise streaming yields dictionaries keyed by node name
                    # Ref: https://python.langchain.com/docs/langgraph#streaming-node-output
                    raise Exception("Expected dictionary output from agent streaming")

                if not len(output.keys()) == 1:
                    raise Exception(
                        "Expected output from one node at a time in step-wise agent streaming output"
                    )

                key = list(output.keys())[0]
                last_answer = output[key]
                yield TracedResponse[T](value=last_answer)

            # yield the final result with the run_id
            if cb.traced_runs:
                run_id = str(cb.traced_runs[0].id)
                yield TracedResponse[T](
                    run_id=run_id,
                    value=last_answer,  # type: ignore
                )
