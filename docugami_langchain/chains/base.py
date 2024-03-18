from abc import abstractmethod
from typing import Any, AsyncIterator

from langchain_core.runnables.utils import AddableDict
from langchain_core.tracers.context import collect_runs

from docugami_langchain.base_runnable import BaseRunnable, T, TracedResponse


class BaseDocugamiChain(BaseRunnable[T]):
    """
    Base class with common functionality for various chains.
    """

    @abstractmethod
    async def run_stream(self, **kwargs: Any) -> AsyncIterator[TracedResponse[T]]:  # type: ignore[override, misc]
        config, kwargs_dict = self._prepare_run_args(kwargs)

        with collect_runs() as cb:
            incremental_answer = None
            async for chunk in self.runnable().astream(
                input=kwargs_dict,
                config=config,  # type: ignore
            ):
                if isinstance(chunk, dict):
                    chunk = AddableDict(chunk)

                if not incremental_answer:
                    incremental_answer = chunk
                else:
                    incremental_answer += chunk

                yield TracedResponse[T](value=incremental_answer)

            # yield the final result with the run_id
            if cb.traced_runs:
                run_id = str(cb.traced_runs[0].id)
                yield TracedResponse[T](
                    run_id=run_id,
                    value=incremental_answer,  # type: ignore
                )
