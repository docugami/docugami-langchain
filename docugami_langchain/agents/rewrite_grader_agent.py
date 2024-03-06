# Adapted with thanks from https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb

from typing import AsyncIterator, Optional

from langchain_core.runnables import Runnable, RunnableConfig

from docugami_langchain.agents.base import BaseDocugamiAgent
from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.params import RunnableParameters


class RewriteGraderRAGAgent(BaseDocugamiAgent[dict]):
    """
    Agent that implements agentic RAG with the following additional optimizations:
    1. Query Rewriting
    2. Retrieval Grading
    """

    def params(self) -> RunnableParameters:
        raise NotImplementedError()

    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """
        raise NotImplementedError()

    def run(  # type: ignore[override]
        self,
        question: str,
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[dict]:
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        question: str,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[dict]]:
        if not question:
            raise Exception("Input required: question")

        async for item in super().run_stream(
            question=question,
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[str],
        config: Optional[RunnableConfig] = None,
    ) -> list[dict]:
        return super().run_batch(
            inputs=[{"question": i} for i in inputs],
            config=config,
        )
