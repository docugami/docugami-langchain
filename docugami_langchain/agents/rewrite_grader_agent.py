# Adapted with thanks from https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb

from typing import AsyncIterator, Dict, Optional

from langchain_core.runnables import Runnable

from docugami_langchain.base_runnable import BaseRunnable, TracedResponse
from docugami_langchain.params import RunnableParameters


class RewriteGraderRAGAgent(BaseRunnable[Dict]):
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
        config: Optional[dict] = None,
    ) -> Dict:
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            config=config,
        )

    def run_stream(  # type: ignore[override]
        self,
        question: str,
        config: Optional[dict] = None,
    ) -> AsyncIterator[TracedResponse[Dict]]:
        if not question:
            raise Exception("Input required: question")

        return super().run_stream(
            question=question,
            config=config,
        )

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[str],
        config: Optional[dict] = None,
    ) -> list[Dict]:
        return super().run_batch(
            inputs=[{"question": i} for i in inputs],
            config=config,
        )
