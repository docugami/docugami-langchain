from operator import itemgetter
from typing import AsyncIterator, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

from docugami_langchain.chains.base import BaseDocugamiChain, TracedChainResponse
from docugami_langchain.chains.params import ChainParameters, ChainSingleParameter


class SimpleRAGChain(BaseDocugamiChain[str]):

    retriever: BaseRetriever

    def chain_params(self) -> ChainParameters:
        return ChainParameters(
            inputs=[
                ChainSingleParameter(
                    "context",
                    "CONTEXT",
                    "Retrieved context, which should be used to answer the question.",
                ),
                ChainSingleParameter(
                    "question",
                    "QUESTION",
                    "Question asked by the user.",
                ),
            ],
            output=ChainSingleParameter(
                "answer",
                "ANSWER",
                "Human readable answer to the question.",
            ),
            task_description="acts as an assistant for question-answering tasks",
            additional_instructions=[
                "- Use only the given pieces of retrieved context to answer the question, don't make up answers.",
                "- If you don't know the answer, just say that you don't know.",
                "- Use three sentences maximum and keep the answer concise.",
            ],
            key_finding_output_parse=False,  # set to False for streaming
        )

    def runnable(self) -> Runnable:
        """
        Custom runnable for this agent.
        """

        def format_retrieved_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        return {
            "context": itemgetter("question") | self.retriever | format_retrieved_docs,
            "question": RunnablePassthrough(),
        } | super().runnable()

    def run(  # type: ignore[override]
        self,
        question: str,
        config: Optional[dict] = None,
    ) -> str:
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
    ) -> AsyncIterator[TracedChainResponse[str]]:
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
    ) -> list[str]:
        return super().run_batch(
            inputs=[{"question": i} for i in inputs],
            config=config,
        )
