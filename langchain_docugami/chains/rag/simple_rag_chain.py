from operator import itemgetter
from typing import AsyncIterator, List, Optional

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStore

from langchain_docugami.chains.base import BaseDocugamiChain, TracedChainResponse
from langchain_docugami.chains.params import ChainParameters, ChainSingleParameter


class SimpleRAGChain(BaseDocugamiChain[str]):

    chunk_vectorstore: VectorStore

    def chain_params(self) -> ChainParameters:
        return ChainParameters(
            inputs=[
                ChainSingleParameter(
                    "question",
                    "QUESTION",
                    "Question asked by the user.",
                ),
                ChainSingleParameter(
                    "context",
                    "CONTEXT",
                    "Retrieved context, which should be used to answer the question.",
                ),
            ],
            output=ChainSingleParameter(
                "answer",
                "ANSWER",
                "Human readable answer to the question.",
            ),
            task_description="acts as an assistant for question-answering tasks.",
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
            "question": RunnablePassthrough(),
            "context": itemgetter("question")
            | self.chunk_vectorstore.as_retriever()
            | format_retrieved_docs,
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