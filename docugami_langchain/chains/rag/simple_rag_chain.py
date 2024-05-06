from operator import itemgetter
from typing import AsyncIterator, Optional

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.chains.rag.models import ExtendedRAGResult
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter
from docugami_langchain.retrievers.fused_summary import FusedSummaryRetriever


class SimpleRAGChain(BaseDocugamiChain[ExtendedRAGResult]):

    retriever: FusedSummaryRetriever

    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "context",
                    "CONTEXT",
                    "Retrieved context, which should be used to answer the question.",
                ),
                RunnableSingleParameter(
                    "question",
                    "QUESTION",
                    "Question asked by the user.",
                ),
            ],
            output=RunnableSingleParameter(
                "answer",
                "ANSWER",
                "Human readable answer to the question, based on the given context.",
            ),
            task_description="acts as an assistant for question-answering tasks",
            additional_instructions=[
                "- Use only the given pieces of retrieved context to answer the question, don't make up answers.",
                "- If you cannot find the answer in the given context, just say that you don't know.",
                "- Your answer should be concise, up to three sentences long.",
            ],
            stop_sequences=["<|eot_id|>"],
            key_finding_output_parse=False,  # set to False for streaming
            include_output_instruction_suffix=True,
        )

    def run_rag(
        self, inputs: dict, config: Optional[RunnableConfig]
    ) -> ExtendedRAGResult:
        """
        Runs rag for the given question against the given context, and returns the result.
        """

        def format_retrieved_docs(docs: list[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        def list_retrieved_docs(docs: list[Document]) -> list[str]:
            return [doc.metadata[self.retriever.source_key] for doc in docs]

        context = inputs.get("context")
        question = inputs.get("question")

        answer = (
            super()
            .runnable()
            .invoke(
                {
                    "context": format_retrieved_docs(context),  # type: ignore
                    "question": question,
                },
                config,
            )
        )

        return {
            "answer": answer,
            "question": question,
            "sources": list_retrieved_docs(context),  # type: ignore
        }

    def runnable(self) -> Runnable:
        """
        Custom runnable for this agent.
        """

        return {
            "context": itemgetter("question") | self.retriever,
            "question": itemgetter("question"),
        } | RunnableLambda(self.run_rag)

    def run(  # type: ignore[override]
        self,
        question: str,
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[ExtendedRAGResult]:
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
    ) -> AsyncIterator[TracedResponse[ExtendedRAGResult]]:
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
    ) -> list[ExtendedRAGResult]:
        return super().run_batch(
            inputs=[
                {
                    "question": i,
                }
                for i in inputs
            ],
            config=config,
        )
