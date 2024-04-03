from operator import itemgetter
from typing import AsyncIterator, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableConfig

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.history import chat_history_to_str
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class SimpleRAGChain(BaseDocugamiChain[str]):

    retriever: BaseRetriever

    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "context",
                    "CONTEXT",
                    "Retrieved context, which should be used to answer the question.",
                ),
                RunnableSingleParameter(
                    "chat_history",
                    "CHAT HISTORY",
                    "Previous chat messages that may provide additional context for this question.",
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
                "Human readable answer to the question.",
            ),
            task_description="acts as an assistant for question-answering tasks",
            additional_instructions=[
                "- Use only the given pieces of retrieved context to answer the question, don't make up answers.",
                "- If you don't know the answer, just say that you don't know.",
                "- Your answer should be concise, up to three sentences long.",
            ],
            stop_sequences=[],
            key_finding_output_parse=False,  # set to False for streaming
        )

    def runnable(self) -> Runnable:
        """
        Custom runnable for this agent.
        """

        def format_retrieved_docs(docs: list[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        return {
            "context": itemgetter("question") | self.retriever | format_retrieved_docs,
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question"),
        } | super().runnable()

    def run(  # type: ignore[override]
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[str]:
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            chat_history=chat_history_to_str(chat_history),
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[str]]:
        if not question:
            raise Exception("Input required: question")

        async for item in super().run_stream(
            question=question,
            chat_history=chat_history_to_str(chat_history),
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, list[tuple[str, str]]]],
        config: Optional[RunnableConfig] = None,
    ) -> list[str]:
        return super().run_batch(
            inputs=[
                {
                    "question": i[0],
                    "chat_history": chat_history_to_str(i[1]),
                }
                for i in inputs
            ],
            config=config,
        )
