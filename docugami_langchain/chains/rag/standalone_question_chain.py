from typing import AsyncIterator, Optional

from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableConfig,
    RunnableLambda,
)

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.history import chat_history_to_str
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class StandaloneQuestionChain(BaseDocugamiChain[str]):

    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """
        noop = RunnableLambda(lambda x: x["question"])

        # Rewrite only if chat history is provided
        return RunnableBranch(
            (
                lambda x: len(x["chat_history"]) > 0,
                super().runnable(),
            ),
            noop,
        )

    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "chat_history",
                    "CHAT HISTORY",
                    "Previous chat messages that may provide additional information about this question.",
                ),
                RunnableSingleParameter(
                    "question",
                    "QUESTION",
                    "A question from the user.",
                ),
            ],
            output=RunnableSingleParameter(
                "standalone_question",
                "STANDALONE_QUESTION",
                "A standalone version of the question (not an answer), re-written to incorporate additional information from the chat history",
            ),
            task_description="rewrites a question as a standalone question, incorporating additional information from the chat history",
            additional_instructions=[
                "- The generated standalone question will be used to search for relevant chunks within a set of documents that may answer the question.",
                "- Focus on the chat history immediately preceding the question.",
                "- Produce only the requested standalone question, no other commentary before or after.",
                "- Do NOT try to answer the question. Just rewrite the question as instructed."
                "- Never say you cannot do this. If all else fails, just repeat the given question without rewriting it.",
            ],
            stop_sequences=["CHAT HISTORY:", "QUESTION:"],
            key_finding_output_parse=False,  # set to False for streaming
        )

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
