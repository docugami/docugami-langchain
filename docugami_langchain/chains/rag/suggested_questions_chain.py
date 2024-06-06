from operator import itemgetter
from typing import AsyncIterator, Optional, Union

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.history import chat_history_to_str
from docugami_langchain.output_parsers.line_separated_list import (
    LineSeparatedListOutputParser,
)
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter
from docugami_langchain.utils.documents import formatted_summaries


class SuggestedQuestionsChain(BaseDocugamiChain[list[str]]):
    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """

        return {
            "summaries": itemgetter("summaries") | RunnableLambda(formatted_summaries),
            "chat_history": itemgetter("chat_history"),
        } | super().runnable()

    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "summaries",
                    "SUMMARIES",
                    "Summaries of representative documents that can be searched to answer questions.",
                ),
                RunnableSingleParameter(
                    "chat_history",
                    "CHAT HISTORY",
                    "Previous chat messages that may provide additional context for further questions.",
                ),
            ],
            output=RunnableSingleParameter(
                "suggested_questions",
                "SUGGESTED QUESTIONS",
                "Some suggested questions that may be answered from a set of the type of document represented by the summaries "
                + "considering the rules and examples provided.",
            ),
            task_description="generates some questions a user may want to ask against against a set of documents",
            additional_instructions=[
                "- Base your questions only on the type of documents represented by the given document summaries",
                "- Ensure all the questions you ask are answerable from the data in the given summaries",
                "- Generate the most likely questions a user may ask, in order of relevance, no more than that.",
                "- If chat history is provided, consider that to determine the next set of questions the user is most likely to ask.",
                "- Generate suggested questions as a list, one question per line.",
            ],
            additional_runnables=[LineSeparatedListOutputParser()],
            stop_sequences=["\n\n", "<|eot_id|>"],
        )

    def run(  # type: ignore[override]
        self,
        summaries: list[Document],
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[list[str]]:
        if not summaries:
            raise Exception("Input required: summaries")

        return super().run(
            summaries=summaries,
            chat_history=chat_history_to_str(chat_history),
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        summaries: list[Document],
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[list[str]]]:
        if not summaries:
            raise Exception("Input required: summaries")

        async for item in super().run_stream(
            summaries=summaries,
            chat_history=chat_history_to_str(chat_history),
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, str]],
        config: Optional[RunnableConfig] = None,
        return_exceptions: bool = True,
    ) -> list[Union[list[str], Exception]]:
        return super().run_batch(
            inputs=[
                {
                    "summaries": i[0],
                    "chat_history": i[1],
                }
                for i in inputs
            ],
            config=config,
            return_exceptions=return_exceptions,
        )
