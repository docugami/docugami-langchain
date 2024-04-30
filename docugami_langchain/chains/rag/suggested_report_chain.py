from operator import itemgetter
from typing import AsyncIterator, Optional

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


class SuggestedReportChain(BaseDocugamiChain[list[str]]):
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
                    "Summaries of representative documents from a set of documents.",
                ),
                RunnableSingleParameter(
                    "chat_history",
                    "CHAT HISTORY",
                    "Previous chat messages that may provide additional context for the type of columns that should be generated.",
                ),
            ],
            output=RunnableSingleParameter(
                "suggested_report_columns",
                "SUGGESTED REPORT COLUMNS",
                "Suggested columns for an automatically generated report against documents similar to the ones provided.",
            ),
            task_description="suggests columns for an automatically generated report against a set of documents, given some summaries of representative documents from the set",
            additional_instructions=[
                "- Generate 'human-like' column labels, i.e. things a human familiar with this particular set of documents might want to know in a diagnostic report about this set of documents",
                "- Bias towards columns highly likely to be found in all or most of the documents.",
                "- Avoid columns that are highly likely to contain boilerplate or uninteresting information that is similar for all the documents.",
                "- Do not include Document Name or File Name in your list, since those are included automatically by the system.",
                "- Make sure the column names you generate are alphanumeric, containing no special characters or parentheses.",
                "- Generate up to 20 suggested columns as a list, one per line.",
            ],
            additional_runnables=[LineSeparatedListOutputParser()],
            stop_sequences=["\n\n", "<|eot_id|>"],
            key_finding_output_parse=False,  # set to False for streaming
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
    ) -> list[list[str]]:
        raise NotImplementedError()
