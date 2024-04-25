from operator import itemgetter
from typing import Any, AsyncIterator, Optional

from langchain_community.utilities.sql_database import SQLDatabase
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
from docugami_langchain.utils.sql import get_table_info_as_create_table


class SuggestedQuestionsChain(BaseDocugamiChain[list[str]]):
    dbs: list[SQLDatabase]

    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """

        def table_info(_: Any) -> str:
            infos = ""
            for db in self.dbs:
                infos += get_table_info_as_create_table(db) + "\n\n"

            return infos.strip()

        return {
            "summaries": itemgetter("summaries") | RunnableLambda(formatted_summaries),
            "table_infos": RunnableLambda(table_info),
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
                    "table_infos",
                    "TABLE DESCRIPTIONS",
                    "Descriptions of tables with pre-built reports that can be queried to answer questions.",
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
                "Some suggested questions that may be answered either by searching the type of document represented by the summaries OR querying the given tables (if any), "
                + "considering the rules and examples provided.",
            ),
            task_description="generates some questions a user may want to ask against against a set of documents either by directly searching them OR by querying some pre-built reports",
            additional_instructions=[
                "- Base your questions only on the given document summaries, OR the columns in the provided pre-built report tables (if any).",
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
    ) -> list[list[str]]:
        raise NotImplementedError()
