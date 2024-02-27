from typing import Any, AsyncIterator

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.runnables import Runnable, RunnableLambda

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base_chain import BaseChainRunnable
from docugami_langchain.output_parsers.line_separated_list import (
    LineSeparatedListOutputParser,
)
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class SuggestedQuestionsChain(BaseChainRunnable[list[str]]):
    db: SQLDatabase

    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """

        def table_info(_: Any) -> str:
            return self.db.get_table_info()

        return {
            "table_info": RunnableLambda(table_info),
        } | super().runnable()

    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "table_info",
                    "TABLE DESCRIPTION",
                    "Description of the table.",
                ),
            ],
            output=RunnableSingleParameter(
                "suggested_questions",
                "SUGGESTED QUESTIONS",
                "Some suggested questions that may be asked against the table, considering the rules and examples provided.",
            ),
            task_description="acts as a SQLite expert and given a table description, generates some questions a user may want to ask against the table",
            additional_instructions=[
                "- Base your questions only on the columns in the table.",
                "- Generate the best 4 questions, no more than that.",
                "- Generate questions as a list, one question per line.",
            ],
            additional_runnables=[LineSeparatedListOutputParser()],
            stop_sequences=["\n\n", "|"],
            key_finding_output_parse=False,  # set to False for streaming
        )

    def run(  # type: ignore[override]
        self,
    ) -> list[str]:
        return super().run()

    def run_stream(  # type: ignore[override]
        self,
    ) -> AsyncIterator[TracedResponse[list[str]]]:
        return super().run_stream()

    def run_batch(  # type: ignore[override]
        self,
    ) -> list[list[str]]:
        raise NotImplementedError()
