from typing import AsyncIterator, Optional

from langchain_core.runnables import RunnableConfig

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.output_parsers.sql_finding import SQLFindingOutputParser
from docugami_langchain.output_parsers.text_cleaning import TextCleaningOutputParser
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class SQLFixupChain(BaseDocugamiChain[str]):
    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "table_info",
                    "TABLE DESCRIPTION",
                    "Description of the table to be queried via SQL.",
                ),
                RunnableSingleParameter(
                    "sql_query",
                    "INPUT SQL QUERY",
                    "SQL query with possible mistakes that should be fixed.",
                ),
                RunnableSingleParameter(
                    "exception",
                    "EXCEPTION",
                    "Optional SQL exception that was returned when this SQL query was executed against the table. If not provided, just ignore.",
                ),
            ],
            output=RunnableSingleParameter(
                "fixed_sql_query",
                "FIXED SQL QUERY",
                "Fixed SQL query, considering the rules and examples provided.",
            ),
            task_description="acts as a SQLite expert and given an input SQL query, fixes common SQL mistakes",
            additional_instructions=[
                "- Fix data type mismatch in predicates",
                "- Make sure the correct number of arguments are used for functions",
                "- Make sure you casting to the correct data type",
                "- Quote all column names and strings appropriately per SQLite syntax",
                "- Make sure all column names in SELECT statements actually exist in the table (update column names if you find a near match)",
                "- Don't select more than 10 columns to avoid making the query so long that it gets truncated",
                "",
                "If you see any of the above mistakes, or any other mistakes, rewrite the query to fix them. If there are no mistakes, just reproduce the original query.",
            ],
            stop_sequences=["\n", ";", "<|eot_id|>"],
            additional_runnables=[TextCleaningOutputParser(), SQLFindingOutputParser()],
            include_output_instruction_suffix=True,
        )

    def run(  # type: ignore[override]
        self,
        table_info: str,
        sql_query: str,
        exception: str = "",
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[str]:
        if not table_info or not sql_query:
            raise Exception("Inputs required: table_info, sql_query")

        return super().run(
            table_info=table_info,
            sql_query=sql_query,
            exception=exception,
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        table_info: str,
        sql_query: str,
        exception: str = "",
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[str]]:
        if not table_info or not sql_query:
            raise Exception("Inputs required: table_info, sql_query")

        async for item in super().run_stream(
            table_info=table_info,
            sql_query=sql_query,
            exception=exception,
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, str, str]],
        config: Optional[RunnableConfig] = None,
    ) -> list[str]:
        return super().run_batch(
            inputs=[
                {
                    "table_info": i[0],
                    "sql_query": i[1],
                    "exception": i[2],
                }
                for i in inputs
            ],
            config=config,
        )
