from typing import AsyncIterator, Optional, Tuple

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base_chain import BaseDocugamiChain
from docugami_langchain.output_parsers.sql_finding import SQLFindingOutputParser
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
                    "SQL exception returned when executing the SQL query with mistakes.",
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
                "- Don't select more than 10 columns to avoid making the query so long that it gets truncated",
                "",
                "If you see any of the above mistakes, or any other mistakes, rewrite the query to fix them. If there are no mistakes, just reproduce the original query.",
            ],
            stop_sequences=[
                "\n",
                ";",
                "</s>",
                "|",
            ],
            additional_runnables=[SQLFindingOutputParser()],
        )

    def run(  # type: ignore[override]
        self,
        table_info: str,
        sql_query: str,
        exception: str,
        config: Optional[dict] = None,
    ) -> str:
        if not table_info or not sql_query or not exception:
            raise Exception("Inputs required: table_info, sql_query, exception")

        return super().run(
            table_info=table_info,
            sql_query=sql_query,
            exception=exception,
            config=config,
        )

    def run_stream(  # type: ignore[override]
        self,
        table_info: str,
        sql_query: str,
        exception: str,
        config: Optional[dict] = None,
    ) -> AsyncIterator[TracedResponse[str]]:
        if not table_info or not sql_query or not exception:
            raise Exception("Inputs required: table_info, sql_query, exception")

        return super().run_stream(
            table_info=table_info,
            sql_query=sql_query,
            exception=exception,
            config=config,
        )

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[Tuple[str, str, str]],
        config: Optional[dict] = None,
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
