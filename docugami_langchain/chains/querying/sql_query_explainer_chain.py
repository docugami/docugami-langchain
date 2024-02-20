from typing import AsyncIterator, Optional, Tuple

from docugami_langchain.chains.base import BaseDocugamiChain, TracedChainResponse
from docugami_langchain.chains.params import ChainParameters, ChainSingleParameter


class SQLQueryExplainerChain(BaseDocugamiChain[str]):
    def chain_params(self) -> ChainParameters:
        return ChainParameters(
            inputs=[
                ChainSingleParameter(
                    "question",
                    "QUESTION",
                    "Question asked by the user.",
                ),
                ChainSingleParameter(
                    "sql_query",
                    "SQL QUERY",
                    "SQL Query that was run by the system, to answer the question asked.",
                ),
                ChainSingleParameter(
                    "sql_result",
                    "SQL RESULT",
                    "Result of the SQL Query.",
                ),
            ],
            output=ChainSingleParameter(
                "query_explanation",
                "QUERY EXPLANATION",
                "Human readable explanation of the query based on the question, SQL Query and the SQL Result, considering the rules and examples provided."
                + "Please give a short one line answer, only describing the query and not the result. Remember not to mention SQL or tables as instructed, just describe what the query is doing.",
            ),
            task_description="acts as a SQLite expert and given an input SQL Query, a SQL Result and a question, creates a human readable explanation of the query, appropriate for non-technical "
            + "users who don't understand SQL",
            additional_instructions=[
                "- Shorter answers are better, but make sure you always explain all the operations and columns in the SQL Query.",
                "- In your answer, never mention SQL or the fact that you are producing a human readable result based on SQL results.",
                "- In your answer, never mention tables and instead use the term 'report' since that is what we use in our UX.",
                "- Only explain the query in context of the question and the result, don't try to explain the result itself."
                "- Make sure you list the actual operation(s) that were done as well as the column(s) used. This will help users understand which column(s) or operation(s) were used and give "
                + "feedback if a mistake was made by the system.",
                "- Make sure your answer NEVER contains any SQL, since the whole point is to not show the user any SQL and instead only show them a human readable explanation.",
            ],
            key_finding_output_parse=False,  # set to False for streaming
        )

    def run(  # type: ignore[override]
        self,
        question: str,
        sql_query: str,
        sql_result: str,
        config: Optional[dict] = None,
    ) -> str:
        if not question or not sql_query:
            raise Exception("Inputs required: question, sql_query")

        return super().run(
            question=question,
            sql_query=sql_query,
            sql_result=sql_result,
            config=config,
        )

    def run_stream(  # type: ignore[override]
        self,
        question: str,
        sql_query: str,
        sql_result: str,
        config: Optional[dict] = None,
    ) -> AsyncIterator[TracedChainResponse[str]]:
        if not question or not sql_query:
            raise Exception("Inputs required: question, sql_query")

        return super().run_stream(
            question=question,
            sql_query=sql_query,
            sql_result=sql_result,
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
                    "question": i[0],
                    "sql_query": i[1],
                    "sql_result": i[2],
                }
                for i in inputs
            ],
            config=config,
        )
