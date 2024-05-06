from operator import itemgetter
from typing import AsyncIterator, Optional

from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableMap,
)

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.chains.querying.models import ExplainedSQLQuestionResult
from docugami_langchain.chains.querying.sql_query_explainer_chain import (
    SQLQueryExplainerChain,
)
from docugami_langchain.chains.querying.sql_result_chain import SQLResultChain
from docugami_langchain.chains.querying.sql_result_explainer_chain import (
    SQLResultExplainerChain,
)
from docugami_langchain.params import RunnableParameters


class DocugamiExplainedSQLQueryChain(BaseDocugamiChain[ExplainedSQLQuestionResult]):
    sql_result_chain: SQLResultChain
    sql_result_explainer_chain: Optional[SQLResultExplainerChain] = None
    sql_query_explainer_chain: Optional[SQLQueryExplainerChain] = None

    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """

        def empty_string(inputs: dict) -> str:
            return ""

        return RunnableMap(
            {
                "question": itemgetter("question"),
                "results": self.sql_result_chain.runnable()
                | {
                    "question": itemgetter("question"),
                    "sql_query": itemgetter("sql_query"),
                    "sql_result": itemgetter("sql_result"),
                }
                | {
                    "sql_query": itemgetter("sql_query"),
                    "sql_result": itemgetter("sql_result"),
                    "explained_sql_result": (
                        self.sql_result_explainer_chain.runnable()
                        if self.sql_result_explainer_chain
                        else RunnableLambda(empty_string)
                    ),
                    "explained_sql_query": (
                        self.sql_query_explainer_chain.runnable()
                        if self.sql_query_explainer_chain
                        else RunnableLambda(empty_string)
                    ),
                },
            }
        )

    def params(self) -> RunnableParameters:
        raise NotImplementedError()

    def run(  # type: ignore[override]
        self,
        question: str,
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[ExplainedSQLQuestionResult]:
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
    ) -> AsyncIterator[TracedResponse[ExplainedSQLQuestionResult]]:
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
    ) -> list[ExplainedSQLQuestionResult]:
        return super().run_batch(
            inputs=[{"question": i} for i in inputs],
            config=config,
        )
