from operator import itemgetter
from typing import AsyncIterator, Dict, Optional

from langchain_core.runnables import Runnable, RunnableMap

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base_chain import BaseChainRunnable
from docugami_langchain.chains.querying.sql_query_explainer_chain import (
    SQLQueryExplainerChain,
)
from docugami_langchain.chains.querying.sql_result_chain import SQLResultChain
from docugami_langchain.chains.querying.sql_result_explainer_chain import (
    SQLResultExplainerChain,
)
from docugami_langchain.params import RunnableParameters


class DocugamiExplainedSQLQueryChain(BaseChainRunnable[Dict]):
    sql_result_chain: SQLResultChain
    sql_result_explainer_chain: SQLResultExplainerChain
    sql_query_explainer_chain: Optional[SQLQueryExplainerChain]

    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """

        return RunnableMap(
            {
                "question": itemgetter("question"),
                "results": self.sql_result_chain.runnable()
                | {
                    "sql_query": itemgetter("sql_query"),
                    "sql_result": itemgetter("sql_result"),
                    "question": itemgetter("question"),
                }
                | {
                    "sql_query": itemgetter("sql_query"),
                    "sql_result": itemgetter("sql_result"),
                    "explained_sql_result": self.sql_result_explainer_chain.runnable(),
                    "explained_sql_query": (
                        self.sql_query_explainer_chain.runnable()
                        if self.sql_query_explainer_chain
                        else None
                    ),
                },
            }
        )

    def params(self) -> RunnableParameters:
        raise NotImplementedError()

    def run(  # type: ignore[override]
        self,
        question: str,
        config: Optional[dict] = None,
    ) -> Dict:
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            config=config,
        )

    def run_stream(  # type: ignore[override]
        self,
        question: str,
        config: Optional[dict] = None,
    ) -> AsyncIterator[TracedResponse[Dict]]:
        if not question:
            raise Exception("Input required: question")

        return super().run_stream(
            question=question,
            config=config,
        )

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[str],
        config: Optional[dict] = None,
    ) -> list[Dict]:
        return super().run_batch(
            inputs=[{"question": i} for i in inputs],
            config=config,
        )
