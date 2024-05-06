import os

import pytest
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.querying import (
    DocugamiExplainedSQLQueryChain,
    SQLFixupChain,
    SQLQueryExplainerChain,
    SQLResultChain,
    SQLResultExplainerChain,
)
from docugami_langchain.chains.querying.models import ExplainedSQLQuestionResult
from docugami_langchain.tools.reports import connect_to_excel
from tests.common import TEST_DATA_DIR, verify_traced_response
from tests.testdata.xlsx.query_test_data import QUERY_TEST_DATA, QueryTestData

SQL_EXAMPLES_FILE = TEST_DATA_DIR / "examples/test_sql_examples.yaml"
SQL_FIXUP_EXAMPLES_FILE = TEST_DATA_DIR / "examples/test_sql_fixup_examples.yaml"


def init_docugami_explained_sql_query_chain(
    db: SQLDatabase,
    sql_llm: BaseLanguageModel,
    explainer_llm: BaseLanguageModel,
    embeddings: Embeddings,
) -> DocugamiExplainedSQLQueryChain:
    fixup_chain = SQLFixupChain(llm=sql_llm, embeddings=embeddings)
    fixup_chain.load_examples(SQL_FIXUP_EXAMPLES_FILE)

    sql_result_chain = SQLResultChain(
        llm=sql_llm,
        embeddings=embeddings,
        db=db,
        sql_fixup_chain=fixup_chain,
    )
    sql_result_chain.load_examples(SQL_EXAMPLES_FILE)
    sql_result_chain.optimize()

    sql_result_explainer_chain = SQLResultExplainerChain(
        llm=explainer_llm,
        embeddings=embeddings,
    )
    sql_result_explainer_chain.load_examples(SQL_EXAMPLES_FILE)

    sql_query_explainer_chain = SQLQueryExplainerChain(
        llm=explainer_llm,
        embeddings=embeddings,
    )
    sql_query_explainer_chain.load_examples(SQL_EXAMPLES_FILE)

    return DocugamiExplainedSQLQueryChain(
        llm=explainer_llm,
        embeddings=embeddings,
        sql_result_chain=sql_result_chain,
        sql_result_explainer_chain=sql_result_explainer_chain,
        sql_query_explainer_chain=sql_query_explainer_chain,
    )


def _runtest(chain: DocugamiExplainedSQLQueryChain, test_data: QueryTestData) -> None:
    response = chain.run(question=test_data.question)
    verify_traced_response(response, test_data.explained_result_answer_fragments)


async def _runtest_streamed(
    chain: DocugamiExplainedSQLQueryChain, test_data: QueryTestData
) -> None:
    chain_response = TracedResponse[ExplainedSQLQuestionResult](value={})
    async for incremental_response in chain.run_stream(question=test_data.question):
        chain_response = incremental_response

    verify_traced_response(chain_response, test_data.explained_result_answer_fragments)


@pytest.mark.parametrize("test_data", QUERY_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_docugami_explained_sql_query(
    test_data: QueryTestData,
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.report.data_file, table_name=test_data.report.name
    )
    _runtest(
        init_docugami_explained_sql_query_chain(
            db=db,
            sql_llm=fireworksai_mixtral,
            explainer_llm=fireworksai_mixtral,
            embeddings=huggingface_minilm,
        ),
        test_data,
    )


@pytest.mark.parametrize("test_data", QUERY_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_streamed_docugami_explained_sql_query(
    test_data: QueryTestData,
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.report.data_file, table_name=test_data.report.name
    )
    await _runtest_streamed(
        init_docugami_explained_sql_query_chain(
            db=db,
            sql_llm=fireworksai_mixtral,
            explainer_llm=fireworksai_mixtral,
            embeddings=huggingface_minilm,
        ),
        test_data,
    )


@pytest.mark.parametrize("test_data", QUERY_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_gpt4_docugami_explained_sql_query(
    test_data: QueryTestData,
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.report.data_file, table_name=test_data.report.name
    )
    _runtest(
        init_docugami_explained_sql_query_chain(
            db=db,
            sql_llm=openai_gpt4,
            explainer_llm=openai_gpt4,
            embeddings=openai_ada,
        ),
        test_data,
    )


@pytest.mark.parametrize("test_data", QUERY_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.asyncio
async def test_openai_gpt4_docugami_streamed_explained_sql_query(
    test_data: QueryTestData,
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.report.data_file, table_name=test_data.report.name
    )
    await _runtest_streamed(
        init_docugami_explained_sql_query_chain(
            db=db,
            sql_llm=openai_gpt4,
            explainer_llm=openai_gpt4,
            embeddings=openai_ada,
        ),
        test_data,
    )
