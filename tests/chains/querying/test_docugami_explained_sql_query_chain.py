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
from docugami_langchain.tools.reports import connect_to_excel
from tests.common import TEST_DATA_DIR, verify_chain_response
from tests.testdata.xlsx.sql_test_data import SQL_TEST_DATA, SQLTestData

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


def _runtest(chain: DocugamiExplainedSQLQueryChain, test_data: SQLTestData) -> None:
    response = chain.run(question=test_data.question)
    results = response.get("results")
    assert results
    verify_chain_response(
        results.get("explained_sql_result"), test_data.explained_result_answer_fragments
    )


async def _runtest_streamed(
    chain: DocugamiExplainedSQLQueryChain, test_data: SQLTestData
) -> None:
    chain_response = TracedResponse[dict](value={})
    async for incremental_response in chain.run_stream(question=test_data.question):
        chain_response = incremental_response

    assert chain_response.value
    assert chain_response.run_id
    results = chain_response.value.get("results")
    assert results
    verify_chain_response(
        results.get("explained_sql_result"), test_data.explained_result_answer_fragments
    )


@pytest.mark.parametrize("test_data", SQL_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_docugami_explained_sql_query(
    test_data: SQLTestData,
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.data_file, table_name=test_data.table_name
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


@pytest.mark.parametrize("test_data", SQL_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_streamed_docugami_explained_sql_query(
    test_data: SQLTestData,
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.data_file, table_name=test_data.table_name
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


@pytest.mark.parametrize("test_data", SQL_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_docugami_explained_sql_query(
    test_data: SQLTestData,
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.data_file, table_name=test_data.table_name
    )
    _runtest(
        init_docugami_explained_sql_query_chain(
            db=db,
            sql_llm=openai_gpt35,
            explainer_llm=openai_gpt35,
            embeddings=openai_ada,
        ),
        test_data,
    )


@pytest.mark.parametrize("test_data", SQL_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.asyncio
async def test_openai_docugami_streamed_explained_sql_query(
    test_data: SQLTestData,
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.data_file, table_name=test_data.table_name
    )
    await _runtest_streamed(
        init_docugami_explained_sql_query_chain(
            db=db,
            sql_llm=openai_gpt35,
            explainer_llm=openai_gpt35,
            embeddings=openai_ada,
        ),
        test_data,
    )
