import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.querying import SQLQueryExplainerChain
from tests.common import TEST_DATA_DIR, verify_traced_response
from tests.testdata.xlsx.query_test_data import QUERY_TEST_DATA, QueryTestData


def init_chain(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> SQLQueryExplainerChain:
    chain = SQLQueryExplainerChain(llm=llm, embeddings=embeddings)
    chain.load_examples(TEST_DATA_DIR / "examples/test_sql_examples.yaml")
    return chain


def _runtest(chain: SQLQueryExplainerChain, test_data: QueryTestData) -> None:
    explained_result = chain.run(
        question=test_data.question,
        sql_query=test_data.sql_query,
        sql_result=test_data.sql_result,
    )
    verify_traced_response(explained_result, test_data.explained_sql_query_fragments)
    assert explained_result


@pytest.mark.parametrize("test_data", QUERY_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_llama3_sql_query_explainer(
    test_data: QueryTestData,
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_llama3, huggingface_minilm)
    _runtest(chain, test_data)


@pytest.mark.parametrize("test_data", QUERY_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_gpt4_sql_query_explainer(
    test_data: QueryTestData,
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    chain = init_chain(openai_gpt4, openai_ada)
    _runtest(chain, test_data)
