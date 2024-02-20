import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains import SQLResultExplainerChain
from tests.conftest import TEST_DATA_DIR, verify_chain_response
from tests.testdata.xlsx.sql_test_data import SQL_TEST_DATA, SQLTestData

SQL_EXAMPLES_FILE = TEST_DATA_DIR / "examples/test_sql_examples.yaml"


@pytest.fixture()
def fireworksai_mixtral_result_explainer_chain(
    fireworksai_mixtral: BaseLanguageModel, huggingface_minilm: Embeddings
) -> SQLResultExplainerChain:
    """
    Fireworks AI chain to do SQL result explanations using mixtral.
    """
    chain = SQLResultExplainerChain(
        llm=fireworksai_mixtral, embeddings=huggingface_minilm
    )
    chain.load_examples(SQL_EXAMPLES_FILE)
    return chain


@pytest.fixture()
def openai_gpt35_result_explainer_chain(
    openai_gpt35: BaseLanguageModel, openai_ada: Embeddings
) -> SQLResultExplainerChain:
    """
    OpenAI chain to do SQL result explanations using GPT 3.5.
    """
    chain = SQLResultExplainerChain(llm=openai_gpt35, embeddings=openai_ada)
    chain.load_examples(SQL_EXAMPLES_FILE)
    return chain


def _runtest(chain: SQLResultExplainerChain, test_data: SQLTestData) -> None:
    explained_result = chain.run(
        question=test_data.question,
        sql_query=test_data.sql_query,
        sql_result=test_data.sql_result,
    )
    verify_chain_response(explained_result, test_data.explained_result_answer_fragments)
    assert explained_result


@pytest.mark.parametrize("test_data", SQL_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_sql_result_explainer(
    fireworksai_mixtral_result_explainer_chain: SQLResultExplainerChain,
    test_data: SQLTestData,
) -> None:
    _runtest(fireworksai_mixtral_result_explainer_chain, test_data)


@pytest.mark.parametrize("test_data", SQL_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_sql_result_explainer(
    openai_gpt35_result_explainer_chain: SQLResultExplainerChain, test_data: SQLTestData
) -> None:
    _runtest(openai_gpt35_result_explainer_chain, test_data)
