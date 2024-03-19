import os

import pytest
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains import SuggestedQuestionsChain
from docugami_langchain.tools.reports import connect_to_excel
from tests.common import TEST_DATA_DIR, verify_traced_response
from tests.testdata.xlsx.sql_test_data import SQL_TEST_DATA, SQLTestData

SQL_SUGGESTIONS_EXAMPLES_FILE = (
    TEST_DATA_DIR / "examples/test_suggestions_examples.yaml"
)


def init_suggested_questions_chain(
    db: SQLDatabase,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
) -> SuggestedQuestionsChain:
    chain = SuggestedQuestionsChain(
        db=db,
        llm=llm,
        embeddings=embeddings,
    )
    chain.load_examples(SQL_SUGGESTIONS_EXAMPLES_FILE)
    return chain


def _runtest(chain: SuggestedQuestionsChain) -> None:
    suggestions = chain.run()
    verify_traced_response(suggestions)
    assert len(suggestions.value) > 0


@pytest.mark.parametrize("test_data", SQL_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_suggestions(
    test_data: SQLTestData,
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.data_file, table_name=test_data.table_name
    )
    _runtest(
        init_suggested_questions_chain(
            db=db,
            llm=fireworksai_mixtral,
            embeddings=huggingface_minilm,
        )
    )


@pytest.mark.parametrize("test_data", SQL_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_suggestions(
    test_data: SQLTestData,
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.data_file, table_name=test_data.table_name
    )
    _runtest(
        init_suggested_questions_chain(
            db=db,
            llm=openai_gpt35,
            embeddings=openai_ada,
        )
    )
