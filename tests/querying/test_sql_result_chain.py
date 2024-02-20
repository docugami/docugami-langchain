import os

import pytest
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains import SQLFixupChain, SQLResultChain
from docugami_langchain.tools.reports import connect_to_excel
from tests.conftest import TEST_DATA_DIR, verify_chain_response
from tests.testdata.xlsx.sql_test_data import SQL_TEST_DATA, SQLTestData

SQL_EXAMPLES_FILE = TEST_DATA_DIR / "examples/test_sql_examples.yaml"
SQL_FIXUP_EXAMPLES_FILE = TEST_DATA_DIR / "examples/test_sql_fixup_examples.yaml"


def init_sql_result_chain(
    db: SQLDatabase,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
) -> SQLResultChain:
    fixup_chain = SQLFixupChain(llm=llm, embeddings=embeddings)
    fixup_chain.load_examples(SQL_FIXUP_EXAMPLES_FILE)

    sql_result_chain = SQLResultChain(
        llm=llm,
        embeddings=embeddings,
        db=db,
        sql_fixup_chain=fixup_chain,
    )
    sql_result_chain.load_examples(SQL_EXAMPLES_FILE)
    return sql_result_chain


def _runtest(chain: SQLResultChain, test_data: SQLTestData) -> None:
    response = chain.run(question=test_data.question)

    # In this test, we are not actually checking what the query returns,
    # only that it does not throw an exception.
    #
    # There are other tests in the explainer chain that look at the answer
    verify_chain_response(
        response.get("sql_result"),
        [],
        empty_ok=True,  # if query is valid but returns nothing, it will be empty
    )


@pytest.mark.parametrize("test_data", SQL_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_sql_result(
    test_data: SQLTestData,
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.data_file, table_name=test_data.table_name
    )
    _runtest(
        init_sql_result_chain(
            db=db,
            llm=fireworksai_mixtral,
            embeddings=huggingface_minilm,
        ),
        test_data,
    )


@pytest.mark.parametrize("test_data", SQL_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_sql_result(
    test_data: SQLTestData,
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.data_file, table_name=test_data.table_name
    )
    _runtest(
        init_sql_result_chain(
            db=db,
            llm=openai_gpt35,
            embeddings=openai_ada,
        ),
        test_data,
    )
