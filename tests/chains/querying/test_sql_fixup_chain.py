import os
from dataclasses import dataclass

import pytest
import sqlparse
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.querying import SQLFixupChain
from tests.common import TEST_DATA_DIR, is_core_tests_only_mode, verify_response


@dataclass
class SQLFixupTestData:
    table_info: str
    sql_query: str
    exception: str
    fixed_sql_query: str
    is_core_test: bool = False


SQL_FIXUP_EXAMPLES_FILE = TEST_DATA_DIR / "examples/test_sql_fixup_examples.yaml"

SQL_FIXUP_TEST_DATA: list[SQLFixupTestData] = [
    SQLFixupTestData(
        table_info="""CREATE TABLE Users (
      "UserID" INTEGER PRIMARY KEY, 
      "Username" TEXT, 
      "Email" TEXT, 
      "Password" TEXT, 
      "DateJoined" TEXT
    )""",
        sql_query="""SELECT "UserID", "Username", "Email", "Password" FROM User""",
        exception="(pysqlite3.dbapi2.OperationalError) no such table: User",
        fixed_sql_query='SELECT "UserID", "Username", "Email", "Password" FROM Users',
        is_core_test=True,
    ),
]

if is_core_tests_only_mode():
    SQL_FIXUP_TEST_DATA = [t for t in SQL_FIXUP_TEST_DATA if t.is_core_test]


@pytest.fixture()
def fireworksai_mixtral_sql_fixup_chain(
    fireworksai_mixtral: BaseLanguageModel, huggingface_minilm: Embeddings
) -> SQLFixupChain:
    """
    Fireworks AI chain to do SQL fixup using mixtral.
    """
    chain = SQLFixupChain(
        llm=fireworksai_mixtral,
        embeddings=huggingface_minilm,
    )
    chain.load_examples(SQL_FIXUP_EXAMPLES_FILE)
    return chain


@pytest.fixture()
def openai_sql_fixup_chain(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> SQLFixupChain:
    """
    OpenAI chain to do SQL fixup.
    """
    chain = SQLFixupChain(llm=openai_gpt35, embeddings=openai_ada)
    chain.load_examples(SQL_FIXUP_EXAMPLES_FILE)
    return chain


def _runtest(chain: SQLFixupChain, test_data: SQLFixupTestData) -> None:
    fixed_sql = chain.run(
        table_info=test_data.table_info,
        sql_query=test_data.sql_query,
        exception=test_data.exception,
    )
    verify_response(fixed_sql)
    assert fixed_sql.value.strip().lower().startswith("select")
    assert sqlparse.parse(fixed_sql.value)


@pytest.mark.parametrize("test_data", SQL_FIXUP_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_sql_fixup(
    fireworksai_mixtral_sql_fixup_chain: SQLFixupChain, test_data: SQLFixupTestData
) -> None:
    _runtest(fireworksai_mixtral_sql_fixup_chain, test_data)


@pytest.mark.parametrize("test_data", SQL_FIXUP_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_sql_fixup(
    openai_sql_fixup_chain: SQLFixupChain, test_data: SQLFixupTestData
) -> None:
    _runtest(openai_sql_fixup_chain, test_data)
