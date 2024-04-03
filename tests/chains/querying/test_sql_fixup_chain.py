import os
from dataclasses import dataclass

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.querying import SQLFixupChain
from tests.common import TEST_DATA_DIR, is_core_tests_only_mode, verify_traced_response


@dataclass
class SQLFixupTestData:
    table_info: str
    sql_query: str
    exception: str
    fixed_sql_query: str
    is_core_test: bool = False


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


def init_chain(llm: BaseLanguageModel, embeddings: Embeddings) -> SQLFixupChain:
    chain = SQLFixupChain(llm=llm, embeddings=embeddings)
    chain.load_examples(TEST_DATA_DIR / "examples/test_sql_fixup_examples.yaml")
    return chain


def _runtest(chain: SQLFixupChain, test_data: SQLFixupTestData) -> None:
    fixed_sql = chain.run(
        table_info=test_data.table_info,
        sql_query=test_data.sql_query,
        exception=test_data.exception,
    )
    verify_traced_response(fixed_sql)
    assert fixed_sql.value.strip().lower().startswith("select")


@pytest.mark.parametrize("test_data", SQL_FIXUP_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_sql_fixup(
    test_data: SQLFixupTestData,
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_mixtral, huggingface_minilm)
    _runtest(chain, test_data)


@pytest.mark.parametrize("test_data", SQL_FIXUP_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_sql_fixup(
    test_data: SQLFixupTestData,
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    chain = init_chain(openai_gpt35, openai_ada)
    _runtest(chain, test_data)
