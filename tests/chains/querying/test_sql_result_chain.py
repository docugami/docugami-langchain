import os

import pytest
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains import SQLFixupChain, SQLResultChain
from docugami_langchain.chains.types.data_type_detection_chain import (
    DataTypeDetectionChain,
)
from docugami_langchain.chains.types.date_parse_chain import DateParseChain
from docugami_langchain.chains.types.float_parse_chain import FloatParseChain
from docugami_langchain.chains.types.int_parse_chain import IntParseChain
from docugami_langchain.tools.reports import connect_to_excel
from tests.common import TEST_DATA_DIR, verify_traced_response
from tests.testdata.xlsx.query_test_data import QUERY_TEST_DATA, QueryTestData

SQL_EXAMPLES_FILE = TEST_DATA_DIR / "examples/test_sql_examples.yaml"
SQL_FIXUP_EXAMPLES_FILE = TEST_DATA_DIR / "examples/test_sql_fixup_examples.yaml"
DATA_TYPE_DETECTION_EXAMPLES_FILE = (
    TEST_DATA_DIR / "examples/test_data_type_detection_examples.yaml"
)
DATE_PARSE_EXAMPLES_FILE = TEST_DATA_DIR / "examples/test_date_parse_examples.yaml"
FLOAT_PARSE_EXAMPLES_FILE = TEST_DATA_DIR / "examples/test_float_parse_examples.yaml"
INT_PARSE_EXAMPLES_FILE = TEST_DATA_DIR / "examples/test_int_parse_examples.yaml"


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

    detection_chain = DataTypeDetectionChain(llm=llm, embeddings=embeddings)
    detection_chain.load_examples(DATA_TYPE_DETECTION_EXAMPLES_FILE)

    date_parse_chain = DateParseChain(llm=llm, embeddings=embeddings)
    date_parse_chain.load_examples(DATE_PARSE_EXAMPLES_FILE)

    float_parse_chain = FloatParseChain(llm=llm, embeddings=embeddings)
    float_parse_chain.load_examples(FLOAT_PARSE_EXAMPLES_FILE)

    int_parse_chain = IntParseChain(llm=llm, embeddings=embeddings)
    int_parse_chain.load_examples(INT_PARSE_EXAMPLES_FILE)

    sql_result_chain.optimize(
        detection_chain=detection_chain,
        date_parse_chain=date_parse_chain,
        float_parse_chain=float_parse_chain,
        int_parse_chain=int_parse_chain,
    )

    return sql_result_chain


def _runtest(chain: SQLResultChain, test_data: QueryTestData) -> None:
    response = chain.run(question=test_data.question)

    # In this test, we are not actually checking what the query returns,
    # only that it does not throw an exception.
    #
    # There are other tests in the explainer chain that look at the answer
    verify_traced_response(
        response,
        [],
        empty_ok=True,  # If query is valid but returns nothing, it will be empty
    )


@pytest.mark.parametrize("test_data", QUERY_TEST_DATA)
@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks API token not set"
)
def test_fireworksai_llama3_sql_result(
    test_data: QueryTestData,
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.report.data_file, table_name=test_data.report.name
    )
    _runtest(
        init_sql_result_chain(
            db=db,
            llm=fireworksai_llama3,
            embeddings=huggingface_minilm,
        ),
        test_data,
    )


@pytest.mark.parametrize("test_data", QUERY_TEST_DATA)
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API token not set")
def test_openai_gpt4_sql_result(
    test_data: QueryTestData,
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    db = connect_to_excel(
        file_path=test_data.report.data_file, table_name=test_data.report.name
    )
    _runtest(
        init_sql_result_chain(
            db=db,
            llm=openai_gpt4,
            embeddings=openai_ada,
        ),
        test_data,
    )
