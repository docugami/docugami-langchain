import os
from pathlib import Path
from typing import Any

import pytest
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.types.data_type_detection_chain import (
    DataTypeDetectionChain,
)
from docugami_langchain.chains.types.date_parse_chain import DateParseChain
from docugami_langchain.chains.types.float_parse_chain import FloatParseChain
from docugami_langchain.chains.types.int_parse_chain import IntParseChain
from docugami_langchain.tools.reports import connect_to_excel
from docugami_langchain.utils.type_detection import convert_to_typed
from tests.common import TEST_DATA_DIR

TEST_DATA = [
    (
        TEST_DATA_DIR / "xlsx/Data Type Test.xlsx",
        "Data Type Test",
        [
            '"Test Bool" INTEGER',
            '"Test Money ($)" INTEGER',
            '"Test Measure (square feet)" REAL',
            '"Test Date" TEXT',
            '"Test Text" TEXT',
        ],
    ),
    (
        TEST_DATA_DIR / "xlsx/Charters Summary.xlsx",
        "Corporate Charters",
        [
            '"FILED Date" TEXT',
            '"FILED Time" TEXT',
            '"SR" REAL',
            '"FileNumber" REAL',
            '"Corporation Name" TEXT',
            '"Registered Address" TEXT',
            '"Shares of Common Stock" INTEGER',
            '"Shares of Preferred Stock" INTEGER',
        ],
    ),
]


def init_chains(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> tuple[DataTypeDetectionChain, DateParseChain, FloatParseChain, IntParseChain]:
    detection_chain = DataTypeDetectionChain(llm=llm, embeddings=embeddings)
    detection_chain.load_examples(
        TEST_DATA_DIR / "examples/test_data_type_detection_examples.yaml"
    )

    date_parse_chain = DateParseChain(llm=llm, embeddings=embeddings)
    date_parse_chain.load_examples(
        TEST_DATA_DIR / "examples/test_date_parse_examples.yaml"
    )

    float_parse_chain = FloatParseChain(llm=llm, embeddings=embeddings)
    float_parse_chain.load_examples(
        TEST_DATA_DIR / "examples/test_float_parse_examples.yaml"
    )
    int_parse_chain = IntParseChain(llm=llm, embeddings=embeddings)
    int_parse_chain.load_examples(
        TEST_DATA_DIR / "examples/test_int_parse_examples.yaml"
    )
    return detection_chain, date_parse_chain, float_parse_chain, int_parse_chain


def _run_test(
    db: SQLDatabase,
    detection_chain: DataTypeDetectionChain,
    date_parse_chain: DateParseChain,
    float_parse_chain: FloatParseChain,
    int_parse_chain: IntParseChain,
    table_name: str,
    typed_columns: list[str],
) -> None:
    converted_db = convert_to_typed(
        db=db,
        data_type_detection_chain=detection_chain,
        date_parse_chain=date_parse_chain,
        float_parse_chain=float_parse_chain,
        int_parse_chain=int_parse_chain,
    )

    info = converted_db.get_table_info()

    # Ensure table name is unchanged and there is only 1 table
    assert f'TABLE "{table_name}"' in info
    assert info.count("CREATE TABLE") == 1

    # Ensure all columns are typed as expected
    for col in typed_columns:
        assert col in info


@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks API token not set"
)
@pytest.mark.parametrize("data_file,table_name,typed_columns", TEST_DATA)
def test_fireworksai_llama3_data_type_conversion(
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    data_file: Path,
    table_name: str,
    typed_columns: list[str],
) -> Any:
    db = connect_to_excel(file_path=data_file, table_name=table_name)
    detection_chain, date_parse_chain, float_parse_chain, int_parse_chain = init_chains(
        fireworksai_llama3, huggingface_minilm
    )
    _run_test(
        db,
        detection_chain,
        date_parse_chain,
        float_parse_chain,
        int_parse_chain,
        table_name,
        typed_columns,
    )


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.parametrize("data_file,table_name,typed_columns", TEST_DATA)
def test_openai_gpt4_data_type_conversion(
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
    data_file: Path,
    table_name: str,
    typed_columns: list[str],
) -> Any:
    db = connect_to_excel(file_path=data_file, table_name=table_name)
    detection_chain, date_parse_chain, float_parse_chain, int_parse_chain = init_chains(
        openai_gpt4, openai_ada
    )
    _run_test(
        db,
        detection_chain,
        date_parse_chain,
        float_parse_chain,
        int_parse_chain,
        table_name,
        typed_columns,
    )
