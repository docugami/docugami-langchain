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
from tests.testdata.xlsx.query_test_data import (
    AVIATION_INCIDENTS_DATA_FILE,
    AVIATION_INCIDENTS_TABLE_NAME,
    CHARTERS_SUMMARY_DATA_FILE,
    CHARTERS_SUMMARY_TABLE_NAME,
    DATA_TYPE_TEST_TABLE_NAME,
)

TEST_DATA = [
    (
        TEST_DATA_DIR / "xlsx/Data Type Test.xlsx",
        DATA_TYPE_TEST_TABLE_NAME,
        [
            '"Test Bool" INTEGER',
            '"Test Money ($)" INTEGER',
            '"Test Measure (square feet)" REAL',
            '"Test Date" TEXT',
            '"Test Text" TEXT',
        ],
        [
            (
                f'SELECT * FROM "{DATA_TYPE_TEST_TABLE_NAME}" LIMIT 1 OFFSET 0',
                """[(1, 123, 2.5, \'1982-07-01T00:00:00\', \'lorem\')]""",
            ),
            (
                f'SELECT * FROM "{DATA_TYPE_TEST_TABLE_NAME}" LIMIT 1 OFFSET 1',
                """[(0, 500, 20.1, \'2012-07-01T00:00:00\', \'ipsum\')]""",
            ),
            (
                f'SELECT * FROM "{DATA_TYPE_TEST_TABLE_NAME}" LIMIT 1 OFFSET 2',
                """[(0, 456, None, None, \'dolor with "quoted" string to test escaping\')]""",
            ),
            (
                f'SELECT * FROM "{DATA_TYPE_TEST_TABLE_NAME}" LIMIT 1 OFFSET 3',
                """[(1, None, None, None, None)]""",
            ),
            (
                f'SELECT * FROM "{DATA_TYPE_TEST_TABLE_NAME}" LIMIT 1 OFFSET 4',
                """[(0, None, 100.5, None, \'different format $100 but save as string\')]""",
            ),
        ],
    ),
    (
        CHARTERS_SUMMARY_DATA_FILE,
        CHARTERS_SUMMARY_TABLE_NAME,
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
        [
            (
                f'SELECT * FROM "{CHARTERS_SUMMARY_TABLE_NAME}" WHERE "Corporation Name" LIKE "%checkerspot%"',
                """[('2018-04-25T00:00:00', '02:31 PM', 20183019534.0, 6058686.0, 'Checkerspot, Inc.', '1209 Orange Street, City of Wilmington, County of New Castle, 19801', 22000000, 7500000)]""",  # noqa: E501
            ),
            (
                f'SELECT * FROM "{CHARTERS_SUMMARY_TABLE_NAME}" WHERE "Corporation Name" LIKE "%brex%"',
                """[(None, '08:10 AM', 20181839629.0, None, 'Brex Inc.', '2711 Centerville Road, Suite 400, Wilmington, New Castle County, Delaware 19808.', 25384336, 11812654)]""",
            ),
        ],
    ),
    (
        AVIATION_INCIDENTS_DATA_FILE,
        AVIATION_INCIDENTS_TABLE_NAME,
        [
            '"Incident Number" TEXT',
            '"Accident Date" TEXT',
            '"Accident Time" TEXT',
            '"Location" TEXT',
            '"Aircraft Registration" TEXT',
            '"Aircraft Make" TEXT',
            '"Aircraft Model" TEXT',
            '"Aircraft Damage" TEXT',
            '"Injuries" INTEGER',
            '"Probable Cause" TEXT',
            '"Pilot Age" INTEGER',
        ],
        [
            (
                f'SELECT * FROM "{AVIATION_INCIDENTS_TABLE_NAME}" WHERE "Incident Number" LIKE "%MIA08CA045%"',
                """[(\'MIA08CA045\', \'2007-12-26T00:00:00\', \'1700 EST\', \'Northampton, MA\', \'N5425K\', \'Cessna\', \'172P\', \'Substantial\', 2, "The National Transportation Safety Board determines the probable cause(s) of this accident to be: The student pilot\'s failure to maintain directional control of the airplane during the takeoff roll and inadequate supervision by the CFI.", 16)]""",  # noqa: E501
            ),
            (
                f'SELECT * FROM "{AVIATION_INCIDENTS_TABLE_NAME}" WHERE "Incident Number" LIKE "%DEN08RA050%"',
                """[(\'DEN08RA050\', \'2007-12-22T00:00:00\', \'1500 UTC\', \'Pointe-A-Pitre, Guadeloupe\', \'N901TP\', \'Partenavia\', \'AP 68 TP\', \'Substantial\', 2, \'The French investigators reported that pilot experienced a problem during the takeoff roll. The pilot aborted the takeoff. Further examination revealed an uncontained "rupture" of the left engine.\', None)]""",  # noqa: E501
            ),
        ],
    ),
]


def init_chains(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> tuple[DataTypeDetectionChain, DateParseChain, FloatParseChain, IntParseChain]:
    detection_chain = DataTypeDetectionChain(llm=llm, embeddings=embeddings)
    detection_chain.langsmith_tracing_enabled = True  # override for debugging
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
    queries: list[tuple[str]],
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

    # Run some queries and make sure results match expectations (are in the right formats)
    for query in queries:
        cmd = query[0]
        expected_result = query[1]  # type: ignore

        result = converted_db.run(cmd)
        assert result == expected_result


@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks API token not set"
)
@pytest.mark.parametrize("data_file,table_name,typed_columns,queries", TEST_DATA)
def test_fireworksai_llama3_data_type_conversion(
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    data_file: Path,
    table_name: str,
    typed_columns: list[str],
    queries: list[tuple[str]],
) -> Any:
    db = connect_to_excel(io=data_file, table_name=table_name)
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
        queries,
    )


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API token not set")
@pytest.mark.parametrize("data_file,table_name,typed_columns,queries", TEST_DATA)
def test_openai_gpt4_data_type_conversion(
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
    data_file: Path,
    table_name: str,
    typed_columns: list[str],
    queries: list[tuple[str]],
) -> Any:
    db = connect_to_excel(io=data_file, table_name=table_name)
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
        queries,
    )
