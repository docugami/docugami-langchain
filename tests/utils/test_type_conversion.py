import os
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.types.data_type_detection_chain import (
    DataTypeDetectionChain,
)
from docugami_langchain.chains.types.date_parse_chain import DateParseChain
from docugami_langchain.chains.types.float_parse_chain import FloatParseChain
from docugami_langchain.tools.reports import connect_to_excel
from docugami_langchain.utils.type_detection import convert_to_typed
from tests.common import TEST_DATA_DIR

DATA_TYPE_TEST_DATA_FILE = TEST_DATA_DIR / "xlsx/Data Type Test.xlsx"
DATA_TYPE_TEST_TABLE_NAME = "Data Type Test"


def init_chains(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> tuple[DataTypeDetectionChain, DateParseChain]:
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
    return detection_chain, date_parse_chain, float_parse_chain


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_data_type_conversion(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> Any:
    db = connect_to_excel(
        file_path=DATA_TYPE_TEST_DATA_FILE, table_name=DATA_TYPE_TEST_TABLE_NAME
    )
    detection_chain, date_parse_chain, float_parse_chain = init_chains(
        fireworksai_mixtral, huggingface_minilm
    )
    convert_to_typed(
        db=db,
        data_type_detection_chain=detection_chain,
        date_parse_chain=date_parse_chain,
        float_parse_chain=float_parse_chain,
    )
