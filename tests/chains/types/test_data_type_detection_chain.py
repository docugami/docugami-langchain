import os
from typing import Any

import pytest
import torch
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.types.common import DataType, DataTypeWithUnit
from docugami_langchain.chains.types.data_type_detection_chain import (
    DataTypeDetectionChain,
)
from tests.common import TEST_DATA_DIR, verify_traced_response

TEST_INPUT_TEXT = "This agreement was signed between Foo and Bar on the 2nd day of September, of the year twenty thirteen."
TEST_PARSED_DATA_TYPE: DataTypeWithUnit = DataTypeWithUnit(
    type=DataType.DATETIME, unit="datetime"
)


def init_chain(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> DataTypeDetectionChain:
    chain = DataTypeDetectionChain(llm=llm, embeddings=embeddings)
    chain.load_examples(
        TEST_DATA_DIR / "examples/test_data_type_detection_examples.yaml"
    )
    return chain


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available, skipping")
@pytest.mark.skipif(
    torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024) < 15,
    reason="Not enough GPU memory to load model, need a larger GPU e.g. a 16GB T4",
)
def test_local_data_type_detection(
    local_mistral7b: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> Any:
    chain = init_chain(local_mistral7b, huggingface_minilm)
    response = chain.run(TEST_INPUT_TEXT)
    verify_traced_response(response)
    assert TEST_PARSED_DATA_TYPE == response.value


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_data_type_detection(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> Any:
    chain = init_chain(fireworksai_mixtral, huggingface_minilm)
    response = chain.run(TEST_INPUT_TEXT)
    verify_traced_response(response)
    assert TEST_PARSED_DATA_TYPE == response.value


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_gpt4_data_type_detection(
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> Any:
    chain = init_chain(openai_gpt4, openai_ada)
    response = chain.run(TEST_INPUT_TEXT)
    verify_traced_response(response)
    assert TEST_PARSED_DATA_TYPE == response.value
