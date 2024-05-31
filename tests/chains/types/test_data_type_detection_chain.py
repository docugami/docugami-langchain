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

TEST_DATA = [
    (
        "on the 2nd day of September, of the year twenty thirteen.",
        DataTypeWithUnit(type=DataType.DATETIME, unit=""),
    ),
    (
        "Excess Liability/Umbrella coverage with a limit of no less than $9,000,000 per occurrence and in the aggregate (such limit may be "
        + "achieved through increase of limits in underlying policies to reach the level of coverage shown here). This policy shall name "
        + "Client as an additional insured with...",
        DataTypeWithUnit(type=DataType.INTEGER, unit="$"),
    ),
]


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
@pytest.mark.parametrize("text,type", TEST_DATA)
def test_local_data_type_detection(
    local_mistral7b: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    text: str,
    type: DataTypeWithUnit,
) -> Any:
    chain = init_chain(local_mistral7b, huggingface_minilm)
    response = chain.run(text)
    verify_traced_response(response)
    assert response.value == type


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
@pytest.mark.parametrize("text,type", TEST_DATA)
def test_fireworksai_llama3_data_type_detection(
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    text: str,
    type: DataTypeWithUnit,
) -> Any:
    chain = init_chain(fireworksai_llama3, huggingface_minilm)
    response = chain.run(text)
    verify_traced_response(response)
    assert response.value == type


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.parametrize("text,type", TEST_DATA)
def test_openai_gpt4_data_type_detection(
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
    text: str,
    type: DataTypeWithUnit,
) -> Any:
    chain = init_chain(openai_gpt4, openai_ada)
    response = chain.run(text)
    verify_traced_response(response)
    assert response.value == type
