import os
from datetime import datetime
from typing import Any

import pytest
import torch
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.types.date_parse_chain import DateParseChain
from tests.common import TEST_DATA_DIR, verify_traced_response

TEST_MESSY_DATE = "22nndd M@ RCH 2oo7"
TEST_PARSED_DATE: datetime = datetime(2007, 3, 22)


@pytest.fixture()
def local_date_parse_chain(
    local_mistral7b: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> DateParseChain:
    """
    Local chain to do date parsing.
    """
    chain = DateParseChain(llm=local_mistral7b, embeddings=huggingface_minilm)
    chain.load_examples(TEST_DATA_DIR / "examples/test_date_parse_examples.yaml")
    return chain


@pytest.fixture()
def fireworksai_date_parse_chain(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> DateParseChain:
    """
    FireworksAI chain to do date parsing.
    """
    chain = DateParseChain(llm=fireworksai_mixtral, embeddings=huggingface_minilm)
    chain.load_examples(TEST_DATA_DIR / "examples/test_date_parse_examples.yaml")
    return chain


@pytest.fixture()
def openai_date_parse_chain(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> DateParseChain:
    """
    OpenAI chain to do date parsing.
    """
    chain = DateParseChain(llm=openai_gpt35, embeddings=openai_ada)
    chain.load_examples(TEST_DATA_DIR / "examples/test_date_parse_examples.yaml")
    return chain


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available, skipping")
@pytest.mark.skipif(
    torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024) < 15,
    reason="Not enough GPU memory to load model, need a larger GPU e.g. a 16GB T4",
)
def test_local_date_parse(local_date_parse_chain: DateParseChain) -> Any:
    response = local_date_parse_chain.run(TEST_MESSY_DATE)
    verify_traced_response(response)
    assert TEST_PARSED_DATE == response.value


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_date_parse(fireworksai_date_parse_chain: DateParseChain) -> Any:
    response = fireworksai_date_parse_chain.run(TEST_MESSY_DATE)
    verify_traced_response(response)
    assert TEST_PARSED_DATE == response.value


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_date_parse(openai_date_parse_chain: DateParseChain) -> Any:
    response = openai_date_parse_chain.run(TEST_MESSY_DATE)
    verify_traced_response(response)
    assert TEST_PARSED_DATE == response.value
