import os
from typing import Any

import pytest
import torch
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.types.timespan_parse_chain import TimespanParseChain
from docugami_langchain.output_parsers.timespan import TimeSpan
from tests.common import TEST_DATA_DIR, verify_traced_response

TEST_MESSY_TIMESPAN = "twnty-four hours"
TEST_PARSED_TIMESPAN = TimeSpan("0:0:0:24:0:0")


@pytest.fixture()
def local_timespan_parse_chain(
    local_mistral7b: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> TimespanParseChain:
    """
    Local chain to do timespan parsing.
    """
    chain = TimespanParseChain(llm=local_mistral7b, embeddings=huggingface_minilm)
    chain.load_examples(TEST_DATA_DIR / "examples/test_timespan_parse_examples.yaml")
    return chain


@pytest.fixture()
def fireworksai_timespan_parse_chain(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> TimespanParseChain:
    """
    FireworksAI chain to do timespan parsing.
    """
    chain = TimespanParseChain(llm=fireworksai_mixtral, embeddings=huggingface_minilm)
    chain.load_examples(TEST_DATA_DIR / "examples/test_timespan_parse_examples.yaml")
    return chain


@pytest.fixture()
def openai_timespan_parse_chain(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> TimespanParseChain:
    """
    OpenAI chain to do timespan parsing.
    """
    chain = TimespanParseChain(llm=openai_gpt35, embeddings=openai_ada)
    chain.load_examples(TEST_DATA_DIR / "examples/test_timespan_parse_examples.yaml")
    return chain


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available, skipping")
@pytest.mark.skipif(
    torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024) < 15,
    reason="Not enough GPU memory to load model, need a larger GPU e.g. a 16GB T4",
)
def test_local_timespan_parse(local_timespan_parse_chain: TimespanParseChain) -> Any:
    response = local_timespan_parse_chain.run(TEST_MESSY_TIMESPAN)
    verify_traced_response(response)
    assert TEST_PARSED_TIMESPAN == response.value


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_timespan_parse(
    fireworksai_timespan_parse_chain: TimespanParseChain,
) -> Any:
    response = fireworksai_timespan_parse_chain.run(TEST_MESSY_TIMESPAN)
    verify_traced_response(response)
    assert TEST_PARSED_TIMESPAN == response.value


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_timespan_parse(openai_timespan_parse_chain: TimespanParseChain) -> Any:
    response = openai_timespan_parse_chain.run(TEST_MESSY_TIMESPAN)
    verify_traced_response(response)
    assert TEST_PARSED_TIMESPAN == response.value
