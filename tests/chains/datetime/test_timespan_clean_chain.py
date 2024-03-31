import os
from typing import Any

import pytest
import torch
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.datetime.timespan_clean_chain import TimespanCleanChain
from docugami_langchain.output_parsers.timespan import TimeSpan
from tests.common import TEST_DATA_DIR, verify_traced_response

TEST_MESSY_TIMESPAN = "twnty-four hours"
TEST_CLEANED_TIMESPAN = TimeSpan("0:0:0:24:0:0")


@pytest.fixture()
def local_timespan_clean_chain(
    local_mistral7b: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> TimespanCleanChain:
    """
    Local chain to do timespan cleanup.
    """
    chain = TimespanCleanChain(llm=local_mistral7b, embeddings=huggingface_minilm)
    chain.load_examples(TEST_DATA_DIR / "examples/test_timespan_clean_examples.yaml")
    return chain


@pytest.fixture()
def fireworksai_timespan_clean_chain(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> TimespanCleanChain:
    """
    FireworksAI chain to do timespan cleanup.
    """
    chain = TimespanCleanChain(llm=fireworksai_mixtral, embeddings=huggingface_minilm)
    chain.load_examples(TEST_DATA_DIR / "examples/test_timespan_clean_examples.yaml")
    return chain


@pytest.fixture()
def openai_timespan_clean_chain(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> TimespanCleanChain:
    """
    OpenAI chain to do timespan cleanup.
    """
    chain = TimespanCleanChain(llm=openai_gpt35, embeddings=openai_ada)
    chain.load_examples(TEST_DATA_DIR / "examples/test_timespan_clean_examples.yaml")
    return chain


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available, skipping")
@pytest.mark.skipif(
    torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024) < 15,
    reason="Not enough GPU memory to load model, need a larger GPU e.g. a 16GB T4",
)
def test_local_timespan_clean(local_timespan_clean_chain: TimespanCleanChain) -> Any:
    response = local_timespan_clean_chain.run(TEST_MESSY_TIMESPAN)
    verify_traced_response(response)
    assert TEST_CLEANED_TIMESPAN == response.value


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_timespan_clean(
    fireworksai_timespan_clean_chain: TimespanCleanChain,
) -> Any:
    response = fireworksai_timespan_clean_chain.run(TEST_MESSY_TIMESPAN)
    verify_traced_response(response)
    assert TEST_CLEANED_TIMESPAN == response.value


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_timespan_clean(openai_timespan_clean_chain: TimespanCleanChain) -> Any:
    response = openai_timespan_clean_chain.run(TEST_MESSY_TIMESPAN)
    verify_traced_response(response)
    assert TEST_CLEANED_TIMESPAN == response.value
