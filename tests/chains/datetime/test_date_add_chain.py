import os
from datetime import datetime
from typing import Any

import pytest
import torch
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.datetime.date_add_chain import DateAddChain
from tests.common import TEST_DATA_DIR, verify_traced_response

TEST_MESSY_START_DATE = "at around $ept3mb3r I5 in thee year 2Oo3"
TEST_MESSY_END_DATE_OR_DURATION = "on the fivth annivrsary"
TEST_ADDED_DATE: datetime = datetime(2008, 9, 15)


@pytest.fixture()
def local_date_add_chain(
    local_mistral7b: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> DateAddChain:
    """
    Local chain to do date addition.
    """
    chain = DateAddChain(llm=local_mistral7b, embeddings=huggingface_minilm)
    chain.load_examples(TEST_DATA_DIR / "examples/test_date_add_examples.yaml")
    return chain


@pytest.fixture()
def fireworksai_date_add_chain(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> DateAddChain:
    """
    FireworksAI chain to do date addition.
    """
    chain = DateAddChain(llm=fireworksai_mixtral, embeddings=huggingface_minilm)
    chain.load_examples(TEST_DATA_DIR / "examples/test_date_add_examples.yaml")
    return chain


@pytest.fixture()
def openai_date_add_chain(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> DateAddChain:
    """
    OpenAI chain to do date addition.
    """
    chain = DateAddChain(llm=openai_gpt35, embeddings=openai_ada)
    chain.load_examples(TEST_DATA_DIR / "examples/test_date_add_examples.yaml")
    return chain


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available, skipping")
@pytest.mark.skipif(
    torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024) < 15,
    reason="Not enough GPU memory to load model, need a larger GPU e.g. a 16GB T4",
)
def test_local_date_add(local_date_add_chain: DateAddChain) -> Any:
    response = local_date_add_chain.run(
        TEST_MESSY_START_DATE, TEST_MESSY_END_DATE_OR_DURATION
    )
    verify_traced_response(response)
    assert TEST_ADDED_DATE == response.value


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_date_add(fireworksai_date_add_chain: DateAddChain) -> Any:
    response = fireworksai_date_add_chain.run(
        TEST_MESSY_START_DATE, TEST_MESSY_END_DATE_OR_DURATION
    )
    verify_traced_response(response)
    assert TEST_ADDED_DATE == response.value


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_date_add(openai_date_add_chain: DateAddChain) -> Any:
    response = openai_date_add_chain.run(
        TEST_MESSY_START_DATE, TEST_MESSY_END_DATE_OR_DURATION
    )
    verify_traced_response(response)
    assert TEST_ADDED_DATE == response.value
