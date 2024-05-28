import os
from typing import Any

import pytest
import torch
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.types.float_parse_chain import FloatParseChain
from tests.common import TEST_DATA_DIR, verify_traced_response

TEST_DATA = [
    ("there is a value in here 42 somewehre", 42.0),
    ("sometimes two is better than five", 2.0),
    ("12.0", 12.0),
]


def init_chain(llm: BaseLanguageModel, embeddings: Embeddings) -> FloatParseChain:
    chain = FloatParseChain(llm=llm, embeddings=embeddings)
    chain.load_examples(TEST_DATA_DIR / "examples/test_float_parse_examples.yaml")
    return chain


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available, skipping")
@pytest.mark.skipif(
    torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024) < 15,
    reason="Not enough GPU memory to load model, need a larger GPU e.g. a 16GB T4",
)
@pytest.mark.parametrize("text,expected", TEST_DATA)
def test_local_float_parse(
    local_mistral7b: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    text: str,
    expected: float,
) -> Any:
    chain = init_chain(local_mistral7b, huggingface_minilm)
    response = chain.run(text)
    verify_traced_response(response)
    assert expected == response.value


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
@pytest.mark.parametrize("text,expected", TEST_DATA)
def test_fireworksai_llama3_float_parse(
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    text: str,
    expected: float,
) -> Any:
    chain = init_chain(fireworksai_llama3, huggingface_minilm)
    response = chain.run(text)
    verify_traced_response(response)
    assert expected == response.value


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.parametrize("text,expected", TEST_DATA)
def test_openai_gpt4_float_parse(
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
    text: str,
    expected: float,
) -> Any:
    chain = init_chain(openai_gpt4, openai_ada)
    response = chain.run(text)
    verify_traced_response(response)
    assert expected == response.value
