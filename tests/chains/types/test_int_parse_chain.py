import os
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.types.int_parse_chain import IntParseChain
from tests.common import TEST_DATA_DIR, verify_traced_response

PARSEABLE_TEST_DATA = [
    ("there is a value in here 42 somewehre", 42),
    ("sometimes two is better than five", 2),
    (
        "Excess Liability/Umbrella coverage with a limit of no less than $9,000,000 per occurrence and in the aggregate (such limit may be "
        + "achieved through increase of limits in underlying policies to reach the level of coverage shown here). This policy shall name "
        + "Client as an additional insured with...",
        9000000,
    ),
    ("12  ", 12),  # directly parseable
]

UNPARSEABLE_TEST_DATA_RAISES_EXCEPTION = [
    "hello world",
    "    ",
]


def init_chain(llm: BaseLanguageModel, embeddings: Embeddings) -> IntParseChain:
    chain = IntParseChain(llm=llm, embeddings=embeddings)
    chain.load_examples(TEST_DATA_DIR / "examples/test_int_parse_examples.yaml")
    return chain


@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks API token not set"
)
@pytest.mark.parametrize("text,expected", PARSEABLE_TEST_DATA)
def test_fireworksai_llama3_int_parse(
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    text: str,
    expected: int,
) -> Any:
    chain = init_chain(fireworksai_llama3, huggingface_minilm)
    response = chain.run(text)
    verify_traced_response(response)
    assert expected == response.value


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OpenAI API token not set"
)
@pytest.mark.parametrize("text,expected", PARSEABLE_TEST_DATA)
def test_openai_gpt4_int_parse(
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
    text: str,
    expected: int,
) -> Any:
    chain = init_chain(openai_gpt4, openai_ada)
    response = chain.run(text)
    verify_traced_response(response)
    assert expected == response.value


@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks API token not set"
)
@pytest.mark.parametrize("text", UNPARSEABLE_TEST_DATA_RAISES_EXCEPTION)
def test_fireworksai_llama3_int_parse_error(
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    text: str,
) -> Any:
    with pytest.raises(ValueError):
        chain = init_chain(fireworksai_llama3, huggingface_minilm)
        _ = chain.run(text)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OpenAI API token not set"
)
@pytest.mark.parametrize("text", UNPARSEABLE_TEST_DATA_RAISES_EXCEPTION)
def test_openai_gpt4_int_parse_error(
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
    text: str,
) -> Any:
    with pytest.raises(ValueError):
        chain = init_chain(openai_gpt4, openai_ada)
        _ = chain.run(text)
