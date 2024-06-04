import os
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.types.address_parse_chain import AddressParseChain
from docugami_langchain.chains.types.common import ParsedAddress
from tests.common import TEST_DATA_DIR, verify_traced_response

TEST_DATA = [
    (
        "FooBar corporation was located at 1 Cherry Tree Lane, Edmonds, WA 98026",
        ParsedAddress(
            street="1 Cherry Tree Lane", city="Edmonds", state="WA", zip="98026"
        ),
    ),
    (
        "located at Prenzlauer Allee 10, Leipzig, Freistaat Sachsen 04129, Germany",
        ParsedAddress(
            street="Prenzlauer Allee 10",
            city="Leipzig",
            state="Freistaat Sachsen",
            zip="04129",
        ),
    ),
    (
        "Microsoft Corporation can be found at 1500-1345 Microsoft Way, Redmond, WA 98111",
        ParsedAddress(
            street="1500-1345 Microsoft Way", city="Redmond", state="WA", zip="98111"
        ),
    ),
]


def init_chain(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> AddressParseChain:
    chain = AddressParseChain(llm=llm, embeddings=embeddings)
    chain.load_examples(TEST_DATA_DIR / "examples/test_address_parse_examples.yaml")
    return chain


@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks API token not set"
)
@pytest.mark.parametrize("text,type", TEST_DATA)
def test_fireworksai_llama3_address_detection(
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    text: str,
    type: ParsedAddress,
) -> Any:
    chain = init_chain(fireworksai_llama3, huggingface_minilm)
    response = chain.run(text)
    verify_traced_response(response)
    assert response.value == type


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OpenAI API token not set"
)
@pytest.mark.parametrize("text,type", TEST_DATA)
def test_openai_gpt4_address_detection(
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
    text: str,
    type: ParsedAddress,
) -> Any:
    chain = init_chain(openai_gpt4, openai_ada)
    response = chain.run(text)
    verify_traced_response(response)
    assert response.value == type
