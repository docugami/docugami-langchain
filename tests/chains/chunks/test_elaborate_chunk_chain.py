import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.chunks.elaborate_chunk_chain import ElaborateChunkChain
from tests.conftest import TEST_DATA_DIR, verify_chain_response

TEST_INSTRUCTIONS = "Force Majeure clause absolving Trustee of liability in case of factors outside their control"


@pytest.fixture()
def fireworksai_mixtral_elaborate_chunk_chain(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> ElaborateChunkChain:
    """
    FireworksAI endpoint chain to do chunk elaborate using mixtral.
    """
    chain = ElaborateChunkChain(llm=fireworksai_mixtral, embeddings=huggingface_minilm)
    chain.load_examples(TEST_DATA_DIR / "examples/test_elaborate_chunk_examples.yaml")
    return chain


@pytest.fixture()
def openai_gpt35_elaborate_chunk_chain(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> ElaborateChunkChain:
    """
    OpenAI chain to do chunk elaborate using GPT 3.5.
    """
    chain = ElaborateChunkChain(llm=openai_gpt35, embeddings=openai_ada)
    chain.load_examples(TEST_DATA_DIR / "examples/test_elaborate_chunk_examples.yaml")
    return chain


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_elaborate_chunk(
    fireworksai_mixtral_elaborate_chunk_chain: ElaborateChunkChain,
) -> None:
    elaboration = fireworksai_mixtral_elaborate_chunk_chain.run(TEST_INSTRUCTIONS)
    verify_chain_response(elaboration)
    assert elaboration
    assert len(elaboration) > len(TEST_INSTRUCTIONS)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_elaborate_chunk(
    openai_gpt35_elaborate_chunk_chain: ElaborateChunkChain,
) -> None:
    elaboration = openai_gpt35_elaborate_chunk_chain.run(TEST_INSTRUCTIONS)
    verify_chain_response(elaboration)
    assert elaboration
    assert len(elaboration) > len(TEST_INSTRUCTIONS)
