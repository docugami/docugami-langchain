import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.chunks.elaborate_chunk_chain import ElaborateChunkChain
from tests.common import TEST_DATA_DIR, verify_traced_response

TEST_INSTRUCTIONS = "Force Majeure clause absolving Trustee of liability in case of factors outside their control"


def init_chain(llm: BaseLanguageModel, embeddings: Embeddings) -> ElaborateChunkChain:
    chain = ElaborateChunkChain(llm=llm, embeddings=embeddings)
    chain.load_examples(TEST_DATA_DIR / "examples/test_elaborate_chunk_examples.yaml")
    return chain


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_llama3_elaborate_chunk(
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_llama3, huggingface_minilm)
    elaboration = chain.run(TEST_INSTRUCTIONS)
    verify_traced_response(elaboration)
    assert len(elaboration.value) > len(TEST_INSTRUCTIONS)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_gpt4_elaborate_chunk(
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    chain = init_chain(openai_gpt4, openai_ada)
    elaboration = chain.run(TEST_INSTRUCTIONS)
    verify_traced_response(elaboration)
    assert len(elaboration.value) > len(TEST_INSTRUCTIONS)
