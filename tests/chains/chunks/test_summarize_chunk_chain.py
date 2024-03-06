import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.chunks.summarize_chunk_chain import SummarizeChunkChain
from tests.common import TEST_DATA_DIR, verify_response

TEST_FORCE_MAJEURE_CLAUSE = """In no event shall the Trustee be responsible or liable for any failure or delay in the performance of its obligations hereunder
arising out of or caused by, directly or indirectly, forces beyond its control, including, without limitation, strikes, work stoppages, accidents, acts of war
or terrorism, civil or military disturbances, nuclear or natural catastrophes or acts of God, and interruptions, loss or malfunctions of utilities, communications
or computer (software and hardware) services; it being understood that the Trustee shall use reasonable efforts which are consistent with accepted practices
in the banking industry to resume performance as soon as practicable under the circumstances."""


@pytest.fixture()
def fireworksai_mixtral_summarize_chunk_chain(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> SummarizeChunkChain:
    """
    Fireworks AI chain to do chunk summarize using mixtral.
    """
    chain = SummarizeChunkChain(llm=fireworksai_mixtral, embeddings=huggingface_minilm)
    chain.load_examples(TEST_DATA_DIR / "examples/test_summarize_chunk_examples.yaml")
    # avoid short circuiting since the test clause is smaller than default threshold
    chain.min_length_to_summarize = len(TEST_FORCE_MAJEURE_CLAUSE) - 1
    return chain


@pytest.fixture()
def openai_gpt35_summarize_chunk_chain(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> SummarizeChunkChain:
    """
    OpenAI chain to do chunk summarize using GPT 3.5.
    """
    chain = SummarizeChunkChain(llm=openai_gpt35, embeddings=openai_ada)
    chain.load_examples(TEST_DATA_DIR / "examples/test_summarize_chunk_examples.yaml")
    # avoid short circuiting since the test clause is smaller than default threshold
    chain.min_length_to_summarize = len(TEST_FORCE_MAJEURE_CLAUSE) - 1
    return chain


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_summarize_chunk(
    fireworksai_mixtral_summarize_chunk_chain: SummarizeChunkChain,
) -> None:
    summary = fireworksai_mixtral_summarize_chunk_chain.run(TEST_FORCE_MAJEURE_CLAUSE)
    verify_response(summary)
    assert len(summary.value) < len(TEST_FORCE_MAJEURE_CLAUSE)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_summarize_chunk(
    openai_gpt35_summarize_chunk_chain: SummarizeChunkChain,
) -> None:
    summary = openai_gpt35_summarize_chunk_chain.run(TEST_FORCE_MAJEURE_CLAUSE)
    verify_response(summary)
    assert len(summary.value) < len(TEST_FORCE_MAJEURE_CLAUSE)
