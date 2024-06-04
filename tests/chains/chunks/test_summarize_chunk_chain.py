import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.chunks.summarize_chunk_chain import SummarizeChunkChain
from tests.common import TEST_DATA_DIR, verify_traced_response

TEST_FORCE_MAJEURE_CLAUSE = """In no event shall the Trustee be responsible or liable for any failure or delay in the performance of its obligations hereunder
arising out of or caused by, directly or indirectly, forces beyond its control, including, without limitation, strikes, work stoppages, accidents, acts of war
or terrorism, civil or military disturbances, nuclear or natural catastrophes or acts of God, and interruptions, loss or malfunctions of utilities, communications
or computer (software and hardware) services; it being understood that the Trustee shall use reasonable efforts which are consistent with accepted practices
in the banking industry to resume performance as soon as practicable under the circumstances."""


def init_chain(llm: BaseLanguageModel, embeddings: Embeddings) -> SummarizeChunkChain:
    chain = SummarizeChunkChain(llm=llm, embeddings=embeddings)
    chain.load_examples(TEST_DATA_DIR / "examples/test_summarize_chunk_examples.yaml")
    chain.min_length_to_summarize = len(TEST_FORCE_MAJEURE_CLAUSE) - 1
    return chain


@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks API token not set"
)
def test_fireworksai_llama3_summarize_chunk(
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_llama3, huggingface_minilm)
    summary = chain.run(TEST_FORCE_MAJEURE_CLAUSE)
    verify_traced_response(summary)
    assert len(summary.value) < len(TEST_FORCE_MAJEURE_CLAUSE)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_gpt4_summarize_chunk(
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    chain = init_chain(openai_gpt4, openai_ada)
    summary = chain.run(TEST_FORCE_MAJEURE_CLAUSE)
    verify_traced_response(summary)
    assert len(summary.value) < len(TEST_FORCE_MAJEURE_CLAUSE)
