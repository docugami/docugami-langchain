import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from langchain_docugami.chains.answer_chain import AnswerChain
from langchain_docugami.chains.base import TracedChainResponse
from tests.conftest import verify_chain_response


@pytest.fixture()
def fireworksai_mixtral_answer_chain(
    fireworksai_mixtral: BaseLanguageModel, huggingface_minilm: Embeddings
) -> AnswerChain:
    """
    FireworksAI endpoint chain to do generic answers using mixtral.
    """
    return AnswerChain(llm=fireworksai_mixtral, embeddings=huggingface_minilm)


@pytest.fixture()
def openai_gpt35_answer_chain(
    openai_gpt35: BaseLanguageModel, openai_ada: Embeddings
) -> AnswerChain:
    """
    OpenAI chain to do generic anwers using GPT 3.5.
    """
    return AnswerChain(llm=openai_gpt35, embeddings=openai_ada)


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="FireworksAI API token not set"
)
def test_fireworksai_answer(fireworksai_mixtral_answer_chain: AnswerChain) -> None:
    answer = fireworksai_mixtral_answer_chain.run(
        "Who formulated the theory of special relativity?"
    )
    verify_chain_response(answer, ["einstein"])


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="FireworksAI API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_streamed_answer(
    fireworksai_mixtral_answer_chain: AnswerChain,
) -> None:
    chain_response = TracedChainResponse[str](value="")
    async for incremental_response in fireworksai_mixtral_answer_chain.run_stream(
        "Who formulated the theory of special relativity?"
    ):
        chain_response = incremental_response

    assert chain_response.value
    assert chain_response.run_id

    verify_chain_response(chain_response.value, ["einstein"])


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_answer(openai_gpt35_answer_chain: AnswerChain) -> None:
    answer = openai_gpt35_answer_chain.run(
        "Who formulated the theory of special relativity?"
    )
    verify_chain_response(answer, ["einstein"])


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_traced_answer(openai_gpt35_answer_chain: AnswerChain) -> None:
    response = openai_gpt35_answer_chain.traced_run(
        question="Who formulated the theory of special relativity?"
    )

    assert response.run_id
    verify_chain_response(response.value, ["einstein"])


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.asyncio
async def test_openai_streamed_answer(openai_gpt35_answer_chain: AnswerChain) -> None:
    chain_response = TracedChainResponse[str](value="")
    async for incremental_response in openai_gpt35_answer_chain.run_stream(
        "Who formulated the theory of special relativity?"
    ):
        chain_response = incremental_response

    assert chain_response.value
    assert chain_response.run_id

    verify_chain_response(chain_response.value, ["einstein"])
