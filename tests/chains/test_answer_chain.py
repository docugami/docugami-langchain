import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.answer_chain import AnswerChain
from tests.common import (
    GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    GENERAL_KNOWLEDGE_ANSWER_WITH_HISTORY_FRAGMENTS,
    GENERAL_KNOWLEDGE_CHAT_HISTORY,
    GENERAL_KNOWLEDGE_QUESTION,
    GENERAL_KNOWLEDGE_QUESTION_WITH_HISTORY,
    TEST_DATA_DIR,
    verify_response,
)


@pytest.fixture()
def fireworksai_mistral_7b_answer_chain_no_examples(
    fireworksai_mistral_7b: BaseLanguageModel, huggingface_minilm: Embeddings
) -> AnswerChain:
    """
    Fireworks AI endpoint chain to do generic answers using mistral-7b (no examples)
    """
    return AnswerChain(llm=fireworksai_mistral_7b, embeddings=huggingface_minilm)


@pytest.fixture()
def fireworksai_mistral_7b_answer_chain_with_examples(
    fireworksai_mistral_7b: BaseLanguageModel, huggingface_minilm: Embeddings
) -> AnswerChain:
    """
    Fireworks AI endpoint chain to do generic answers using mistral-7b.
    """
    chain = AnswerChain(llm=fireworksai_mistral_7b, embeddings=huggingface_minilm)
    chain.load_examples(TEST_DATA_DIR / "examples/test_answer_examples.yaml")
    return chain


@pytest.fixture()
def fireworksai_mixtral_answer_chain_no_examples(
    fireworksai_mixtral: BaseLanguageModel, huggingface_minilm: Embeddings
) -> AnswerChain:
    """
    Fireworks AI endpoint chain to do generic answers using mixtral (no examples).
    """
    return AnswerChain(llm=fireworksai_mixtral, embeddings=huggingface_minilm)


@pytest.fixture()
def fireworksai_mixtral_answer_chain_with_examples(
    fireworksai_mixtral: BaseLanguageModel, huggingface_minilm: Embeddings
) -> AnswerChain:
    """
    Fireworks AI endpoint chain to do generic answers using mixtral.
    """
    chain = AnswerChain(llm=fireworksai_mixtral, embeddings=huggingface_minilm)
    chain.load_examples(TEST_DATA_DIR / "examples/test_answer_examples.yaml")
    return chain


@pytest.fixture()
def openai_gpt35_answer_chain(
    openai_gpt35: BaseLanguageModel, openai_ada: Embeddings
) -> AnswerChain:
    """
    OpenAI chain to do generic anwers using GPT 3.5.
    """
    chain = AnswerChain(llm=openai_gpt35, embeddings=openai_ada)
    chain.load_examples(TEST_DATA_DIR / "examples/test_answer_examples.yaml")
    return chain


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
def test_fireworksai_mistral_7b_answer_no_examples(
    fireworksai_mistral_7b_answer_chain_no_examples: AnswerChain,
) -> None:
    answer = fireworksai_mistral_7b_answer_chain_no_examples.run(
        GENERAL_KNOWLEDGE_QUESTION
    )
    verify_response(answer, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
def test_fireworksai_mistral_7b_answer_with_examples(
    fireworksai_mistral_7b_answer_chain_with_examples: AnswerChain,
) -> None:
    answer = fireworksai_mistral_7b_answer_chain_with_examples.run(
        GENERAL_KNOWLEDGE_QUESTION
    )
    verify_response(answer, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
def test_fireworksai_mixtral_answer_no_examples(
    fireworksai_mixtral_answer_chain_no_examples: AnswerChain,
) -> None:
    answer = fireworksai_mixtral_answer_chain_no_examples.run(
        GENERAL_KNOWLEDGE_QUESTION
    )
    verify_response(answer, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
def test_fireworksai_mixtral_answer_with_examples(
    fireworksai_mixtral_answer_chain_with_examples: AnswerChain,
) -> None:
    answer = fireworksai_mixtral_answer_chain_with_examples.run(
        GENERAL_KNOWLEDGE_QUESTION
    )
    verify_response(answer, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_mixtral_streamed_answer(
    fireworksai_mixtral_answer_chain_with_examples: AnswerChain,
) -> None:
    chain_response = TracedResponse[str](value="")
    async for (
        incremental_response
    ) in fireworksai_mixtral_answer_chain_with_examples.run_stream(
        GENERAL_KNOWLEDGE_QUESTION
    ):
        chain_response = incremental_response

    verify_response(chain_response, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)

@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_mixtral_streamed_answer_with_history(
    fireworksai_mixtral_answer_chain_with_examples: AnswerChain,
) -> None:
    chain_response = TracedResponse[str](value="")
    async for incremental_response in fireworksai_mixtral_answer_chain_with_examples.run_stream(
        GENERAL_KNOWLEDGE_QUESTION_WITH_HISTORY,
        GENERAL_KNOWLEDGE_CHAT_HISTORY,
    ):
        chain_response = incremental_response

    verify_response(chain_response, GENERAL_KNOWLEDGE_ANSWER_WITH_HISTORY_FRAGMENTS)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_gpt35_answer(openai_gpt35_answer_chain: AnswerChain) -> None:
    answer = openai_gpt35_answer_chain.run(GENERAL_KNOWLEDGE_QUESTION)
    verify_response(answer, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.asyncio
async def test_openai_gpt35_streamed_answer(
    openai_gpt35_answer_chain: AnswerChain,
) -> None:
    chain_response = TracedResponse[str](value="")
    async for incremental_response in openai_gpt35_answer_chain.run_stream(
        GENERAL_KNOWLEDGE_QUESTION
    ):
        chain_response = incremental_response

    verify_response(chain_response, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.asyncio
async def test_openai_gpt35_streamed_answer_with_history(
    openai_gpt35_answer_chain: AnswerChain,
) -> None:
    chain_response = TracedResponse[str](value="")
    async for incremental_response in openai_gpt35_answer_chain.run_stream(
        GENERAL_KNOWLEDGE_QUESTION_WITH_HISTORY,
        GENERAL_KNOWLEDGE_CHAT_HISTORY,
    ):
        chain_response = incremental_response

    verify_response(chain_response, GENERAL_KNOWLEDGE_ANSWER_WITH_HISTORY_FRAGMENTS)
