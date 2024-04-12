import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains import StandaloneQuestionChain
from tests.common import (
    GENERAL_KNOWLEDGE_CHAT_HISTORY,
    GENERAL_KNOWLEDGE_QUESTION,
    GENERAL_KNOWLEDGE_QUESTION_WITH_HISTORY,
    TEST_DATA_DIR,
    verify_traced_response,
)


def init_chain(
    llm: BaseLanguageModel, embeddings: Embeddings, examples: bool = True
) -> StandaloneQuestionChain:
    chain = StandaloneQuestionChain(llm=llm, embeddings=embeddings)
    if examples:
        chain.load_examples(
            TEST_DATA_DIR / "examples/test_standalone_question_examples.yaml"
        )
    return chain


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
def test_fireworksai_mistral_7b_standalone_question_no_history(
    fireworksai_mistral_7b: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_mistral_7b, huggingface_minilm, examples=True)
    standalone_question = chain.run(GENERAL_KNOWLEDGE_QUESTION)
    verify_traced_response(standalone_question)


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_mixtral_streamed_standalone_question_no_history(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_mixtral, huggingface_minilm, examples=True)
    chain_response = TracedResponse[str](value="")
    async for incremental_response in chain.run_stream(GENERAL_KNOWLEDGE_QUESTION):
        chain_response = incremental_response

    verify_traced_response(chain_response)


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_mixtral_streamed_standalone_question_with_history(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_mixtral, huggingface_minilm, examples=True)
    chain_response = TracedResponse[str](value="")
    async for incremental_response in chain.run_stream(
        GENERAL_KNOWLEDGE_QUESTION_WITH_HISTORY,
        GENERAL_KNOWLEDGE_CHAT_HISTORY,
    ):
        chain_response = incremental_response

    verify_traced_response(chain_response)


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_mixtral_streamed_standalone_question_topic_change(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_mixtral, huggingface_minilm, examples=True)
    chain_response = TracedResponse[str](value="")
    async for incremental_response in chain.run_stream(
        "tell me a joke",
        GENERAL_KNOWLEDGE_CHAT_HISTORY,
    ):
        chain_response = incremental_response

    verify_traced_response(chain_response, "joke")


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.asyncio
async def test_openai_gpt4_gpt4_streamed_standalone_question_with_history(
    openai_gpt4: BaseLanguageModel, openai_ada: Embeddings
) -> None:
    chain = init_chain(openai_gpt4, openai_ada, examples=True)
    chain_response = TracedResponse[str](value="")
    async for incremental_response in chain.run_stream(
        GENERAL_KNOWLEDGE_QUESTION_WITH_HISTORY,
        GENERAL_KNOWLEDGE_CHAT_HISTORY,
    ):
        chain_response = incremental_response

    verify_traced_response(chain_response)
