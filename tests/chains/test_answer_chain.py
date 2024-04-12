import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.answer_chain import AnswerChain
from tests.common import (
    GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    GENERAL_KNOWLEDGE_QUESTION,
    TEST_DATA_DIR,
    verify_traced_response,
)


def init_chain(
    llm: BaseLanguageModel, embeddings: Embeddings, examples: bool = True
) -> AnswerChain:
    chain = AnswerChain(llm=llm, embeddings=embeddings)
    if examples:
        chain.load_examples(TEST_DATA_DIR / "examples/test_answer_examples.yaml")
    return chain


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
def test_fireworksai_mistral_7b_answer_no_examples(
    fireworksai_mistral_7b: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_mistral_7b, huggingface_minilm, examples=False)
    answer = chain.run(GENERAL_KNOWLEDGE_QUESTION)
    verify_traced_response(answer, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
def test_fireworksai_mistral_7b_answer_with_examples(
    fireworksai_mistral_7b: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_mistral_7b, huggingface_minilm, examples=True)
    answer = chain.run(GENERAL_KNOWLEDGE_QUESTION)
    verify_traced_response(answer, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks AI API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_mixtral_streamed_answer(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_mixtral, huggingface_minilm, examples=True)
    chain_response = TracedResponse[str](value="")
    async for incremental_response in chain.run_stream(GENERAL_KNOWLEDGE_QUESTION):
        chain_response = incremental_response

    verify_traced_response(chain_response, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)
