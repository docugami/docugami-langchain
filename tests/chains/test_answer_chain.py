import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

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
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks AI API token not set"
)
def test_fireworksai_mistral_7b_answer_no_examples(
    fireworksai_mistral_7b: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_mistral_7b, huggingface_minilm, examples=False)
    answer = chain.run(GENERAL_KNOWLEDGE_QUESTION)
    verify_traced_response(answer, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)


@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks AI API token not set"
)
def test_fireworksai_mistral_7b_answer_with_examples(
    fireworksai_mistral_7b: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_mistral_7b, huggingface_minilm, examples=True)
    answer = chain.run(GENERAL_KNOWLEDGE_QUESTION)
    verify_traced_response(answer, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OpenAI API token not set"
)
def test_openai_openai_answer_no_examples(
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    chain = init_chain(openai_gpt4, openai_ada, examples=False)
    answer = chain.run(GENERAL_KNOWLEDGE_QUESTION)
    verify_traced_response(answer, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OpenAI API token not set"
)
def test_openai_openai_answer_with_examples(
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    chain = init_chain(openai_gpt4, openai_ada, examples=True)
    answer = chain.run(GENERAL_KNOWLEDGE_QUESTION)
    verify_traced_response(answer, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)
