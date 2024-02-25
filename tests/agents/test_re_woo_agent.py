import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from docugami_langchain.agents import ReWOOAgent
from tests.common import GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS, GENERAL_KNOWLEDGE_QUESTION, verify_response

TEST_QUESTION = "What is the accident number for the incident in madill, oklahoma?"
TEST_ANSWER_OPTIONS = ["DFW08CA044"]


@pytest.fixture()
def fireworksai_mixtral_rewoo_agent(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    huggingface_retrieval_tool: BaseTool,
) -> ReWOOAgent:
    """
    Fireworks AI ReWOO Agent using mixtral.
    """
    chain = ReWOOAgent(
        llm=fireworksai_mixtral,
        embeddings=huggingface_minilm,
        tools=[huggingface_retrieval_tool],
    )
    return chain


@pytest.fixture()
def openai_gpt35_rewoo_agent(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
    openai_retrieval_tool: BaseTool,
) -> ReWOOAgent:
    """
    OpenAI ReWOO Agent using GPT 3.5.
    """
    chain = ReWOOAgent(
        llm=openai_gpt35, embeddings=openai_ada, tools=[openai_retrieval_tool]
    )
    return chain


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
@pytest.mark.skip("Not working well with Mixtral, needs to be debugged")
def test_fireworksai_rewoo(
    fireworksai_mixtral_rewoo_agent: ReWOOAgent,
) -> None:

    # test general LLM response from agent
    response = fireworksai_mixtral_rewoo_agent.run(GENERAL_KNOWLEDGE_QUESTION)
    result = response.get("result")
    assert result
    verify_response(result, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)

    # test retrieval response from agent
    response = fireworksai_mixtral_rewoo_agent.run(TEST_QUESTION)
    result = response.get("result")
    assert result
    verify_response(result, TEST_ANSWER_OPTIONS)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_rewoo(openai_gpt35_rewoo_agent: ReWOOAgent) -> None:

    # test general LLM response from agent
    response = openai_gpt35_rewoo_agent.run(GENERAL_KNOWLEDGE_QUESTION)
    result = response.get("result")
    assert result
    verify_response(result, GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS)

    # test retrieval response from agent
    response = openai_gpt35_rewoo_agent.run(TEST_QUESTION)
    result = response.get("result")
    assert result
    verify_response(result, TEST_ANSWER_OPTIONS)
