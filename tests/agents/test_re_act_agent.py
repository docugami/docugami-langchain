import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from docugami_langchain.agents import ReActAgent
from tests.common import verify_chain_response

TEST_QUESTION = "What is the accident number for the incident in madill, oklahoma?"
TEST_ANSWER_OPTIONS = ["DFW08CA044"]


@pytest.fixture()
def fireworksai_mixtral_re_act_agent(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    huggingface_retrieval_tool: BaseTool,
) -> ReActAgent:
    """
    Fireworks AI ReAct Agent using mixtral.
    """
    chain = ReActAgent(
        llm=fireworksai_mixtral,
        embeddings=huggingface_minilm,
        tools=[huggingface_retrieval_tool],
    )
    return chain


@pytest.fixture()
def openai_gpt35_re_act_agent(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
    openai_retrieval_tool: BaseTool,
) -> ReActAgent:
    """
    OpenAI ReAct Agent using GPT 3.5.
    """
    chain = ReActAgent(
        llm=openai_gpt35, embeddings=openai_ada, tools=[openai_retrieval_tool]
    )
    return chain


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_re_act(
    fireworksai_mixtral_re_act_agent: ReActAgent,
) -> None:
    
    # test general LLM response from agent
    response = fireworksai_mixtral_re_act_agent.run(
        "Who formulated the theory of special relativity?"
    )
    result = response.get("result")
    assert result
    verify_chain_response(result, ["einstein"])

    # test retrieval response from agent
    response = fireworksai_mixtral_re_act_agent.run(TEST_QUESTION)
    result = response.get("result")
    assert result
    verify_chain_response(result, TEST_ANSWER_OPTIONS)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_re_act(openai_gpt35_re_act_agent: ReActAgent) -> None:

    # test general LLM response from agent
    response = openai_gpt35_re_act_agent.run(
        "Who formulated the theory of special relativity?"
    )
    result = response.get("result")
    assert result
    verify_chain_response(result, ["einstein"])

    # test retrieval response from agent
    response = openai_gpt35_re_act_agent.run(TEST_QUESTION)
    result = response.get("result")
    assert result
    verify_chain_response(result, TEST_ANSWER_OPTIONS)
