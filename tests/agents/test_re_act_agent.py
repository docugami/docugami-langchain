import os

import pytest
from langchain_core.agents import AgentFinish
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from docugami_langchain.agents import ReActAgent
from docugami_langchain.agents.re_act_agent import AgentState
from tests.common import verify_response

TEST_QUESTION = "What is the accident number for the incident in madill, oklahoma?"
TEST_ANSWER_OPTIONS = ["DFW08CA044"]


def _get_answer(response: AgentState) -> str:
    outcome = response.get("agent_outcome")
    assert outcome
    assert isinstance(outcome, AgentFinish)
    assert outcome.return_values
    answer = outcome.return_values.get("output")
    assert answer
    return answer


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
    answer = _get_answer(response)
    verify_response(answer, ["einstein"])

    # test retrieval response from agent
    response = fireworksai_mixtral_re_act_agent.run(TEST_QUESTION)
    answer = _get_answer(response)
    verify_response(answer, TEST_ANSWER_OPTIONS)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_re_act(openai_gpt35_re_act_agent: ReActAgent) -> None:

    # test general LLM response from agent
    response = openai_gpt35_re_act_agent.run(
        "Who formulated the theory of special relativity?"
    )
    answer = _get_answer(response)
    verify_response(answer, ["einstein"])

    # test retrieval response from agent
    response = openai_gpt35_re_act_agent.run(TEST_QUESTION)
    answer = _get_answer(response)
    verify_response(answer, TEST_ANSWER_OPTIONS)
