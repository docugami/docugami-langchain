import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from docugami_langchain.agents import ReActAgent
from docugami_langchain.agents.re_act_agent import AgentState
from docugami_langchain.base_runnable import TracedResponse
from tests.common import (
    GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    GENERAL_KNOWLEDGE_QUESTION,
    verify_response,
)

TEST_QUESTION = "What is the accident number for the incident in madill, oklahoma?"
TEST_ANSWER_OPTIONS = ["DFW08CA044"]

TEST_CHAT_HISTORY = [
    (
        "What is the county seat of Marshall county, OK?",
        "Madill is a city in and the county seat of Marshall County, Oklahoma, United States.",
    ),
    (
        "Do you know who it was named after?",
        "It was named in honor of George Alexander Madill, an attorney for the St. Louis-San Francisco Railway.",
    ),
]
TEST_QUESTION_WITH_HISTORY = "List the accident numbers for any aviation incidents that happened at this location?"


@pytest.fixture()
def fireworksai_mixtral_re_act_agent(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    huggingface_retrieval_tool: BaseTool,
) -> ReActAgent:
    """
    Fireworks AI ReAct Agent using mixtral.
    """
    agent = ReActAgent(
        llm=fireworksai_mixtral,
        embeddings=huggingface_minilm,
        tools=[huggingface_retrieval_tool],
    )
    return agent


@pytest.fixture()
def openai_gpt35_re_act_agent(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
    openai_retrieval_tool: BaseTool,
) -> ReActAgent:
    """
    OpenAI ReAct Agent using GPT 3.5.
    """
    agent = ReActAgent(
        llm=openai_gpt35,
        embeddings=openai_ada,
        tools=[openai_retrieval_tool],
    )
    return agent


def _runtest(
    agent: ReActAgent,
    question: str,
    answer_options: list[str],
    chat_history: list[tuple[str, str]] = [],
) -> None:
    response = agent.run(question=question, chat_history=chat_history)
    verify_response(response, answer_options)


async def _runtest_streamed(
    agent: ReActAgent,
    question: str,
    answer_options: list[str],
    chat_history: list[tuple[str, str]] = [],
) -> None:
    response = TracedResponse[AgentState](value={})  # type: ignore

    steps: list = []
    async for incremental_response in agent.run_stream(
        question=question,
        chat_history=chat_history,
    ):
        human_readable_step = agent.to_human_readable(incremental_response.value)
        if human_readable_step not in steps:
            steps.append(human_readable_step)

        response = incremental_response

    assert steps
    verify_response(response, answer_options)


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_re_act(fireworksai_mixtral_re_act_agent: ReActAgent) -> None:
    # test general LLM response from agent
    _runtest(
        fireworksai_mixtral_re_act_agent,
        GENERAL_KNOWLEDGE_QUESTION,
        GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    )

    # test retrieval response from agent
    _runtest(
        fireworksai_mixtral_re_act_agent,
        TEST_QUESTION,
        TEST_ANSWER_OPTIONS,
    )


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_streamed_re_act(
    fireworksai_mixtral_re_act_agent: ReActAgent,
) -> None:
    # test general LLM response from agent
    await _runtest_streamed(
        fireworksai_mixtral_re_act_agent,
        GENERAL_KNOWLEDGE_QUESTION,
        GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    )

    # test retrieval response from agent
    await _runtest_streamed(
        fireworksai_mixtral_re_act_agent,
        TEST_QUESTION,
        TEST_ANSWER_OPTIONS,
    )


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_streamed_re_act_with_history(
    fireworksai_mixtral_re_act_agent: ReActAgent,
) -> None:
    # test general LLM response from agent
    await _runtest_streamed(
        fireworksai_mixtral_re_act_agent,
        TEST_QUESTION_WITH_HISTORY,
        TEST_ANSWER_OPTIONS,
        TEST_CHAT_HISTORY,
    )


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_re_act(openai_gpt35_re_act_agent: ReActAgent) -> None:
    # test general LLM response from agent
    _runtest(
        openai_gpt35_re_act_agent,
        GENERAL_KNOWLEDGE_QUESTION,
        GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    )

    # test retrieval response from agent
    _runtest(
        openai_gpt35_re_act_agent,
        TEST_QUESTION,
        TEST_ANSWER_OPTIONS,
    )


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.asyncio
async def test_openai_streamed_re_act(openai_gpt35_re_act_agent: ReActAgent) -> None:
    # test general LLM response from agent
    await _runtest_streamed(
        openai_gpt35_re_act_agent,
        GENERAL_KNOWLEDGE_QUESTION,
        GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    )

    # test retrieval response from agent
    await _runtest_streamed(
        openai_gpt35_re_act_agent,
        TEST_QUESTION,
        TEST_ANSWER_OPTIONS,
    )


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.asyncio
async def test_openai_streamed_re_act_with_history(
    openai_gpt35_re_act_agent: ReActAgent,
) -> None:
    # test general LLM response from agent
    await _runtest_streamed(
        openai_gpt35_re_act_agent,
        TEST_QUESTION_WITH_HISTORY,
        TEST_ANSWER_OPTIONS,
        TEST_CHAT_HISTORY,
    )
