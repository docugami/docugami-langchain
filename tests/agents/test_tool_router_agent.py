import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from docugami_langchain.agents import ToolRouterAgent
from docugami_langchain.agents.base import AgentState
from docugami_langchain.base_runnable import TracedResponse
from tests.common import (
    GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    GENERAL_KNOWLEDGE_QUESTION,
    RAG_ANSWER_FRAGMENTS,
    RAG_ANSWER_WITH_HISTORY_FRAGMENTS,
    RAG_CHAT_HISTORY,
    RAG_QUESTION,
    RAG_QUESTION_WITH_HISTORY,
    verify_response,
)


@pytest.fixture()
def fireworksai_mixtral_tool_router_agent(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    huggingface_retrieval_tool: BaseTool,
    huggingface_query_tool: BaseTool,
    huggingface_common_tools: list[BaseTool],
) -> ToolRouterAgent:
    """
    Fireworks AI ReAct Agent using mixtral.
    """
    agent = ToolRouterAgent(
        llm=fireworksai_mixtral,
        embeddings=huggingface_minilm,
        tools=[huggingface_retrieval_tool, huggingface_query_tool]
        + huggingface_common_tools,
    )
    return agent


@pytest.fixture()
def openai_gpt35_tool_router_agent(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
    openai_retrieval_tool: BaseTool,
    openai_query_tool: BaseTool,
    openai_common_tools: list[BaseTool],
) -> ToolRouterAgent:
    """
    OpenAI ReAct Agent using GPT 3.5.
    """
    agent = ToolRouterAgent(
        llm=openai_gpt35,
        embeddings=openai_ada,
        tools=[openai_retrieval_tool, openai_query_tool] + openai_common_tools,
    )
    return agent


def _runtest(
    agent: ToolRouterAgent,
    question: str,
    answer_options: list[str],
    chat_history: list[tuple[str, str]] = [],
) -> None:
    response = agent.run(
        question=question,
        chat_history=chat_history,
    )
    verify_response(response, answer_options)


async def _runtest_streamed(
    agent: ToolRouterAgent,
    question: str,
    answer_options: list[str],
    chat_history: list[tuple[str, str]] = [],
) -> None:
    last_response = TracedResponse[AgentState](value={})  # type: ignore

    steps: list = []
    async for incremental_response in agent.run_stream(
        question=question,
        chat_history=chat_history,
    ):
        step = incremental_response.value.get("current_answer")
        steps.append(step)

        last_response = incremental_response

    assert steps
    verify_response(last_response, answer_options)


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_tool_router(
    fireworksai_mixtral_tool_router_agent: ToolRouterAgent,
) -> None:
    # test general LLM response from agent
    _runtest(
        fireworksai_mixtral_tool_router_agent,
        GENERAL_KNOWLEDGE_QUESTION,
        GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    )

    # test retrieval response from agent
    _runtest(
        fireworksai_mixtral_tool_router_agent,
        RAG_QUESTION,
        RAG_ANSWER_FRAGMENTS,
    )


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_streamed_tool_router(
    fireworksai_mixtral_tool_router_agent: ToolRouterAgent,
) -> None:
    # test general LLM response from agent
    await _runtest_streamed(
        fireworksai_mixtral_tool_router_agent,
        GENERAL_KNOWLEDGE_QUESTION,
        GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    )

    # test retrieval response from agent
    await _runtest_streamed(
        fireworksai_mixtral_tool_router_agent,
        RAG_QUESTION,
        RAG_ANSWER_FRAGMENTS,
    )


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_streamed_tool_router_with_history(
    fireworksai_mixtral_tool_router_agent: ToolRouterAgent,
) -> None:
    # test general LLM response from agent
    await _runtest_streamed(
        fireworksai_mixtral_tool_router_agent,
        RAG_QUESTION_WITH_HISTORY,
        RAG_ANSWER_WITH_HISTORY_FRAGMENTS,
        RAG_CHAT_HISTORY,
    )


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_tool_router(openai_gpt35_tool_router_agent: ToolRouterAgent) -> None:
    # test general LLM response from agent
    _runtest(
        openai_gpt35_tool_router_agent,
        GENERAL_KNOWLEDGE_QUESTION,
        GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    )

    # test retrieval response from agent
    _runtest(
        openai_gpt35_tool_router_agent,
        RAG_QUESTION,
        RAG_ANSWER_FRAGMENTS,
    )


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.asyncio
async def test_openai_streamed_tool_router(
    openai_gpt35_tool_router_agent: ToolRouterAgent,
) -> None:
    # test general LLM response from agent
    await _runtest_streamed(
        openai_gpt35_tool_router_agent,
        GENERAL_KNOWLEDGE_QUESTION,
        GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    )

    # test retrieval response from agent
    await _runtest_streamed(
        openai_gpt35_tool_router_agent,
        RAG_QUESTION,
        RAG_ANSWER_FRAGMENTS,
    )


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.asyncio
async def test_openai_streamed_tool_router_with_history(
    openai_gpt35_tool_router_agent: ToolRouterAgent,
) -> None:
    # test general LLM response from agent
    await _runtest_streamed(
        openai_gpt35_tool_router_agent,
        RAG_QUESTION_WITH_HISTORY,
        RAG_ANSWER_WITH_HISTORY_FRAGMENTS,
        RAG_CHAT_HISTORY,
    )
