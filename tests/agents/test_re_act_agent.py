import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.agents import ReActAgent
from docugami_langchain.tools.common import BaseDocugamiTool
from tests.agents.common import run_agent_test, run_streaming_agent_test
from tests.common import (
    GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    GENERAL_KNOWLEDGE_QUESTION,
    RAG_ANSWER_FRAGMENTS,
    RAG_ANSWER_WITH_HISTORY_FRAGMENTS,
    RAG_CHAT_HISTORY,
    RAG_QUESTION,
    RAG_QUESTION_WITH_HISTORY,
)


@pytest.fixture()
def fireworksai_mixtral_re_act_agent(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    huggingface_retrieval_tool: BaseDocugamiTool,
    huggingface_query_tool: BaseDocugamiTool,
    huggingface_common_tools: list[BaseDocugamiTool],
) -> ReActAgent:
    """
    Fireworks AI ReAct Agent using mixtral.
    """
    agent = ReActAgent(
        llm=fireworksai_mixtral,
        embeddings=huggingface_minilm,
        tools=[huggingface_retrieval_tool, huggingface_query_tool]
        + huggingface_common_tools,
    )
    return agent


@pytest.fixture()
def openai_gpt35_re_act_agent(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
    openai_retrieval_tool: BaseDocugamiTool,
    openai_query_tool: BaseDocugamiTool,
    openai_common_tools: list[BaseDocugamiTool],
) -> ReActAgent:
    """
    OpenAI ReAct Agent using GPT 3.5.
    """
    agent = ReActAgent(
        llm=openai_gpt35,
        embeddings=openai_ada,
        tools=[openai_retrieval_tool, openai_query_tool] + openai_common_tools,
    )
    return agent


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_re_act(
    fireworksai_mixtral_re_act_agent: ReActAgent,
) -> None:
    # test general LLM response from agent
    run_agent_test(
        fireworksai_mixtral_re_act_agent,
        GENERAL_KNOWLEDGE_QUESTION,
        GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    )

    # test retrieval response from agent
    run_agent_test(
        fireworksai_mixtral_re_act_agent,
        RAG_QUESTION,
        RAG_ANSWER_FRAGMENTS,
    )


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_streamed_re_act(
    fireworksai_mixtral_re_act_agent: ReActAgent,
) -> None:
    # test general LLM response from agent
    await run_streaming_agent_test(
        fireworksai_mixtral_re_act_agent,
        GENERAL_KNOWLEDGE_QUESTION,
        GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    )

    # test retrieval response from agent
    await run_streaming_agent_test(
        fireworksai_mixtral_re_act_agent,
        RAG_QUESTION,
        RAG_ANSWER_FRAGMENTS,
    )


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_streamed_re_act_with_history(
    fireworksai_mixtral_re_act_agent: ReActAgent,
) -> None:
    # test general LLM response from agent
    await run_streaming_agent_test(
        fireworksai_mixtral_re_act_agent,
        RAG_QUESTION_WITH_HISTORY,
        RAG_ANSWER_WITH_HISTORY_FRAGMENTS,
        RAG_CHAT_HISTORY,
    )


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_re_act(openai_gpt35_re_act_agent: ReActAgent) -> None:
    # test general LLM response from agent
    run_agent_test(
        openai_gpt35_re_act_agent,
        GENERAL_KNOWLEDGE_QUESTION,
        GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    )

    # test retrieval response from agent
    run_agent_test(
        openai_gpt35_re_act_agent,
        RAG_QUESTION,
        RAG_ANSWER_FRAGMENTS,
    )


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.asyncio
async def test_openai_streamed_re_act(
    openai_gpt35_re_act_agent: ReActAgent,
) -> None:
    # test general LLM response from agent
    await run_streaming_agent_test(
        openai_gpt35_re_act_agent,
        GENERAL_KNOWLEDGE_QUESTION,
        GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    )

    # test retrieval response from agent
    await run_streaming_agent_test(
        openai_gpt35_re_act_agent,
        RAG_QUESTION,
        RAG_ANSWER_FRAGMENTS,
    )


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
@pytest.mark.asyncio
async def test_openai_streamed_re_act_with_history(
    openai_gpt35_re_act_agent: ReActAgent,
) -> None:
    # test general LLM response from agent
    await run_streaming_agent_test(
        openai_gpt35_re_act_agent,
        RAG_QUESTION_WITH_HISTORY,
        RAG_ANSWER_WITH_HISTORY_FRAGMENTS,
        RAG_CHAT_HISTORY,
    )
