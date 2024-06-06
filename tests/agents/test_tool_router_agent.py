import os
import threading
from typing import Callable, Optional

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.agents import ToolRouterAgent
from docugami_langchain.chains.rag import ToolFinalAnswerChain, ToolOutputGraderChain
from docugami_langchain.chains.rag.standalone_question_chain import (
    StandaloneQuestionChain,
)
from tests.agents.common import (
    build_test_common_tools,
    build_test_query_tool,
    build_test_retrieval_tool,
    run_agent_test,
    run_streaming_agent_test,
)
from tests.common import (
    GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
    GENERAL_KNOWLEDGE_QUESTION,
    TEST_DATA_DIR,
)
from tests.testdata.docsets.docset_test_data import (
    DOCSET_TEST_DATA,
    DocsetTestData,
)


def init_tool_router_agent(
    docset: DocsetTestData,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    optimization_completion_callback: Optional[
        Callable[[bool, Optional[Exception]], None]
    ] = None,
) -> ToolRouterAgent:

    final_answer_chain = ToolFinalAnswerChain(llm=llm, embeddings=embeddings)
    final_answer_chain.load_examples(
        TEST_DATA_DIR / "examples/test_tool_output_examples.yaml"
    )

    output_grader_chain = ToolOutputGraderChain(llm=llm, embeddings=embeddings)
    output_grader_chain.load_examples(
        TEST_DATA_DIR / "examples/test_tool_output_examples.yaml"
    )

    tools = []
    tools.append(build_test_retrieval_tool(llm, embeddings, docset))
    if docset.report:
        tools.append(
            build_test_query_tool(
                report=docset.report,
                llm=llm,
                embeddings=embeddings,
                optimization_completion_callback=optimization_completion_callback,
            )
        )
    tools += build_test_common_tools(llm, embeddings)

    standalone_questions_chain = StandaloneQuestionChain(
        llm=llm,
        embeddings=embeddings,
    )
    standalone_questions_chain.load_examples(
        TEST_DATA_DIR / "examples/test_standalone_question_examples.yaml"
    )

    agent = ToolRouterAgent(
        llm=llm,
        embeddings=embeddings,
        tools=tools,
        standalone_question_chain=standalone_questions_chain,
        final_answer_chain=final_answer_chain,
        output_grader_chain=output_grader_chain,
    )
    agent.load_examples(TEST_DATA_DIR / "examples/test_tool_router_examples.yaml")
    return agent


@pytest.mark.parametrize("test_data", DOCSET_TEST_DATA)
@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks API token not set"
)
def test_fireworksai_llama3_tool_router(
    test_data: DocsetTestData,
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    optimization_done = threading.Event()

    def on_complete(success: bool, exception: Optional[Exception]) -> None:
        try:
            if success:
                # test general LLM response from agent
                run_agent_test(
                    agent,
                    GENERAL_KNOWLEDGE_QUESTION,
                    GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
                )

                for question in test_data.questions:
                    run_agent_test(
                        agent,
                        question.question,
                        question.acceptable_answer_fragments,
                        question.chat_history,
                        question.acceptable_citation_label_fragments,
                    )
            else:
                if exception:
                    raise exception
                else:
                    raise Exception("Optimization failed")
        finally:
            optimization_done.set()

    agent = init_tool_router_agent(
        docset=test_data,
        llm=fireworksai_llama3,
        embeddings=huggingface_minilm,
        optimization_completion_callback=on_complete,
    )

    # Wait for the optimization to complete
    optimization_done.wait()


@pytest.mark.parametrize("test_data", DOCSET_TEST_DATA)
@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks API token not set"
)
@pytest.mark.asyncio
async def test_fireworksai_llama3_streamed_tool_router(
    test_data: DocsetTestData,
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    optimization_done = threading.Event()

    def on_complete(success: bool, exception: Optional[Exception]) -> None:
        try:
            if success:
                # test general LLM response from agent
                run_streaming_agent_test(
                    agent,
                    GENERAL_KNOWLEDGE_QUESTION,
                    GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
                ).close()

                for question in test_data.questions:
                    run_streaming_agent_test(
                        agent,
                        question.question,
                        question.acceptable_answer_fragments,
                        question.chat_history,
                        question.acceptable_citation_label_fragments,
                    ).close()
            else:
                if exception:
                    raise exception
                else:
                    raise Exception("Optimization failed")
        finally:
            optimization_done.set()

    agent = init_tool_router_agent(
        docset=test_data,
        llm=fireworksai_llama3,
        embeddings=huggingface_minilm,
        optimization_completion_callback=on_complete,
    )

    # Wait for the optimization to complete
    optimization_done.wait()


@pytest.mark.parametrize("test_data", DOCSET_TEST_DATA)
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API token not set")
def test_openai_gpt4_tool_router(
    test_data: DocsetTestData,
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    optimization_done = threading.Event()

    def on_complete(success: bool, exception: Optional[Exception]) -> None:
        try:
            if success:
                # test general LLM response from agent
                run_agent_test(
                    agent,
                    GENERAL_KNOWLEDGE_QUESTION,
                    GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
                )

                for question in test_data.questions:
                    run_agent_test(
                        agent,
                        question.question,
                        question.acceptable_answer_fragments,
                        question.chat_history,
                        question.acceptable_citation_label_fragments,
                    )
            else:
                if exception:
                    raise exception
                else:
                    raise Exception("Optimization failed")
        finally:
            optimization_done.set()

    agent = init_tool_router_agent(
        docset=test_data,
        llm=openai_gpt4,
        embeddings=openai_ada,
        optimization_completion_callback=on_complete,
    )

    # Wait for the optimization to complete
    optimization_done.wait()


@pytest.mark.parametrize("test_data", DOCSET_TEST_DATA)
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API token not set")
@pytest.mark.asyncio
async def test_openai_gpt4_streamed_tool_router(
    test_data: DocsetTestData,
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    optimization_done = threading.Event()

    def on_complete(success: bool, exception: Optional[Exception]) -> None:
        try:
            if success:
                # test general LLM response from agent
                run_streaming_agent_test(
                    agent,
                    GENERAL_KNOWLEDGE_QUESTION,
                    GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS,
                ).close()

                for question in test_data.questions:
                    run_streaming_agent_test(
                        agent,
                        question.question,
                        question.acceptable_answer_fragments,
                        question.chat_history,
                        question.acceptable_citation_label_fragments,
                    ).close()
            else:
                if exception:
                    raise exception
                else:
                    raise Exception("Optimization failed")
        finally:
            optimization_done.set()

    agent = init_tool_router_agent(
        docset=test_data,
        llm=openai_gpt4,
        embeddings=openai_ada,
        optimization_completion_callback=on_complete,
    )

    # Wait for the optimization to complete
    optimization_done.wait()
