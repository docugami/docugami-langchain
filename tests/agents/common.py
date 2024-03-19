from typing import Optional

from docugami_langchain.agents.base import AgentState, BaseDocugamiAgent
from docugami_langchain.agents.models import CitedAnswer
from docugami_langchain.base_runnable import TracedResponse
from tests.common import verify_value


def run_agent_test(
    agent: BaseDocugamiAgent,
    question: str,
    answer_options: list[str],
    chat_history: list[tuple[str, str]] = [],
) -> None:

    response = agent.run(
        question=question,
        chat_history=chat_history,
    )

    assert response.run_id
    cited_answer: Optional[CitedAnswer] = response.value.get("cited_answer")

    assert cited_answer
    assert cited_answer.is_final
    verify_value(cited_answer.answer, answer_options)


async def run_streaming_agent_test(
    agent: BaseDocugamiAgent,
    question: str,
    answer_options: list[str],
    chat_history: list[tuple[str, str]] = [],
) -> None:
    last_response = TracedResponse[AgentState](value={})  # type: ignore

    streamed_answers: list = []
    async for incremental_response in agent.run_stream(
        question=question,
        chat_history=chat_history,
    ):
        streamed_answer = incremental_response.value.get("cited_answer")
        if streamed_answer:
            streamed_answers.append(streamed_answer)

        last_response = incremental_response

    assert streamed_answers
    assert last_response.run_id
    cited_answer: Optional[CitedAnswer] = last_response.value.get("cited_answer")

    assert cited_answer
    assert cited_answer.is_final
    verify_value(cited_answer.answer, answer_options)
