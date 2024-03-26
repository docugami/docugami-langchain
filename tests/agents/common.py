import logging
import time
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
    step_deltas: list = []
    start_time = time.time()  # Start timing before the test begins

    async for incremental_response in agent.run_stream(
        question=question,
        chat_history=chat_history,
    ):

        streamed_answer = incremental_response.value.get("cited_answer")
        if streamed_answer:
            streamed_answers.append(streamed_answer)
            current_time = time.time()
            step_deltas.append(
                (
                    streamed_answer.answer,
                    round(
                        current_time - start_time - sum([s[1] for s in step_deltas]), 2
                    ),
                )
            )

        last_response = incremental_response

    # Log the spacing between steps
    if len(step_deltas) > 1:
        logging.info("Streamed answers, with time deltas")
        for s in step_deltas:
            logging.info(f"{s[0]}|{s[1]}")

    assert streamed_answers
    assert last_response.run_id
    cited_answer: Optional[CitedAnswer] = last_response.value.get("cited_answer")

    assert cited_answer
    assert cited_answer.is_final
    verify_value(cited_answer.answer, answer_options)
