from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from docugami_langchain.agents.models import StepState
from docugami_langchain.config import ________SINGLE_TOKEN_LINE________

HUMAN_MARKER = "Human"
AI_MARKER = "AI"


def get_question_from_messages(messages: list[BaseMessage]) -> str:
    """
    Extracts a question (the last HumanMessage) from a list of messages.
    """
    if not messages or not isinstance(messages[-1], HumanMessage):
        raise Exception(
            "Please provide input messages, making sure the last one is a HumanMessage"
        )

    return str(messages[-1].content)


def get_chat_history_from_messages(
    messages: list[BaseMessage],
) -> list[tuple[str, str]]:
    """
    Extracts chat history (a list of request/response string tuples) from a list of messages.
    """
    if not messages or not len(messages) > 1:
        return []

    message_history = messages[:-1]
    if not (len(message_history) % 2 == 0):
        raise Exception(
            "There must be an even number of messages in chat history, i.e. a list of HumanMessage / AIMessage pairs"
        )

    string_history: list[tuple[str, str]] = []
    for i in range(0, len(message_history), 2):
        request = message_history[i]
        if not isinstance(request, HumanMessage):
            raise Exception(f"Request message must be HumanMessage: {request}")

        response = message_history[i + 1]
        if not isinstance(response, AIMessage):
            raise Exception(f"Response message must be AIMessage: {response}")

        string_history.append((str(request.content), str(response.content)))

    return string_history


def chat_history_to_str(
    chat_history: list[tuple[str, str]],
    include_human_marker: bool = False,
) -> str:

    if not chat_history:
        return ""

    formatted_history: str = ""
    if chat_history:
        for human, ai in chat_history:
            formatted_history += f"{HUMAN_MARKER}: {human}\n"
            formatted_history += ________SINGLE_TOKEN_LINE________ + "\n"
            formatted_history += f"{AI_MARKER}: {ai}\n"
            formatted_history += ________SINGLE_TOKEN_LINE________ + "\n"

    formatted_history = "\n" + formatted_history + "\n"

    if include_human_marker:
        formatted_history += HUMAN_MARKER + ": "

    return formatted_history


def steps_to_str(steps: Sequence[StepState]) -> str:

    if not steps:
        return ""

    formatted_steps: str = ""
    if steps:
        for step in steps:
            formatted_steps += f"Tool Name: {step.invocation.tool_name}\n"
            formatted_steps += f"\tinput: {step.invocation.tool_input}\n"
            formatted_steps += f"\toutput: {step.output}\n"

    return "\n" + formatted_steps
