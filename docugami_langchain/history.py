from typing import Sequence

from docugami_langchain.agents.models import StepState
from docugami_langchain.config import ________SINGLE_TOKEN_LINE________

HUMAN_MARKER = "Human"
AI_MARKER = "AI"


def chat_history_to_str(
    chat_history: list[tuple[str, str]],
    include_human_marker: bool = False,
) -> str:

    if not chat_history:
        return ""

    formatted_history: str = ""
    if chat_history:
        for human, ai in chat_history:
            formatted_history += ________SINGLE_TOKEN_LINE________ + "\n"
            formatted_history += f"{HUMAN_MARKER}: {human}\n"
            formatted_history += ________SINGLE_TOKEN_LINE________ + "\n"
            formatted_history += f"{AI_MARKER}: {ai}\n"

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
