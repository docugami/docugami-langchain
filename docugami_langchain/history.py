from docugami_langchain.agents.models import StepState


def chat_history_to_str(chat_history: list[tuple[str, str]]) -> str:

    if not chat_history:
        return ""

    formatted_history: str = ""
    if chat_history:
        for human, ai in chat_history:
            formatted_history += f"Human: {human}\n"
            formatted_history += f"AI: {ai}\n"
    return "\n" + formatted_history


def steps_to_str(steps: list[StepState]) -> str:

    if not steps:
        return ""

    formatted_steps: str = ""
    if steps:
        for step in steps:
            formatted_steps += f"Tool Name: {step.invocation.tool_name}\n"
            formatted_steps += f"\tinput: {step.invocation.tool_input}\n"
            formatted_steps += f"\toutput: {step.output}\n"

    return "\n" + formatted_steps
