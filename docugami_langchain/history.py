from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def chat_history_to_messages(chat_history: list[tuple[str, str]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []

    if chat_history:
        for human, ai in chat_history:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=f"{ai}"))
    return messages


def chat_history_to_str(chat_history: list[tuple[str, str]]) -> str:

    if not chat_history:
        return ""

    messages: str = ""
    if chat_history:
        for human, ai in chat_history:
            messages += f"Human: {human}\n"
            messages += f"AI: {ai}\n"
    return "\n" + messages
