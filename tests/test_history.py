import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from docugami_langchain.history import (
    get_chat_history_from_messages,
    get_question_from_messages,
)


def test_get_question_from_messages_with_valid_input() -> None:
    messages = [
        HumanMessage(content="Hi"),
        AIMessage(content="Hello"),
        HumanMessage(content="What's the weather?"),
    ]
    assert get_question_from_messages(messages) == "What's the weather?"


def test_get_question_from_messages_with_only_human_message() -> None:
    messages: list[BaseMessage] = [HumanMessage(content="Hello")]
    assert get_question_from_messages(messages) == "Hello"


def test_get_question_from_messages_with_no_human_message() -> None:
    messages: list[BaseMessage] = [AIMessage(content="Hello")]
    with pytest.raises(Exception):
        get_question_from_messages(messages)


def test_get_question_from_messages_with_empty_list() -> None:
    messages: list[BaseMessage] = []
    with pytest.raises(Exception):
        get_question_from_messages(messages)


def test_get_chat_history_from_messages_valid_input() -> None:
    messages: list[BaseMessage] = [
        HumanMessage(content="Hi"),
        AIMessage(content="Hello"),
        HumanMessage(content="What's up?"),
        AIMessage(content="Not much"),
        HumanMessage(content="What's the weather like?"),
    ]
    expected = [("Hi", "Hello"), ("What's up?", "Not much")]
    assert get_chat_history_from_messages(messages) == expected


def test_get_chat_history_from_messages_uneven_history() -> None:
    messages: list[BaseMessage]  = [
        AIMessage(content="foo"),
        HumanMessage(content="Hi"),
        AIMessage(content="Hello"),
        HumanMessage(content="What's up?"),
    ]
    with pytest.raises(Exception):
        get_chat_history_from_messages(messages)


def test_get_chat_history_from_messages_invalid_request_type() -> None:
    messages: list[BaseMessage]  = [
        AIMessage(content="Hi"),
        HumanMessage(content="Hello"),
    ]
    with pytest.raises(Exception):
        get_chat_history_from_messages(messages)


def test_get_chat_history_from_messages_invalid_response_type() -> None:
    messages: list[BaseMessage] = [
        HumanMessage(content="Hi"),
        HumanMessage(content="Hello"),
    ]
    with pytest.raises(Exception):
        get_chat_history_from_messages(messages)


def test_get_chat_history_from_messages_empty_list() -> None:
    messages: list[BaseMessage]  = []
    assert get_chat_history_from_messages(messages) == []
