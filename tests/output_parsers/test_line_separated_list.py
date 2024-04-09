from typing import Any

from docugami_langchain.output_parsers.line_separated_list import (
    LineSeparatedListOutputParser,
)


def test_unnumbered_list_parse() -> Any:
    text = """a\nb\nc\n"""
    parser = LineSeparatedListOutputParser()
    parsed_text = parser.parse(text)

    assert parsed_text == ["a", "b", "c"]


def test_pleasantary_list_parse() -> Any:
    text = """Sure! here is the list you asked for:\n\na\nb\nc\n"""
    parser = LineSeparatedListOutputParser()
    parsed_text = parser.parse(text)

    assert parsed_text == ["a", "b", "c"]


def test_numbered_list_parse() -> Any:
    text = """1. a\n2. b\n3. c\n"""
    parser = LineSeparatedListOutputParser()
    parsed_text = parser.parse(text)

    assert parsed_text == ["a", "b", "c"]


def test_unescaped_list_parse() -> Any:
    text = (
        """'1. What is the estimated total for the services provided to Inity, Inc.?\\n2. What are the services provided to Zinga, Inc. under the SOW?\\n3. What are the deliverables """
        + """for the SOW with Inity, Inc.?\\n4. What are the terms and conditions for the SOW with Tuvalu, Inc.?\\n5. What is the fixed price for the SOW with Inity, Inc.?
6. What is the SOW end date for Inity, Inc.?
7. What is the start date for the SOW with Inity, Inc.?
8. What are the hours of operation for the SOW with Inity, Inc.?
9. Is there a travel requirement for the SOW with Inity, Inc.?
10. Who are the parties that agreed to the SOW with Inity, Inc.?'"""
    )
    parser = LineSeparatedListOutputParser()
    parsed_text = parser.parse(text)

    assert len(parsed_text) == 10
