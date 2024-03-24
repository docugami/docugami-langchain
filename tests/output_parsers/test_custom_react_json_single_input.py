from typing import Union

import pytest

from docugami_langchain.agents.models import Invocation
from docugami_langchain.output_parsers.custom_react_json_single_input import (
    FINAL_ANSWER_ACTION,
    CustomReActJsonSingleInputOutputParser,
    escape_non_escaped_backslashes,
    replace_null_outside_quotes,
)


# Define a fixture for the parser instance
@pytest.fixture
def parser() -> CustomReActJsonSingleInputOutputParser:
    return CustomReActJsonSingleInputOutputParser()


# Use parametrize to test different scenarios
@pytest.mark.parametrize(
    "text,expected",
    [
        # Case 1: JSON in ReAct format with preceding and following text
        (
            """
(some preceding text)

Action:
```
{
    "tool_name": "xyz some value",
    "tool_input": "xyz some value"
}
```

(some following text)
    """,
            Invocation(
                tool_name="xyz some value",
                tool_input="xyz some value",
            ),
        ),
        # Case 2: JSON embedded without backticks, with preceding and following text
        (
            """
(some preceding text)

{
    "tool_name": "xyz some value",
    "tool_input": "xyz some value"
}

(some following text)
    """,
            Invocation(
                tool_name="xyz some value",
                tool_input="xyz some value",
            ),
        ),
        # Case 3: The entire string is the JSON blob
        (
            """
{
    "tool_name": "xyz some value",
    "tool_input": "xyz some value"
}
    """,
            Invocation(
                tool_name="xyz some value",
                tool_input="xyz some value",
            ),
        ),
        # Case 4: Incorrect format (no JSON or action keys)
        ("No JSON here", "No JSON here"),
        # Case 5: Multiple JSON blobs; expect the first one to be returned
        (
            """
{
    "tool_name": "first action",
    "tool_input": "first input"
}
Random text
{
    "tool_name": "second action",
    "tool_input": "second input"
}
    """,
            Invocation(
                tool_name="first action",
                tool_input="first input",
            ),
        ),
        # Case 6: Empty input
        (
            "",
            "",
        ),
        # Case 7: Input without any JSON but with the final answer action string
        (
            f"Some text before the action. {FINAL_ANSWER_ACTION} Only text after the final answer action.",
            "Only text after the final answer action.",
        ),
        # Case 8: Input with both JSON and text with the final answer string
        # In this case, ensure the JSON is parsed and returned before the final answer action string
        (
            f"""
Some text before.
{FINAL_ANSWER_ACTION} Text after the final answer, but we have JSON after this that:
```
{{
    "tool_name": "action before final answer",
    "tool_input": "input before final answer"
}}
```
    """,
            Invocation(
                tool_name="action before final answer",
                tool_input="input before final answer",
            ),
        ),
        # Case 9: Backslash inside JSON (actual failure seen during testing in staging)
        (
            """Thought: The user has requested gross income data for multiple companies and quarters. I don't have this information in the current document set. I will use the human\_intervention tool
to request the user to provide a query\_financials tool with the necessary data.

Action:
```json
{
  "tool_name": "human_intervention",
  "tool_input": "Please create or update a query_financials tool with data sufficient to answer questions like this one: company, quarter, eps, and gross\_income."
}
```
""",
            Invocation(
                tool_name="human_intervention",
                tool_input="Please create or update a query_financials tool with data sufficient to answer questions like this one: company, quarter, eps, and gross\_income.",
            ),
        ),
        # Case 10: Null value inside JSON (actual failure seen during testing in staging)
        (
            """Question: How much wood can a woodchuck chunk? 
Thought: Just enough to make an XML dev scream, "Please, no more nested tags!".
Action:
```json
{
  "tool_name": "human_intervention",
  "tool_input": null
}
```
""",
            Invocation(
                tool_name="human_intervention",
                tool_input="",
            ),
        ),
    ],
)
def test_parse(
    parser: CustomReActJsonSingleInputOutputParser,
    text: str,
    expected: Union[Invocation, str],
) -> None:
    result = parser.parse(text)

    if isinstance(result, Invocation) and isinstance(expected, Invocation):
        assert (
            result.tool_name == expected.tool_name
        ), f"Expected {expected.tool_name}, got {result.tool_name}"
        assert (
            result.tool_input == expected.tool_input
        ), f"Expected {expected.tool_input}, got {result.tool_input}"
    elif isinstance(result, str) and isinstance(expected, str):
        assert "{" not in result  # no json in output
        assert "}" not in result  # no json in output
        assert result == expected.split(FINAL_ANSWER_ACTION)[-1].strip()
    else:
        raise Exception(f"Mismatched types: {result, expected}")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This is a test.", "This is a test."),  # No backslashes
        (
            "Line break\\nNew line\\nTab\\t",
            "Line break\\nNew line\\nTab\\t",
        ),  # Known escape sequences are not escaped
        (
            "This \\is a test.",
            "This \\\\is a test.",
        ),  # Non-escape backslashes are escaped
        ("\\a\\b\\c", "\\\\a\\\\b\\\\c"),  # Multiple non-escape backslashes are escaped
        ("Line\\nBreak \\and tab\\t", "Line\\nBreak \\\\and tab\\t"),  # Mixed cases
        ("", ""),  # Empty string
        ("\\This is a test", "\\\\This is a test"),  # Backslash at the start is handled
        ("This is a test\\", "This is a test\\\\"),  # Backslash at the end is handled
    ],
)
def test_escape_non_escaped_backslashes(text: str, expected: str) -> None:
    assert escape_non_escaped_backslashes(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Just some text.", "Just some text."),  # No "null"
        (
            "only full word null not substringnullsofstring replaced",
            'only full word "" not substringnullsofstring replaced',
        ),  # Substring null
        (
            "null should become empty.",
            '"" should become empty.',
        ),  # "null" outside quotes
        ('"null" should stay.', '"null" should stay.'),  # "null" inside quotes
        (
            'Outside null and "inside null".',
            'Outside "" and "inside null".',
        ),  # Mix inside and outside
        ("", ""),  # Empty string
        ("NuLl", '""'),  # Case-insensitive "null"
        (
            'Multiple nulls null "null" null.',
            'Multiple nulls "" "null" "".',
        ),  # Multiple instances
        (
            '"Just some quoted text."',
            '"Just some quoted text."',
        ),  # Edge case with quotes but no "null"
    ],
)
def test_replace_null_outside_quotes(text: str, expected: str) -> None:
    assert replace_null_outside_quotes(text) == expected
