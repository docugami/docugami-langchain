from typing import Union

import pytest

from docugami_langchain.agents.models import Invocation
from docugami_langchain.agents.re_act_agent import FINAL_ANSWER_MARKER
from docugami_langchain.output_parsers.custom_react_json_single_input import (
    CustomReActJsonSingleInputOutputParser,
)


# Define a fixture for the parser instance
@pytest.fixture
def parser() -> CustomReActJsonSingleInputOutputParser:
    return CustomReActJsonSingleInputOutputParser()


# Use parametrize to test different scenarios
@pytest.mark.parametrize(
    "text,expected",
    [
        # JSON in ReAct format with preceding and following text
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
        # JSON embedded without backticks, with preceding and following text
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
        # The entire string is the JSON blob
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
        # Multiple JSON blobs; expect the first one to be returned
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
        # Input without any JSON but with the final answer action string
        (
            f"Some text before the action. {FINAL_ANSWER_MARKER} Only text after the final answer action.",
            "Only text after the final answer action.",
        ),
        # Input with both JSON and text with the final answer string
        # In this case, ensure the JSON is parsed and returned before the final answer action string
        (
            f"""
Some text before.
{FINAL_ANSWER_MARKER} Text after the final answer, but we have JSON after this that:
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
        # Backslash inside JSON (actual failure seen during testing in staging)
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
        # Null value inside JSON (actual failure seen during testing in staging)
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
        # JSON without delimiters, and nested quotes
        (
            """Thought: I can use the query_aviation_incidents_report tool to find the accident number for the incident in Madill, Oklahoma.

Action:
{
  "tool_name": "query_aviation_incidents_report",
  "tool_input": "SELECT \\"Accident Number\\" FROM \\"Aviation Incidents Report\\" WHERE \\"Location\\" LIKE '%Madill, Oklahoma%'"
}""",
            Invocation(
                tool_name="query_aviation_incidents_report",
                tool_input='SELECT "Accident Number" FROM "Aviation Incidents Report" WHERE "Location" LIKE \'%Madill, Oklahoma%\'',
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
        assert result == expected.split(FINAL_ANSWER_MARKER)[-1].strip()
    else:
        raise Exception(f"Mismatched types: {result, expected}")
