import pytest

from docugami_langchain.agents.models import Invocation
from docugami_langchain.output_parsers.custom_react_json_single_input import (
    FINAL_ANSWER_ACTION,
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
        # Case 1: JSON in ReAct format with preceding and following text
        (
            """
    (some preceding text)

    Action:
    ```
    {
      "action": "xyz some value",
      "action_input": "xyz some value"
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
      "action": "xyz some value",
      "action_input": "xyz some value"
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
      "action": "xyz some value",
      "action_input": "xyz some value"
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
      "action": "first action",
      "action_input": "first input"
    }
    Random text
    {
      "action": "second action",
      "action_input": "second input"
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
      "action": "action before final answer",
      "action_input": "input before final answer"
    }}
    ```
    """,
            Invocation(
                tool_name="action before final answer",
                tool_input="input before final answer",
            ),
        ),
    ],
)
def test_parse(
    parser: CustomReActJsonSingleInputOutputParser,
    text: str,
    expected: Invocation,
) -> None:
    result = parser.parse(text)
    if isinstance(result, Invocation):
        assert (
            result.tool_name == expected.tool_name
        ), f"Expected {expected.tool_name}, got {result.tool_name}"
        assert (
            result.tool_input == expected.tool_input
        ), f"Expected {expected.tool_input}, got {result.tool_input}"
    elif isinstance(result, str):
        assert result == text.split(FINAL_ANSWER_ACTION)[-1].strip()
