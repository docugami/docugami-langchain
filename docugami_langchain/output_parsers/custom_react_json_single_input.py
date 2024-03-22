import json
import re
from typing import Union

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

from docugami_langchain.agents.models import Invocation

FINAL_ANSWER_ACTION = "Final Answer:"

STRICT_REACT_PATTERN = re.compile(r"^.*?`{3}(?:json)?\n?(.*?)`{3}.*?$", re.DOTALL)
"""Regex pattern to parse the output strictly, delimited by ``` as instructed in a ReAct prompt."""

SIMPLE_JSON_PATTERN = re.compile(
    r'(\{(?:\s*"[^"]+?"\s*:\s*(?:"[^"]*?"|\d+|\[\])\s*,?\s*)+\})'
)
"""Regex pattern to just find any simple JSON objects in the output, not delimited by anything."""


def replace_null_outside_quotes(text: str) -> str:
    """
    Looks for null outside quotes, and if found replaces it with "".
    """

    def replacement(match: re.Match) -> str:
        before = text[: match.start()]
        if before.count('"') % 2 == 0:  # Even number of quotes before 'null'
            return '""'
        else:
            return str(match.group(0))  # 'null' is inside quotes, don't replace

    return re.sub(r"\bnull\b", replacement, text, flags=re.IGNORECASE)


def escape_non_escaped_backslashes(text: str) -> str:
    """
    Escape backslashes that are not part of a known escape sequence.

    Looks for a backslash that is not a part of any known escape characters ('n', 'r', 't', 'f', '\\', '"'), and escapes it.
    """
    return re.sub(r'\\(?!["\\nrtf])', r"\\\\", text)


class CustomReActJsonSingleInputOutputParser(BaseOutputParser[Union[Invocation, str]]):
    """
    A custom version of ReActJsonSingleInputOutputParser from the
    langchain lib with the following changes:

    1. Decouples from langchain dependency and returns a simple custom TypedDict.
    2. If the standard ReAct style output is not found in the text, try to parse
    any json found in the text and return that if it matches the return type.
    3. Permissive parsing mode that assumes unparsable output is final answer,
    since some weaker models fail to respect the ReAct prompt format when producing
    the final answer.

    Ref: libs/langchain/langchain/agents/output_parsers/react_json_single_input.py
    """

    permissive = True
    """Softer parsing. Specifies whether unparsable input is considered final output."""

    @property
    def _type(self) -> str:
        return "custom-react-json-single-input"

    def _parse_regex(self, text: str, regex: re.Pattern[str]) -> dict:
        found = regex.search(text)
        if not found:
            raise ValueError("Invocation text not found")
        invocation_text = found.group(1)
        invocation_text = replace_null_outside_quotes(invocation_text)
        invocation_text = escape_non_escaped_backslashes(invocation_text)

        return json.loads(invocation_text.strip())

    def parse(self, text: str) -> Union[Invocation, str]:
        includes_answer = FINAL_ANSWER_ACTION in text

        try:
            # First, try parsing with STRICT_REACT_PATTERN
            response = self._parse_regex(text, STRICT_REACT_PATTERN)
            action = response.get("action", "")

            if not action:
                raise Exception(f"could not find action in text: {text}")

            return Invocation(
                tool_name=action,
                tool_input=response.get("action_input", ""),
                log=text,
            )
        except Exception:
            # Next, try parsing with SIMPLE_JSON_PATTERN
            try:
                response = self._parse_regex(text, SIMPLE_JSON_PATTERN)
                action = response.get("action", "")

                if not action:
                    raise Exception(f"could not find action in text: {text}")

                return Invocation(
                    tool_name=action,
                    tool_input=response.get("action_input", ""),
                    log=text,
                )
            except Exception:
                # If neither pattern matches, handle according to permissive mode
                if not includes_answer:
                    if not self.permissive:
                        raise OutputParserException(
                            f"Could not parse LLM output: {text}"
                        )

                output = text.split(FINAL_ANSWER_ACTION)[-1].strip()

                if "{" in output:
                    raise OutputParserException(f"Potential JSON in parsed {output}")

                return output
