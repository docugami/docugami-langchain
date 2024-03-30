import json
import re
from typing import Union

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

from docugami_langchain.agents.models import Invocation
from docugami_langchain.utils.string_cleanup import (
    escape_non_escape_sequence_backslashes,
    replace_null_outside_quotes,
    unescape_escaped_chars_outside_quoted_strings,
)

THOUGHT_MARKER = "Thought:"
OBSERVATION_MARKER = "Observation:"
FINAL_ANSWER_MARKER = "Final Answer:"

STRICT_REACT_PATTERN = re.compile(r"^.*?`{3}(?:json)?\n?(.*?)`{3}.*?$", re.DOTALL)
"""Regex pattern to parse the output strictly, JSON delimited by ``` as instructed in a ReAct prompt."""

SIMPLE_JSON_PATTERN = re.compile(r"(\{[^}]*\})")
"""Regex pattern to just find any simple (non-nested) JSON in the output, not delimited by anything."""


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
        invocation_text = escape_non_escape_sequence_backslashes(invocation_text)

        return json.loads(invocation_text.strip())

    def parse(self, text: str) -> Union[Invocation, str]:
        includes_answer = FINAL_ANSWER_MARKER in text

        try:
            # First, try parsing with STRICT_REACT_PATTERN
            response = self._parse_regex(text, STRICT_REACT_PATTERN)
            tool_name = response.get("tool_name", "")

            if not tool_name:
                raise Exception(f"could not find tool_name in text: {text}")

            return Invocation(
                tool_name=tool_name,
                tool_input=response.get("tool_input", ""),
                log=text,
            )
        except Exception:
            # Next, try parsing with SIMPLE_JSON_PATTERN
            try:
                response = self._parse_regex(text, SIMPLE_JSON_PATTERN)
                tool_name = response.get("tool_name", "")

                if not tool_name:
                    raise Exception(f"could not find tool_name in text: {text}")

                return Invocation(
                    tool_name=tool_name,
                    tool_input=response.get("tool_input", ""),
                    log=text,
                )
            except Exception:
                # If neither pattern matches, handle according to permissive mode
                if not includes_answer:
                    if not self.permissive:
                        raise OutputParserException(
                            f"Could not parse LLM output: {text}"
                        )

                output = text.split(FINAL_ANSWER_MARKER)[-1].strip()

                if "{" in output:
                    raise OutputParserException(
                        f"Potential JSON in parsed output {output}"
                    )

                if THOUGHT_MARKER in output:
                    raise OutputParserException(
                        f"Potential Thought Marker in parsed output {output}"
                    )

                if OBSERVATION_MARKER in output:
                    raise OutputParserException(
                        f"Potential Observation Marker in parsed output {output}"
                    )

                output = escape_non_escape_sequence_backslashes(output)
                output = unescape_escaped_chars_outside_quoted_strings(output)

                return output
