import json
import re
from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

FINAL_ANSWER_ACTION = "Final Answer:"


class SoftReActJsonSingleInputOutputParser(
    BaseOutputParser[Union[AgentAction, AgentFinish]]
):
    """
    A more tolerant version of ReActJsonSingleInputOutputParser from the
    langchain lib

    Ref: libs/langchain/langchain/agents/output_parsers/react_json_single_input.py
    """

    pattern = re.compile(r"^.*?`{3}(?:json)?\n(.*?)`{3}.*?$", re.DOTALL)
    """Regex pattern to parse the output."""

    @property
    def _type(self) -> str:
        return "soft-react-json-single-input"

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse the output of an LLM call."""

        includes_answer = FINAL_ANSWER_ACTION in text
        try:
            found = self.pattern.search(text)
            if not found:
                # Fast fail to parse Final Answer.
                raise ValueError("action not found")
            action = found.group(1)
            response = json.loads(action.strip())
            includes_action = "action" in response
            if includes_answer and includes_action:
                raise OutputParserException(
                    "Parsing LLM output produced a final answer "
                    f"and a parse-able action: {text}"
                )
            return AgentAction(
                response["action"], response.get("action_input", {}), text
            )

        except Exception:
            if not includes_answer:
                # no explicit answer key, just return text verbatim
                return AgentFinish({"output": text}, text)

            # return text after final answer action marker
            output = text.split(FINAL_ANSWER_ACTION)[-1].strip()
            return AgentFinish({"output": output}, text)
