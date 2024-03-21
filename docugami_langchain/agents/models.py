from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict, Union

from langchain_core.pydantic_v1 import BaseModel


class CitationLink(BaseModel):
    label: str
    href: str


class Citation(BaseModel):
    text: str
    links: list[CitationLink]


class CitedAnswer(BaseModel):
    source: str
    answer: str
    citations: list[tuple[str, list[Citation]]] = []
    is_final: bool = False
    metadata: dict = {}


class Invocation(BaseModel):
    tool_name: str
    tool_input: str
    log: str = ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Invocation):
            return NotImplemented

        # Compare tool_name and tool_input for equality
        return (self.tool_name, self.tool_input) == (other.tool_name, other.tool_input)


class StepState(BaseModel):
    output: str
    invocation: Optional[Invocation] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StepState):
            return NotImplemented

        # Compare invocations for equality
        return self.invocation == other.invocation


class AgentState(TypedDict):
    # **** Inputs
    # The list of previous messages in the conversation
    chat_history: list[tuple[str, str]]

    # The input question
    question: str

    # **** Internal State

    # Descriptions of all tools available to the agent
    tool_descriptions: Union[str, None]

    # The next tool invocation that must be made
    tool_invocation: Union[Invocation, None]

    # List of steps taken so far (this state is added to, not overwritten)
    intermediate_steps: Annotated[list[StepState], operator.add]

    # **** Output
    cited_answer: CitedAnswer
