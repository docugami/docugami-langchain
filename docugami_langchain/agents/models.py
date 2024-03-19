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


class StepState(BaseModel):
    output: str
    invocation: Optional[Invocation] = None


class AgentState(TypedDict):
    # **** Inputs
    # The list of previous messages in the conversation
    chat_history: list[tuple[str, str]]

    # The input question
    question: str

    # **** Internal State

    # The next tool invocation that must be made
    # Needs `None` as a valid type, since this is what this will start as
    tool_invocation: Union[Invocation, None]

    # List of steps taken so far (this state is added to, not overwritten)
    intermediate_steps: Annotated[list[StepState], operator.add]

    # **** Output
    cited_answer: CitedAnswer
