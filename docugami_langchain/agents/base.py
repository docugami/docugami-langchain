import operator
from typing import Annotated, Optional, TypedDict, Union

from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from docugami_langchain.base_runnable import BaseRunnable, T, TracedResponse


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
    invocation: Invocation
    output: str


def format_steps_to_str(
    intermediate_steps: list[StepState],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = ""
    if intermediate_steps:
        for step in intermediate_steps:
            if step.invocation:
                thoughts += step.invocation.log

            thoughts += f"\n{observation_prefix}{step.output}\n{llm_prefix}"
    return thoughts


class AgentState(TypedDict):
    # **** Inputs
    # The list of previous messages in the conversation
    chat_history: list[BaseMessage]

    # The input question
    question: str

    # **** Internal State

    # The next tool invocation that must be made
    # Needs `None` as a valid type, since this is what this will start as
    tool_invocation: Union[Invocation, None]

    # List of steps taken so far (this state is added to, not overwritten)
    intermediate_steps: Annotated[list[StepState], operator.add]

    # **** Output
    current_answer: CitedAnswer


THINKING = "Thinking..."


class BaseDocugamiAgent(BaseRunnable[T]):
    """
    Base class with common functionality for various chains.
    """

    tools: list[BaseTool] = []

    def execute_tool(
        self,
        state: AgentState,
        config: Optional[RunnableConfig],
    ) -> AgentState:
        # Get the most recent tool invocation (added by the agent) and execute it
        inv_model = state.get("tool_invocation")
        if not inv_model:
            raise Exception(f"No tool invocation in model: {state}")

        inv_obj = ToolInvocation(
            tool=inv_model.tool_name,
            tool_input=inv_model.tool_input,
        )

        tool_executor = ToolExecutor(self.tools)
        output = tool_executor.invoke(inv_obj, config)

        step = StepState(
            invocation=inv_model,
            output=str(output),
        )
        return {"intermediate_steps": [step]}  # appended

    def run(  # type: ignore[override]
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[T]:
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            chat_history=chat_history,
            config=config,
        )

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, list[tuple[str, str]]]],
        config: Optional[RunnableConfig] = None,
    ) -> list[T]:
        return super().run_batch(
            inputs=[
                {
                    "question": i[0],
                    "chat_history": i[1],
                }
                for i in inputs
            ],
            config=config,
        )
