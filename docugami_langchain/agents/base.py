from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from docugami_langchain.agents.models import AgentState, StepState
from docugami_langchain.base_runnable import BaseRunnable, T, TracedResponse

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
