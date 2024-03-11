import operator
from typing import Annotated, AsyncIterator, Optional, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor

from docugami_langchain.agents.base import (
    BaseDocugamiAgent,
    chat_history_to_str,
    render_text_description,
)
from docugami_langchain.base_runnable import CitedAnswer, TracedResponse
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class ToolRouterState(TypedDict):
    # The input question
    question: str
    # The list of previous messages in the conversation
    chat_history: list[BaseMessage]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


class ToolRouterAgent(BaseDocugamiAgent[CitedAnswer]):
    """
    Agent that implements agentic RAG with a tool router implementation.
    """

    tools: list[BaseTool] = []

    def params(self) -> RunnableParameters:
        """The params are directly implemented in the runnable."""
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "question",
                    "QUESTION",
                    "Question asked by the user, which must be answered from one of the given tools.",
                ),
                RunnableSingleParameter(
                    "chat_history",
                    "CHAT HISTORY",
                    "Previous chat messages that may provide additional context for this question.",
                ),
                RunnableSingleParameter(
                    "tool_names",
                    "TOOL NAMES",
                    "List (names) of tools that you must exclusively pick one from, in order to answer the given question.",
                ),
                RunnableSingleParameter(
                    "tool_descriptions",
                    "TOOL DESCRIPTIONS",
                    "Detailed description of tools that you must exclusively pick one from, in order to answer the given question.",
                ),
            ],
            output=RunnableSingleParameter(
                "tool_invocation_json",
                "TOOL INVOCATION JSON",
                "A JSON blob with the name of the tool to use (`tool_name`) and the input to send it per the tool description (`tool_input`)",
            ),
            task_description="selects an appropriate tool for the question a user is asking, and builds a tool invocation JSON blob for the tool",
            additional_instructions=[
                "- Your output must be a valid JSON blob, with a `tool_name` key (with the name of the tool to use) and a `tool_input` key (with the string input to the tool going here).",
                "- You must pick one of these values for the `tool_name` key: {tool_names}",
            ],
            additional_runnables=[JsonOutputParser(pydantic_object=AgentAction)],
        )

    def runnable(self) -> Runnable:
        """
        Custom runnable for this agent.
        """

        tool_executor = ToolExecutor(self.tools)

        def run_agent(
            data: ToolRouterState, config: Optional[RunnableConfig]
        ) -> ToolRouterState:
            return {
                "agent_outcome": super().runnable().invoke(data, config),
            }

        def execute_tools(
            data: ToolRouterState, config: Optional[RunnableConfig]
        ) -> ToolRouterState:
            # Get the most recent agent_outcome - this is the key added in the `agent` above
            agent_action = data["agent_outcome"]
            output = tool_executor.invoke(agent_action, config)
            return {"intermediate_steps": [(agent_action, str(output))]}

        def should_continue(data: ToolRouterState) -> str:
            # If the agent outcome is an AgentFinish, then we return `exit` string
            # This will be used when setting up the graph to define the flow
            if isinstance(data["agent_outcome"], AgentFinish):
                return "end"
            # Otherwise, an AgentAction is returned
            # Here we return `continue` string
            # This will be used when setting up the graph to define the flow
            else:
                return "continue"

        # Define a new graph
        workflow = StateGraph(ToolRouterAgent)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", run_agent)
        workflow.add_node("action", execute_tools)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        workflow.set_entry_point("agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
            {
                # If `tools`, then we call the tool node.
                "continue": "action",
                # Otherwise we finish.
                "end": END,
            },
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        workflow.add_edge("action", "agent")

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        return workflow.compile()

    def run(  # type: ignore[override]
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[CitedAnswer]:
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            chat_history=chat_history_to_str(chat_history),
            tool_names=", ".join([tool.name for tool in self.tools]),
            tool_descriptions=render_text_description(self.tools),
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[CitedAnswer]]:
        if not question:
            raise Exception("Input required: question")

        async for item in super().run_stream(
            question=question,
            chat_history=chat_history_to_str(chat_history),
            tool_names=", ".join([tool.name for tool in self.tools]),
            tool_descriptions=render_text_description(self.tools),
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, list[tuple[str, str]]]],
        config: Optional[RunnableConfig] = None,
    ) -> list[CitedAnswer]:
        tool_names = (", ".join([tool.name for tool in self.tools]),)
        tool_descriptions = (render_text_description(self.tools),)

        return super().run_batch(
            inputs=[
                {
                    "question": i[0],
                    "chat_history": chat_history_to_str(i[1]),
                    "tool_names": tool_names,
                    "tool_descriptions": tool_descriptions,
                }
                for i in inputs
            ],
            config=config,
        )
