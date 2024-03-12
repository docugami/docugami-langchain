# Adapted with thanks from https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/base.ipynb
from __future__ import annotations

import operator
from typing import (
    Annotated,
    AsyncIterator,
    Optional,
    TypedDict,
    Union,
)

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor

from docugami_langchain.agents.base import (
    BaseDocugamiAgent,
    chat_history_to_messages,
    format_log_to_str,
    render_text_description,
)
from docugami_langchain.base_runnable import TracedResponse, standard_sytem_instructions
from docugami_langchain.config import DEFAULT_EXAMPLES_PER_PROMPT
from docugami_langchain.output_parsers.soft_react_json_single_input import (
    SoftReActJsonSingleInputOutputParser,
)
from docugami_langchain.params import RunnableParameters

REACT_AGENT_SYSTEM_MESSAGE = (
    standard_sytem_instructions("answers user queries based only on given context")
    + """
You have access to the following tools that you use only if necessary:

{tools}

The way you use these tools is by specifying a json blob. Specifically:

- This json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the string input to the tool going here).
- The only values that may exist in the "action" field are (one of): {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT_STRING
}}
```

ALWAYS use the following format:

Question: The input question you must answer
Thought: You should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, with citation describing which tool you used and how. See tool description above for how to cite each type of tool.

Begin! Remember to ALWAYS use the format specified, especially the $JSON_BLOB (enclosed by ```) and marking your Final Answer. Any output that does not follow this format is unparseable.
"""
)


class ReActState(TypedDict):
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


class ReActAgent(BaseDocugamiAgent[ReActState]):
    """
    Agent that implements simple agentic RAG using the ReAct prompt style.
    """

    tools: list[BaseTool] = []

    @staticmethod
    def to_human_readable(state: ReActState) -> str:
        outcome = state.get("agent_outcome", None)
        if outcome:
            if isinstance(outcome, AgentAction):
                tool_name = outcome.tool
                tool_input = outcome.tool_input
                if tool_name.startswith("search"):
                    return f"Searching documents for '{tool_input}'"
                elif tool_name.startswith("query"):
                    return f"Querying report for '{tool_input}'"
            elif isinstance(outcome, AgentFinish):
                return_values = outcome.return_values
                if return_values:
                    answer = return_values.get("output")
                    if answer:
                        return answer

        return "Thinking..."

    def create_finish_state(self, content: str) -> ReActState:
        return ReActState(
            question="",
            chat_history=[],
            agent_outcome=AgentFinish(return_values={"output": content}, log=""),
            intermediate_steps=[],
        )

    def params(self) -> RunnableParameters:
        """The params are directly implemented in the runnable."""
        raise NotImplementedError()

    def prompt(
        self,
        params: RunnableParameters,
        num_examples: int = DEFAULT_EXAMPLES_PER_PROMPT,
    ) -> BasePromptTemplate:
        """The prompt is directly implemented in the runnable."""
        raise NotImplementedError()

    def runnable(self) -> Runnable:
        """
        Custom runnable for this agent.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    REACT_AGENT_SYSTEM_MESSAGE,
                ),
                ("human", "{chat_history}\n\n{question}\n\n{agent_scratchpad}"),
            ]
        )

        agent_runnable: Runnable = (
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: chat_history_to_messages(x["chat_history"]),
                "agent_scratchpad": lambda x: format_log_to_str(
                    x["intermediate_steps"]
                ),
                "tools": lambda x: render_text_description(self.tools),
                "tool_names": lambda x: ", ".join([t.name for t in self.tools]),
            }
            | prompt
            | self.llm.bind(stop=["\nObservation"])
            | SoftReActJsonSingleInputOutputParser()
        )

        tool_executor = ToolExecutor(self.tools)

        def run_agent(data: ReActState, config: Optional[RunnableConfig]) -> ReActState:
            agent_outcome = agent_runnable.invoke(data, config)
            return {"agent_outcome": agent_outcome}

        def execute_tools(
            data: ReActState, config: Optional[RunnableConfig]
        ) -> ReActState:
            # Get the most recent agent_outcome - this is the key added in the `agent` above
            agent_action = data["agent_outcome"]
            output = tool_executor.invoke(agent_action, config)
            return {"intermediate_steps": [(agent_action, str(output))]}

        def should_continue(data: ReActState) -> str:
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
        workflow = StateGraph(ReActState)

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
    ) -> TracedResponse[ReActState]:
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            chat_history=chat_history,
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[ReActState]]:
        if not question:
            raise Exception("Input required: question")

        async for item in super().run_stream(
            question=question,
            chat_history=chat_history,
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, list[tuple[str, str]]]],
        config: Optional[RunnableConfig] = None,
    ) -> list[ReActState]:
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
