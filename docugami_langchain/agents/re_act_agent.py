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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor

from docugami_langchain.agents.base import BaseDocugamiAgent
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

There are two kinds of tools:

1. Tools with names that start with search_*. Use one of these if you think the answer to the question is likely to come from one or a few documents.
   Use the tool description to decide which tool to use in particular if there are multiple search_* tools. For the final result from these tools, cite your answer
   as follows after your final answer:

        SOURCE: I formulated an answer based on information I found in [document names, found in context]

2. Tools with names that start with query_*. Use one of these if you think the answer to the question is likely to come from a lot of documents or
   requires a calculation (e.g. an average, sum, or ordering values in some way). Make sure you use the tool description to decide whether the particular
   tool given knows how to do the calculation intended, especially if there are multiple query_* tools. For the final result from these tools, cite your answer
   as follows after your final answer:

        SOURCE: [Human readable version of SQL query from the tool's output. Do NOT include the SQL very verbatim, describe it in english for a non-technical user.]

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
Final Answer: the final answer to the original input question, with citation describing which tool you used and how. See notes above for how to cite each type of tool.

You may also choose not to use a tool in the following cases:
1. The question is conversational in nature (e.g. a greeting or a joke) and you can directly reply without using a tool
2. The question pertains to general knowledge (e.g. a well known fact) and you can directly reply without using a tool
3. The question can be directly answered based on the provided conversation history or context without using a tool

Note that you should answer directly only if you know the answer. If you don't know the answer, try one of the given tools to attempt to answer it.

If you choose not to use a tool, you don't need to take an action and can just do something like:

Question: The input question you must answer
Thought: I can answer this question directly without using a tool
Final Answer: The final answer to the original input question. Note that no citation or SOURCE is needed for such direct answers.

Remember to ALWAYS use the format specified, since any output that does not follow this format is unparseable.

Begin!
"""
)


class AgentState(TypedDict):
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


class ReActAgent(BaseDocugamiAgent[AgentState]):
    """
    Agent that implements simple agentic RAG using the ReAct prompt style.
    """

    tools: list[BaseTool] = []

    @staticmethod  # type: ignore
    def to_human_readable(state: AgentState) -> str:
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
        Custom runnable for this chain.
        """

        def format_chat_history(
            chat_history: list[tuple[str, str]]
        ) -> list[BaseMessage]:
            messages: list[BaseMessage] = []

            if chat_history:
                for human, ai in chat_history:
                    messages.append(HumanMessage(content=human))
                    messages.append(AIMessage(content=ai))
            return messages

        def format_log_to_str(
            intermediate_steps: list[tuple[AgentAction, str]],
            observation_prefix: str = "Observation: ",
            llm_prefix: str = "Thought: ",
        ) -> str:
            """Construct the scratchpad that lets the agent continue its thought process."""
            thoughts = ""
            if intermediate_steps:
                for action, observation in intermediate_steps:
                    thoughts += action.log
                    thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
            return thoughts

        def render_text_description(tools: list[BaseTool]) -> str:
            """Render the tool name and description in plain text.

            Output will be in the format of:

            .. code-block:: markdown

                search: This tool is used for search
                calculator: This tool is used for math
            """
            tool_strings = []
            for tool in tools:
                tool_strings.append(f"{tool.name}: {tool.description}")
            return "\n".join(tool_strings)

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
                "chat_history": lambda x: format_chat_history(x["chat_history"]),
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

        def run_agent(data: dict, config: Optional[RunnableConfig]) -> dict:
            agent_outcome = agent_runnable.invoke(data, config)
            return {"agent_outcome": agent_outcome}

        def execute_tools(data: dict, config: Optional[RunnableConfig]) -> dict:
            # Get the most recent agent_outcome - this is the key added in the `agent` above
            agent_action = data["agent_outcome"]
            output = tool_executor.invoke(agent_action, config)
            return {"intermediate_steps": [(agent_action, str(output))]}

        def should_continue(data: dict) -> str:
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
        workflow = StateGraph(AgentState)

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
    ) -> TracedResponse[AgentState]:
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
    ) -> AsyncIterator[TracedResponse[AgentState]]:
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
        inputs: list[str],
        chat_history: list[list[tuple[str, str]]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> list[AgentState]:
        raise NotImplementedError()
