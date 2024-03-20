# Adapted with thanks from https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/base.ipynb
from __future__ import annotations

from typing import Optional

from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, StateGraph

from docugami_langchain.agents.base import THINKING, BaseDocugamiAgent
from docugami_langchain.agents.models import (
    AgentState,
    CitedAnswer,
    Invocation,
    StepState,
)
from docugami_langchain.base_runnable import standard_sytem_instructions
from docugami_langchain.config import DEFAULT_EXAMPLES_PER_PROMPT
from docugami_langchain.history import chat_history_to_str
from docugami_langchain.output_parsers.custom_react_json_single_input import (
    FINAL_ANSWER_ACTION,
    CustomReActJsonSingleInputOutputParser,
)
from docugami_langchain.params import RunnableParameters
from docugami_langchain.tools.common import render_text_description

REACT_AGENT_SYSTEM_MESSAGE = (
    standard_sytem_instructions("answers user queries based only on given context")
    + """
You have access to the following tools that you use only if necessary:

{tool_descriptions}

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
Final Answer: The final answer to the original input question. Make sure a complete answer follows the "Final Answer:" prefix, since any text before this label will not be shown to the user.

Don't give up easily. If you cannot find an answer using a tool, try using a different tool or the same tool with different inputs.

Begin! Remember to ALWAYS use the format specified, especially being mindful of using the Thought/Action/Observation and "Final Answer" prefixes in your output.
Any output that does not follow the EXACT format above is unparsable.
"""
)


def format_steps_to_react_scratchpad(
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


class ReActAgent(BaseDocugamiAgent):
    """
    Agent that implements simple agentic RAG using the ReAct prompt style.
    """

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
                "chat_history": lambda x: chat_history_to_str(x["chat_history"]),
                "agent_scratchpad": lambda x: format_steps_to_react_scratchpad(
                    x["intermediate_steps"]
                ),
                "tool_names": lambda x: ", ".join([t.name for t in self.tools]),
                "tool_descriptions": lambda x: render_text_description(self.tools),
            }
            | prompt
            | self.llm.bind(stop=["\nObservation"])
            | CustomReActJsonSingleInputOutputParser()
        )

        def run_agent(
            state: AgentState, config: Optional[RunnableConfig]
        ) -> AgentState:
            react_output = agent_runnable.invoke(state, config)

            answer_source = ReActAgent.__name__
            if isinstance(react_output, Invocation):
                # Agent wants to invoke a tool
                tool_name = react_output.tool_name
                tool_input = react_output.tool_input
                if tool_name and tool_input:
                    busy_text = THINKING
                    if tool_name.startswith("retrieval"):
                        busy_text = f"Searching documents for '{tool_input}'"
                    elif tool_name.startswith("query"):
                        busy_text = f"Querying report for '{tool_input}'"

                return {
                    "tool_invocation": react_output,
                    "cited_answer": CitedAnswer(
                        source=answer_source,
                        answer=busy_text,  # Show the user interim output.
                    ),
                }
            elif isinstance(react_output, str):
                # Agent thinks it has a final answer.

                # Source final answer from the last invocation, if any.
                tool_invocation = state.get("tool_invocation")
                if tool_invocation and tool_invocation.tool_name:
                    answer_source = tool_invocation.tool_name

                return {
                    "cited_answer": CitedAnswer(
                        source=answer_source,
                        is_final=True,
                        answer=react_output,  # This is the final answer.
                    ),
                }

            raise Exception(f"Unrecognized agent output: {react_output}")

        def should_continue(state: AgentState) -> str:
            # Decide whether to continue, based on the current state
            answer = state.get("cited_answer")
            if answer and answer.is_final:
                return "end"
            else:
                return "continue"

        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the two nodes we will cycle between
        workflow.add_node("run_agent", run_agent)  # type: ignore
        workflow.add_node("execute_tool", self.execute_tool)  # type: ignore

        # Set the entrypoint node
        workflow.set_entry_point("run_agent")

        # Add an edge from tool output back to the agent to look at the tool output
        workflow.add_edge("execute_tool", "run_agent")

        # Decide whether to end iteration if agent determines final answer is achieved
        # otherwise keep iterating
        workflow.add_conditional_edges(
            "run_agent",
            should_continue,
            {
                "continue": "execute_tool",
                "end": END,
            },
        )

        # Compile
        return workflow.compile()

    def parse_final_answer(self, text: str) -> str:
        if FINAL_ANSWER_ACTION in text:
            return str(text).split(FINAL_ANSWER_ACTION)[-1].strip()

        return ""  # not found
