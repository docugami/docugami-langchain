# Adapted with thanks from https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/base.ipynb
from __future__ import annotations

from typing import Optional, Sequence

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

- This json should have an `tool_name` key (with the name of the tool to use) and a `tool_input` key (with the string input to the tool).
- The only values that may exist in the "tool_name" field are (one of): {tool_names}

Here is an example of a valid $JSON_BLOB:

```
{{
  "tool_name": $TOOL_NAME,
  "tool_input": $INPUT_STRING
}}
```

ALWAYS use the following format:

Question: The question you must answer
Thought: You should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: The final answer to the original input question. Make sure this is a complete answer, since only text after this label will be shown to the user.

Don't give up easily. If you cannot find an answer using a tool, try using a different tool or the same tool with different inputs.

If you think you need clarifying information to answer the question, just ask the user. The user will see your final answer, and their reply will be sent back to you in the 
form of another question, and you can combine that with chat history to better answer the question.

If you think the user is not getting the answer they need, suggest that they rephrase the question or ask them to build reports against the docset mentioned in your available tools,
since you will be able to query those reports to answer questions better. Do this as a final answer, not an action, since the user sees only your final answers.

Never mention tools (directly by name, or the fact that you have access to tools, or the topic of tools in general) in your response. Tools are an internal implementation detail,
and the user only knows about document sets as well as reports built against document sets.

Make extra sure that the "Final Answer" prefix marks the output you want to show to the user.

Begin! Remember to ALWAYS use the format specified, since output that does not follow the EXACT format above is unparsable.
"""
)


def steps_to_react_str(
    intermediate_steps: Sequence[StepState],
    observation_prefix: str = "Observation: ",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = ""
    if intermediate_steps:
        for step in intermediate_steps:
            if step.invocation:
                thoughts += step.invocation.log

            thoughts += f"\n{observation_prefix}{step.output}\n"
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

        def run_agent(
            state: AgentState, config: Optional[RunnableConfig]
        ) -> AgentState:
            return {
                "tool_names": ", ".join([t.name for t in self.tools]),
                "tool_descriptions": "\n" + render_text_description(self.tools),
            }

        agent_runnable: Runnable = (
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: chat_history_to_str(
                    x["chat_history"], include_human_marker=True
                ),
                "tool_names": lambda x: ", ".join([t.name for t in self.tools]),
                "tool_descriptions": lambda x: render_text_description(self.tools),
                "intermediate_steps": lambda x: steps_to_react_str(
                    x["intermediate_steps"]
                ),
            }
            | ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        REACT_AGENT_SYSTEM_MESSAGE,
                    ),
                    (
                        "human",
                        "{chat_history}Question: {question}\n\n{intermediate_steps}",
                    ),
                ]
            )
            | self.llm.bind(stop=["\nObservation"])
            | CustomReActJsonSingleInputOutputParser()
        )

        def generate_re_act(
            state: AgentState, config: Optional[RunnableConfig]
        ) -> AgentState:
            react_output = agent_runnable.invoke(state, config)

            answer_source = ReActAgent.__name__
            if isinstance(react_output, Invocation):
                # Agent wants to invoke a tool
                tool_name = react_output.tool_name
                tool_input = react_output.tool_input
                busy_text = THINKING
                if tool_name and tool_input:
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

        # Define the nodes of the graph
        workflow.add_node("run_agent", run_agent)  # type: ignore
        workflow.add_node("generate_re_act", generate_re_act)  # type: ignore
        workflow.add_node("execute_tool", self.execute_tool)  # type: ignore

        # Set the entrypoint node
        workflow.set_entry_point("run_agent")

        # Add edges
        workflow.add_edge("run_agent", "generate_re_act")
        workflow.add_edge("execute_tool", "generate_re_act")  # loop back

        # Decide whether to end iteration if agent determines final answer is achieved
        # otherwise keep iterating
        workflow.add_conditional_edges(
            "generate_re_act",
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
