from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, StateGraph

from docugami_langchain.agents.base import THINKING, BaseDocugamiAgent
from docugami_langchain.agents.models import (
    AgentState,
    CitedAnswer,
    Invocation,
    StepState,
)
from docugami_langchain.chains.rag.tool_final_answer_chain import ToolFinalAnswerChain
from docugami_langchain.history import chat_history_to_str, steps_to_str
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter
from docugami_langchain.tools.common import render_text_description


class ToolRouterAgent(BaseDocugamiAgent):
    """
    Agent that implements agentic RAG with a tool router implementation.
    """

    final_answer_chain: ToolFinalAnswerChain

    def params(self) -> RunnableParameters:
        """The params are directly implemented in the runnable."""
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "chat_history",
                    "CHAT HISTORY",
                    "Previous chat messages that may provide additional context for this question.",
                ),
                RunnableSingleParameter(
                    "question",
                    "QUESTION",
                    "Question asked by the user, which must be answered from one of the given tools.",
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
                RunnableSingleParameter(
                    "intermediate_steps",
                    "INTERMEDIATE STEPS",
                    "The inputs and outputs to various intermediate steps an AI agent has previously taken to consider the question using specialized tools. "
                    + "Carefully consider these intermediate steps as you decide on your next tool invocation, think step by step!",
                ),
            ],
            output=RunnableSingleParameter(
                "tool_invocation_json",
                "TOOL INVOCATION JSON",
                "A JSON blob with the name of the tool to use (`tool_name`) and the input to send it per the tool description (`tool_input`)",
            ),
            task_description="selects an appropriate tool for the question a user is asking, and builds a tool invocation JSON blob for the tool",
            additional_instructions=[
                """- Here is an example of a valid JSON blob for your output. Please STRICTLY follow this format:
{{
  "tool_name": $TOOL_NAME,
  "tool_input": $INPUT_STRING
}}""",
                "- Always use one of the tools, don't try to directly answer the question even if you think you know the answer",
                "- $TOOL_NAME is the name of the tool to use, and must be one of these values: {tool_names}",
                "- $INPUT_STRING is the (string) input carefully crafted to answer the question using the given tool.",
                "- Before retrying a tool, look at previous attempts at running the tool (in intermediate steps) and try to update the inputs to the tool before trying again",
            ],
            stop_sequences=[],
            additional_runnables=[PydanticOutputParser(pydantic_object=Invocation)],
        )

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

        tool_invocation_runnable: Runnable = {
            "question": lambda x: x["question"],
            "chat_history": lambda x: chat_history_to_str(x["chat_history"]),
            "tool_names": lambda x: x["tool_names"],
            "tool_descriptions": lambda x: x["tool_descriptions"],
            "intermediate_steps": lambda x: steps_to_str(x["intermediate_steps"]),
        } | super().runnable()

        def generate_tool_invocation(
            state: AgentState, config: Optional[RunnableConfig]
        ) -> AgentState:
            invocation: Invocation = tool_invocation_runnable.invoke(state, config)
            answer_source = ToolRouterAgent.__name__

            # This agent always decides to invoke a tool
            tool_name = invocation.tool_name
            tool_input = invocation.tool_input
            if tool_name and tool_input:
                busy_text = THINKING
                if tool_name.startswith("retrieval"):
                    busy_text = f"Searching documents for '{tool_input}'"
                elif tool_name.startswith("query"):
                    busy_text = f"Querying report for '{tool_input}'"

            return {
                "tool_invocation": invocation,
                "cited_answer": CitedAnswer(
                    source=answer_source,
                    answer=busy_text,  # Show the user interim output.
                ),
            }

        def generate_final_answer(
            state: AgentState, config: Optional[RunnableConfig]
        ) -> AgentState:
            chain_response = self.final_answer_chain.run(
                question=state.get("question") or "",
                chat_history=state.get("chat_history") or [],
                tool_descriptions=state.get("tool_descriptions") or "",
                intermediate_steps=state.get("intermediate_steps") or [],
                config=config,
            )

            final_answer_candidate = chain_response.value

            return {
                "cited_answer": final_answer_candidate,
                "intermediate_steps": [
                    StepState(
                        invocation=state.get("tool_invocation"),
                        output=str(
                            final_answer_candidate.answer,
                        ),
                    )
                ],
            }

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
        workflow.add_node("generate_tool_invocation", generate_tool_invocation)  # type: ignore
        workflow.add_node("execute_tool", self.execute_tool)  # type: ignore
        workflow.add_node("generate_final_answer", generate_final_answer)  # type: ignore

        # Set the entrypoint
        workflow.set_entry_point("run_agent")

        # Add edges
        workflow.add_edge("run_agent", "generate_tool_invocation")
        workflow.add_edge("generate_tool_invocation", "execute_tool")
        workflow.add_edge("execute_tool", "generate_final_answer")

        # Decide whether to end iteration if agent determines final answer is achieved
        # otherwise keep iterating
        workflow.add_conditional_edges(
            "generate_final_answer",
            should_continue,
            {
                "continue": "generate_tool_invocation",
                "end": END,
            },
        )

        # Compile
        return workflow.compile()

    def parse_final_answer(self, text: str) -> str:
        return text  # no special delimiter in final answer
