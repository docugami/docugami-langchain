from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, StateGraph

from docugami_langchain.agents.base import BaseDocugamiAgent
from docugami_langchain.agents.models import (
    AgentState,
    CitedAnswer,
    Invocation,
)
from docugami_langchain.chains.rag import (
    StandaloneQuestionChain,
    ToolFinalAnswerChain,
    ToolOutputGraderChain,
)
from docugami_langchain.history import steps_to_str
from docugami_langchain.output_parsers import TextCleaningOutputParser
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter
from docugami_langchain.tools.common import render_text_description


class ToolRouterAgent(BaseDocugamiAgent):
    """
    Agent that implements agentic RAG with a tool router implementation.
    """

    standalone_question_chain: StandaloneQuestionChain
    output_grader_chain: ToolOutputGraderChain
    final_answer_chain: ToolFinalAnswerChain

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
                "- $TOOL_NAME is the (string) name of the tool to use, and must be one of these values: {tool_names}",
                "- $INPUT_STRING is the (string) input carefully crafted to answer the question using the given tool.",
                "- Before retrying a tool, look at previous attempts at running the tool (in intermediate steps) and try to update the inputs to the tool before trying again",
            ],
            stop_sequences=["<|eot_id|>"],
            additional_runnables=[TextCleaningOutputParser(), PydanticOutputParser(pydantic_object=Invocation)],  # type: ignore
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
                "intermediate_steps": [],
            }

        tool_invocation_runnable: Runnable = {
            "question": lambda x: x["question"],
            "tool_names": lambda x: x["tool_names"],
            "tool_descriptions": lambda x: x["tool_descriptions"],
            "intermediate_steps": lambda x: steps_to_str(x["intermediate_steps"]),
        } | super().runnable()

        def standalone_question(
            state: AgentState, config: Optional[RunnableConfig]
        ) -> AgentState:
            question = state.get("question")
            chat_history = state.get("chat_history")

            if question and chat_history:
                standalone_question_response = self.standalone_question_chain.run(
                    question, chat_history, config
                )

                if standalone_question_response.value:
                    state["question"] = standalone_question_response.value

            return state

        def generate_tool_invocation(
            state: AgentState, config: Optional[RunnableConfig]
        ) -> AgentState:
            invocation: Invocation = tool_invocation_runnable.invoke(state, config)
            answer_source = ToolRouterAgent.__name__

            return self.invocation_answer(invocation, answer_source)

        def grade_output(state: AgentState, config: Optional[RunnableConfig]) -> str:
            question = state.get("question")
            tool_descriptions = state.get("tool_descriptions")
            intermediate_steps = state.get("intermediate_steps")

            if question and tool_descriptions:
                if not intermediate_steps:
                    intermediate_steps = []

                grader_output_response = self.output_grader_chain.run(
                    question, tool_descriptions, intermediate_steps, config
                )

                grader_output_result = grader_output_response.value.lower()
                if "true" in grader_output_result or "yes" in grader_output_result:
                    return "true"

            return "false"

        def generate_final_answer(
            state: AgentState, config: Optional[RunnableConfig]
        ) -> AgentState:
            question = state.get("question")
            tool_descriptions = state.get("tool_descriptions")
            intermediate_steps = state.get("intermediate_steps")

            if not question or not tool_descriptions or not intermediate_steps:
                raise Exception("incomplete inputs")

            final_answer_response = self.final_answer_chain.run(
                question, tool_descriptions, intermediate_steps, config
            )

            # Source the answer from the last step, if any
            answer_source = ToolRouterAgent.__name__
            citations = []
            if intermediate_steps:
                last_step = intermediate_steps[-1]
                answer_source = last_step.invocation.tool_name
                citations = last_step.citations
            return {
                "cited_answer": CitedAnswer(
                    source=answer_source,
                    is_final=True,
                    citations=citations,
                    answer=final_answer_response.value,
                ),
            }

        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes of the graph
        workflow.add_node("run_agent", run_agent)  # type: ignore
        workflow.add_node("standalone_question", standalone_question)  # type: ignore
        workflow.add_node("generate_tool_invocation", generate_tool_invocation)  # type: ignore
        workflow.add_node("execute_tool", self.execute_tool)  # type: ignore
        workflow.add_node("generate_final_answer", generate_final_answer)  # type: ignore

        # Set the entrypoint
        workflow.set_entry_point("run_agent")

        # Add edges
        workflow.add_edge("run_agent", "standalone_question")
        workflow.add_edge("standalone_question", "generate_tool_invocation")
        workflow.add_edge("generate_tool_invocation", "execute_tool")
        workflow.add_edge("generate_final_answer", END)

        # Decide whether to keep iterating or generate final answer
        workflow.add_conditional_edges(
            "execute_tool",
            grade_output,
            {
                "true": "generate_final_answer",
                "false": "generate_tool_invocation",  # try again
            },
        )

        # Compile
        return workflow.compile()

    def streamable_node_names(self) -> list[str]:
        """Node names in the graph from which token by token output should be streamed."""
        return ["generate_final_answer"]

    def parse_final_answer_from_streamed_output(self, text: str) -> str:
        """Given output stream from a streamable node, parses out the final answer (e.g. past a delimiter)."""
        return text  # no special delimiter in final answer
