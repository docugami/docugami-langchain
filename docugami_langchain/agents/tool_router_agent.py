from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, StateGraph

from docugami_langchain.agents.base import BaseDocugamiAgent
from docugami_langchain.agents.models import AgentState, Invocation
from docugami_langchain.history import chat_history_to_str
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter
from docugami_langchain.tools.common import render_text_description


class ToolRouterAgent(BaseDocugamiAgent):
    """
    Agent that implements agentic RAG with a tool router implementation.
    """

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
                "- $TOOL_NAME is the name of the tool to use, and must be one of these values: {tool_names}",
                "- $INPUT_STRING is the (string) input carefully crafted to answer the question using the given tool.",
                "- Always use one of the tools, don't try to directly answer the question even if you think you know the answer",
            ],
            stop_sequences=[],
            additional_runnables=[PydanticOutputParser(pydantic_object=Invocation)],
        )

    def runnable(self) -> Runnable:
        """
        Custom runnable for this agent.
        """

        agent_runnable: Runnable = {
            "question": lambda x: x["question"],
            "chat_history": lambda x: chat_history_to_str(x["chat_history"]),
            "tool_names": lambda x: ", ".join([t.name for t in self.tools]),
            "tool_descriptions": lambda x: "\n" + render_text_description(self.tools),
        } | super().runnable()

        def run_agent(
            state: AgentState, config: Optional[RunnableConfig]
        ) -> AgentState:
            invocation = agent_runnable.invoke(state, config)
            return {"tool_invocation": invocation}

        # def should_continue(state: AgentState) -> str:
        #     ... reflect on answer, and decide to continue or not

        #     if answer and answer.is_final:
        #         return "end"
        #     else:
        #         return "continue"

        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes of the graph (no cycles for now)
        workflow.add_node("run_agent", run_agent)  # type: ignore
        workflow.add_node("execute_tool", self.execute_tool)  # type: ignore

        # Set the entrypoint
        workflow.set_entry_point("run_agent")

        # Add edges
        workflow.add_edge("run_agent", "execute_tool")

        # TODO 1: add final answer edge that takes history, question, tool invocation, and result of tool invocation... and builds cited answer
        # Problem is this is probably going to be a different implementation for each tool (rag tool returns docs? query tool returns frame ? etc),
        # so maybe have this function exposed from each tool class? or maybe create custom chains / functions sand wire them up here conditionally based on the state in the graph?

        # TODO 2: add reflection stage... look at examples but basically ask the llm if the answer is good and if not, loop back
        # however make sure on the loop back the agent knows what you tried before and why it didn't work so it can try something else?
        # Use this as an example:

        # TODO 3: for explained citations, how do we do that at the end in a special node? The streaming implementation will need to stream the answer before the citations

        workflow.add_edge("execute_tool", END)

        # Compile
        return workflow.compile()

    def parse_final_answer(self, text: str) -> str:
        return text  # no special delimiter in final answer
