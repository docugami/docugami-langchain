from typing import AsyncIterator, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tracers.context import collect_runs
from langgraph.graph import END, StateGraph

from docugami_langchain.agents.base import BaseDocugamiAgent
from docugami_langchain.agents.models import AgentState
from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.history import chat_history_to_str
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter
from docugami_langchain.tools.common import render_text_description


class ToolRouterAgent(BaseDocugamiAgent[AgentState]):
    """
    Agent that implements agentic RAG with a tool router implementation.
    """

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
                "- Your output must be a valid JSON blob, with a `tool_name` key (with the name of the tool to use) and a `tool_input` key (with the string input to the tool).",
                "- You must pick one of these values for the `tool_name` key: {tool_names}",
            ],
            stop_sequences=[],
            additional_runnables=[JsonOutputParser()],
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

    async def run_stream(  # type: ignore[override]
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[AgentState]]:
        if not question:
            raise Exception("Input required: question")

        config, kwargs_dict = self._prepare_run_args(
            {
                "question": question,
                "chat_history": chat_history,
            }
        )

        with collect_runs() as cb:
            last_response_value = None
            async for output in self.runnable().astream(
                input=kwargs_dict,
                config=config,
            ):
                # stream() yields dictionaries with output keyed by node name
                for key, value in output.items():

                    if not isinstance(value, dict):
                        # agent step-wise streaming yields dictionaries keyed by node name
                        # Ref: https://python.langchain.com/docs/langgraph#streaming-node-output
                        raise Exception(
                            "Expected dictionary output from agent streaming"
                        )

                    last_response_value = value
                    yield TracedResponse[AgentState](value=last_response_value)  # type: ignore

            # yield the final result with the run_id
            if cb.traced_runs:
                run_id = str(cb.traced_runs[0].id)
                yield TracedResponse[AgentState](
                    run_id=run_id,
                    value=last_response_value,  # type: ignore
                )
