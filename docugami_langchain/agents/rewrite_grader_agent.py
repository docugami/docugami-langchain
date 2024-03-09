# Adapted with thanks from https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb

import operator
from typing import Annotated, AsyncIterator, Optional, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from docugami_langchain.agents.base import BaseDocugamiAgent
from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.config import DEFAULT_EXAMPLES_PER_PROMPT
from docugami_langchain.params import RunnableParameters

from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor

class RewriteGraderRAGState(TypedDict):
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


class RewriteGraderRAGAgent(BaseDocugamiAgent[dict]):
    """
    Agent that implements agentic RAG with the following additional optimizations:
    1. Query Rewriting
    2. Retrieval Grading
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
        
        # Define a new graph
        workflow = StateGraph(RewriteGraderRAGState)

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
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[dict]:
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        question: str,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[dict]]:
        if not question:
            raise Exception("Input required: question")

        async for item in super().run_stream(
            question=question,
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[str],
        config: Optional[RunnableConfig] = None,
    ) -> list[dict]:
        return super().run_batch(
            inputs=[{"question": i} for i in inputs],
            config=config,
        )
