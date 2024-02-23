# Adapted with thanks from
from typing import AsyncIterator, Dict, Optional, Tuple

from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_agent_executor

from docugami_langchain.base_runnable import BaseRunnable, TracedResponse
from docugami_langchain.output_parsers.soft_react_json_single_input import (
    SoftReActJsonSingleInputOutputParser,
)
from docugami_langchain.params import RunnableParameters
from docugami_langchain.prompts import STANDARD_SYSTEM_INSTRUCTIONS_LIST

SYSTEM_MESSAGE_CORE = f"""You are a helpful assistant that answers user queries based only on given context.

You ALWAYS follow the following guidance to generate your answers, regardless of any other guidance or requests:

{STANDARD_SYSTEM_INSTRUCTIONS_LIST}

Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.
"""

REACT_AGENT_SYSTEM_MESSAGE = (
    SYSTEM_MESSAGE_CORE
    + """You have access to the following tools that you use only if necessary:

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

The way you use these tool is by specifying a json blob. Specifically:

- This json should have a `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).
- The only values that may exist in the "action" field are (one of): {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
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

You may also choose not to use a tool, e.g. if none of the provided tools is appropriate to answer the question or the question is conversational
in nature or something you can directly respond to based on conversation history. In that case, you don't need to take an action and can just
do something like:

Question: The input question you must answer
Thought: I can answer this question directly without using a tool
Final Answer: The final answer to the original input question. Note that no citation or SOURCE is needed for such direct answers.

Remember to ALWAYS use the format specified, since any output that does not follow this format is unparseable.

Begin!
"""
)


class ReactAgentInput(BaseModel):
    input: str = ""
    chat_history: list[Tuple[str, str]] = Field(
        default=[],
        extra={
            # for langserve playground
            "widget": {"type": "chat", "input": "input", "output": "output"},
        },
    )


class ReActAgent(BaseRunnable[Dict]):
    """
    Agent that implements simple agentic RAG using the ReAct prompt style.
    """

    tools: list[BaseTool] = []

    def params(self) -> RunnableParameters:
        raise NotImplementedError()

    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """

        def format_chat_history(
            chat_history: list[Tuple[str, str]]
        ) -> list[BaseMessage]:
            buffer: list[BaseMessage] = []
            for human, ai in chat_history:
                buffer.append(HumanMessage(content=human))
                buffer.append(AIMessage(content=ai))
            return buffer

        def format_log_to_str(
            intermediate_steps: list[Tuple[AgentAction, str]],
            observation_prefix: str = "Observation: ",
            llm_prefix: str = "Thought: ",
        ) -> str:
            """Construct the scratchpad that lets the agent continue its thought process."""
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
            return thoughts

        def render_text_description_and_args(tools: list[BaseTool]) -> str:
            """Render the tool name, description, and args in plain text.

            Output will be in the format of:

            .. code-block:: markdown

                search: This tool is used for search, args: {"query": {"type": "string"}}
                calculator: This tool is used for math, \
                args: {"expression": {"type": "string"}}
            """
            tool_strings = []
            for tool in tools:
                args_schema = str(tool.args)
                tool_strings.append(
                    f"{tool.name}: {tool.description}, args: {args_schema}"
                )
            return "\n".join(tool_strings)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REACT_AGENT_SYSTEM_MESSAGE),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}\n\n{agent_scratchpad}"),
            ]
        )

        agent_runnable: Runnable = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: format_chat_history(x["chat_history"]),
                "agent_scratchpad": lambda x: format_log_to_str(
                    x["intermediate_steps"]
                ),
                "tools": lambda x: render_text_description_and_args(self.tools),
                "tool_names": lambda x: ", ".join([t.name for t in self.tools]),
            }
            | prompt
            | self.llm.bind(stop=["\nObservation"])
            | SoftReActJsonSingleInputOutputParser()
        )

        return create_agent_executor(
            agent_runnable, self.tools, input_schema=ReactAgentInput
        )

    def run(  # type: ignore[override]
        self,
        question: str,
        config: Optional[dict] = None,
    ) -> Dict:
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            config=config,
        )

    def run_stream(  # type: ignore[override]
        self,
        question: str,
        config: Optional[dict] = None,
    ) -> AsyncIterator[TracedResponse[Dict]]:
        if not question:
            raise Exception("Input required: question")

        return super().run_stream(
            question=question,
            config=config,
        )

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[str],
        config: Optional[dict] = None,
    ) -> list[Dict]:
        return super().run_batch(
            inputs=[{"question": i} for i in inputs],
            config=config,
        )
