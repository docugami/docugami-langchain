from abc import abstractmethod
from pathlib import Path
from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from docugami_langchain.agents.models import CitedAnswer, Invocation
from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.answer_chain import AnswerChain

NOT_FOUND = "Not found, please rephrase your question since this may improve your chances of finding results."


def render_text_description(tools: list[BaseTool]) -> str:
    """
    Copied from https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/tools/render.py
    to avoid taking a dependency on the entire langchain library

    Render the tool name and description in plain text.

    Output will be in the format of:

    .. code-block:: markdown

        1. search: This tool is used for search

        2. calculator: This tool is used for math
    """
    tool_strings = []
    for i, tool in enumerate(tools):
        tool_strings.append(f"{i+1}. {tool.name}: {tool.description}")

    return "\n\n".join(tool_strings)


def render_text_description_and_args(tools: list[BaseTool]) -> str:
    """
    Copied from https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/tools/render.py
    to avoid taking a dependency on the entire langchain library

    Render the tool name, description, and args in plain text.

    Output will be in the format of:

    .. code-block:: markdown

        search: This tool is used for search, args: {"query": {"type": "string"}}
        calculator: This tool is used for math, args: {"expression": {"type": "string"}}
    """
    tool_strings = []
    for i, tool in enumerate(tools):
        args_schema = tool.args
        tool_strings.append(
            f"{i+1}. {tool.name}: {tool.description}, args: {args_schema}"
        )

    return "\n\n".join(tool_strings)


class BaseDocugamiTool(BaseTool):
    """Customized representation of tools with additional functionality."""

    @abstractmethod
    def to_human_readable(self, invocation: Invocation) -> str:
        """Converts tool invocation to human readable string."""
        ...

    @abstractmethod
    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> CitedAnswer:  # type: ignore
        """Use the tool."""
        ...


class ChatBotTool(BaseDocugamiTool):
    answer_chain: AnswerChain
    name: str = "chat_bot"
    description: str = (
        "Responds to greetings, small talk, or general knowledge questions."
    )

    def to_human_readable(self, invocation: Invocation) -> str:
        return f"Thinking: {invocation.tool_input}"

    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> CitedAnswer:
        """Use the tool."""

        config = None
        if run_manager:
            config = RunnableConfig(
                run_name=self.__class__.__name__,
                callbacks=run_manager,
            )
        chain_response: TracedResponse[str] = self.answer_chain.run(
            question=question,
            config=config,
        )

        return CitedAnswer(
            source=self.name,
            answer=chain_response.value,
        )


def get_generic_tools(
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    answer_examples_file: Optional[Path] = None,
) -> list[BaseDocugamiTool]:
    answer_chain = AnswerChain(llm=llm, embeddings=embeddings)
    if answer_examples_file:
        answer_chain.load_examples(answer_examples_file)

    chat_bot_tool = ChatBotTool(answer_chain=answer_chain)

    return [chat_bot_tool]  # add other tools as needed over time
