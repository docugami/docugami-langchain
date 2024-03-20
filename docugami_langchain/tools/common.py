from pathlib import Path
from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from docugami_langchain.agents.models import CitedAnswer
from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.answer_chain import AnswerChain


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


class SmallTalkTool(BaseTool):
    answer_chain: AnswerChain
    name: str = "small_talk"
    description: str = (
        "Responds to greetings, small talk, or questions that can be directly answered from the chat history."
    )

    def _run(
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> CitedAnswer:  # type: ignore
        """Use the tool."""

        chain_response: TracedResponse[str] = self.answer_chain.run(
            question=question,
            chat_history=chat_history,
        )

        return CitedAnswer(source=self.name, answer=chain_response.value)


class GeneralKnowlegeTool(BaseTool):
    answer_chain: AnswerChain
    name: str = "general_knowledge"
    description: str = "Answers general knowledge questions"

    def _run(
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> CitedAnswer:  # type: ignore
        """Use the tool."""

        chain_response: TracedResponse[str] = self.answer_chain.run(
            question=question,
            chat_history=chat_history,
        )

        return CitedAnswer(source=self.name, answer=chain_response.value)


class HumanInterventionTool(BaseTool):
    name: str = "human_intervention"
    description: str = (
        """This tool will request the user to create or update a query_* tool with data sufficient to answer questions like this one via SQL queries against a table. """
        """Use this tool if the question IS LIKELY to be answerable with the document set described by the retrieval_* tool, however there is no given """
        """query_* tool that has the requisite information in its table schema to answer the question via SQL query. """
    )

    def _run(
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> CitedAnswer:  # type: ignore
        """Use the tool."""

        return CitedAnswer(
            source=self.name,
            answer="""Sorry, I don't have enough information to answer this question. Please try rephrasing the question, or please """
            """create or update reports against the relevant docset that maybe queried to answer questions like this one""",
        )


def get_generic_tools(
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    answer_examples_file: Optional[Path] = None,
) -> list[BaseTool]:
    answer_chain = AnswerChain(llm=llm, embeddings=embeddings)
    if answer_examples_file:
        answer_chain.load_examples(answer_examples_file)

    small_talk_tool = SmallTalkTool(answer_chain=answer_chain)
    general_knowledge_Tool = GeneralKnowlegeTool(answer_chain=answer_chain)
    human_intervention_tool = HumanInterventionTool()

    return [small_talk_tool, general_knowledge_Tool, human_intervention_tool]
