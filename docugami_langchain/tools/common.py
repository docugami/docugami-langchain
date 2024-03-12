from pathlib import Path
from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from docugami_langchain.base_runnable import CitedAnswer, TracedResponse
from docugami_langchain.chains.answer_chain import AnswerChain


class SmallTalkTool(BaseTool):
    answer_chain: AnswerChain
    name: str = "small_talk"
    description: str = (
        "Use to respond to greetings, small talk, or question you can directly answer based on the chat history."
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

        return CitedAnswer(
            source=self.name,
            answer=chain_response.value,
            citations=[],
            metadata={},
        )


class GeneralKnowlegeTool(BaseTool):
    answer_chain: AnswerChain
    name: str = "general_knowledge"
    description: str = "Use to answer general knowledge questions"

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

        return CitedAnswer(
            source=self.name,
            answer=chain_response.value,
            citations=[],
            metadata={},
        )


class HumanInterventionTool(BaseTool):
    name: str = "human_intervention"
    description: str = (
        """Use you think the question can likely be answered from the document set described by the retrieval_* tool, however there is no given """
        """query_* tool that has the requisite information in its table schema to answer the question. This tool will request the user to update the """
        """query_* tool or create a new one that can answer questions like this one."""
    )

    def _run(
        self,
        question: str,
        chat_history: tuple[str, str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> CitedAnswer:  # type: ignore
        """Use the tool."""

        return CitedAnswer(
            source=self.name,
            answer="""Sorry, I don't have enough information to answer this question. Please try rephrasing the question, or please """
            """create or update reports against the relevant docset that maybe queried to answer questions like this one""",
            citations=[],
            metadata={},
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
