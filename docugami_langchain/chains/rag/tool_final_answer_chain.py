from typing import AsyncIterator, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig

from docugami_langchain.agents.models import CitedAnswer, StepState
from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.history import chat_history_to_str, steps_to_str
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class ToolFinalAnswerChain(BaseDocugamiChain[CitedAnswer]):
    def params(self) -> RunnableParameters:
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
                    "A question from the user.",
                ),
                RunnableSingleParameter(
                    "tool_descriptions",
                    "TOOL DESCRIPTIONS",
                    "Detailed description of tools that the AI agent must exclusively pick one from, in order to answer the given question.",
                ),
                RunnableSingleParameter(
                    "intermediate_steps",
                    "INTERMEDIATE STEPS",
                    "The inputs and outputs to various intermediate steps an AI agent has previously taken to try and answer the question using specialized tools. "
                    + "Try to compose your final answer from these intermediate steps, or if you cannot then explain why you cannot in your answer.",
                ),
            ],
            output=RunnableSingleParameter(
                "cited_answer_json",
                "CITED ANSWER JSON",
                "A JSON blob with a cited answer to the given question after considering the information in intermediate steps",
            ),
            task_description="generates a final answer to a question, considering the output from an AI agent that has used specialized tools that know how to answer questions",
            additional_instructions=[
                """- Here is an example of a valid JSON blob for your output. Please STRICTLY follow this format:
{{
  "source": $ANSWER_SOURCE,
  "answer": $ANSWER,
  "is_final": $IS_FINAL
}}""",
                "- Always consider the intermediate steps to formulate your answer. Don't try to directly answer the question even if you think you know the answer",
                "- $ANSWER is the (string) final answer to the user's question, after carefully considering the intermediate steps.",
                "- $IS_FINAL is a boolean judment of self-critiquing your own final answer. If you think it adequately answers the user's question, set this to True. "
                + "Otherwise set this to False. Your output will be sent back to the AI agent and it will try again with different tools or inputs.",
            ],
            stop_sequences=[],
            additional_runnables=[PydanticOutputParser(pydantic_object=CitedAnswer)],
        )

    def run(  # type: ignore[override]
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        tool_descriptions: str = "",
        intermediate_steps: list[StepState] = [],
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[CitedAnswer]:
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            chat_history=chat_history_to_str(chat_history),
            tool_descriptions=tool_descriptions,
            intermediate_steps=steps_to_str(intermediate_steps),
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        question: str,
        chat_history: list[tuple[str, str]] = [],
        tool_descriptions: str = "",
        intermediate_steps: list[StepState] = [],
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[CitedAnswer]]:
        if not question:
            raise Exception("Input required: question")

        async for item in super().run_stream(
            question=question,
            chat_history=chat_history_to_str(chat_history),
            tool_descriptions=tool_descriptions,
            intermediate_steps=steps_to_str(intermediate_steps),
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, list[tuple[str, str]], str, list[StepState]]],
        config: Optional[RunnableConfig] = None,
    ) -> list[CitedAnswer]:
        return super().run_batch(
            inputs=[
                {
                    "question": i[0],
                    "chat_history": chat_history_to_str(i[1]),
                    "tool_descriptions": i[2],
                    "intermediate_steps": steps_to_str(i[3]),
                }
                for i in inputs
            ],
            config=config,
        )
