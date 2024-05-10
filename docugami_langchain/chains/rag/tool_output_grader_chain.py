from typing import AsyncIterator, Optional, Sequence

from langchain_core.runnables import RunnableConfig

from docugami_langchain.agents.models import StepState
from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.history import steps_to_str
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class ToolOutputGraderChain(BaseDocugamiChain[str]):
    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
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
                    "The inputs and outputs to various intermediate steps an AI agent has previously taken to try and answer the question using specialized tools. ",
                ),
            ],
            output=RunnableSingleParameter(
                "is_answered",
                "IS ANSWERED",
                "A boolean (true/false) value indicating whether or not the question is answered completely by the sequence of intermediate steps",
            ),
            task_description="determines whether a question has been answered or not by an AI agent, considering intermediate output from specialized tools that know how to answer questions",
            additional_instructions=[
                "- The output must be a boolean (true/false) judgment against only, with no preamble or other explanation.",
                "- If you think the intermediate steps adequately and completely answer the question, output true otherwise output false. "
                "- Your output will be used by the AI agent to determine if it is ready to generate the final answer from the intermediate steps, "
                + "or if should try more intermediate steps with different tools and/or inputs.",
            ],
            stop_sequences=["<|eot_id|>"],
        )

    def run(  # type: ignore[override]
        self,
        question: str,
        tool_descriptions: str = "",
        intermediate_steps: Sequence[StepState] = [],
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[str]:
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            tool_descriptions=tool_descriptions,
            intermediate_steps=steps_to_str(intermediate_steps),
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        question: str,
        tool_descriptions: str = "",
        intermediate_steps: Sequence[StepState] = [],
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[str]]:
        if not question:
            raise Exception("Input required: question")

        async for item in super().run_stream(
            question=question,
            tool_descriptions=tool_descriptions,
            intermediate_steps=steps_to_str(intermediate_steps),
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, str, Sequence[StepState]]],
        config: Optional[RunnableConfig] = None,
    ) -> list[str]:
        return super().run_batch(
            inputs=[
                {
                    "question": i[0],
                    "tool_descriptions": i[1],
                    "intermediate_steps": steps_to_str(i[2]),
                }
                for i in inputs
            ],
            config=config,
        )
