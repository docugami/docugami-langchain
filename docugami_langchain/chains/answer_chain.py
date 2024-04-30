from typing import AsyncIterator, Optional

from langchain_core.runnables import RunnableConfig

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class AnswerChain(BaseDocugamiChain[str]):
    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "question",
                    "QUESTION",
                    "A question from the user.",
                ),
            ],
            output=RunnableSingleParameter(
                "answer",
                "ANSWER",
                "A helpful answer, aligned with the rules outlined above.",
            ),
            task_description="answers general questions",
            additional_instructions=["- Shorter answers are better."],
            stop_sequences=["CHAT HISTORY:", "QUESTION:", "<|eot_id|>"],
            key_finding_output_parse=False,  # set to False for streaming
        )

    def run(  # type: ignore[override]
        self,
        question: str,
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[str]:
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
    ) -> AsyncIterator[TracedResponse[str]]:
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
    ) -> list[str]:
        return super().run_batch(
            inputs=[
                {
                    "question": i,
                }
                for i in inputs
            ],
            config=config,
        )
