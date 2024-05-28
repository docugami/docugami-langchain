from typing import AsyncIterator, Optional

from langchain_core.runnables import RunnableConfig

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.output_parsers.truthy import TruthyOutputParser
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class RetrievalGraderChain(BaseDocugamiChain[bool]):
    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "question",
                    "QUESTION",
                    "A question from the user.",
                ),
                RunnableSingleParameter(
                    "context",
                    "CONTEXT",
                    "Retrieved context, which you need to grade",
                ),
            ],
            output=RunnableSingleParameter(
                "is_relevant",
                "IS RELEVANT",
                "A boolean (true/false) value indicating whether the retrieved context is relevant to the question.",
            ),
            task_description="acts as a grader assessing relevance of a retrieved context to a user question",
            additional_instructions=[
                "- The output must be a boolean (true/false) judgment only, with no preamble or other explanation.",
                "- If the retrieved context contains information or keywords related to the user question, grade it as relevant (true)."
                "- It does not need to be a stringent test. The goal is to filter out erroneous retrievals.",
            ],
            stop_sequences=["<|eot_id|>"],
            additional_runnables=[TruthyOutputParser()],
            include_output_instruction_suffix=True,
        )

    def run(  # type: ignore[override]
        self,
        question: str,
        context: str,
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[bool]:
        if not question or not context:
            raise Exception("Inputs required: question, context")

        return super().run(
            question=question,
            context=context,
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        question: str,
        context: str,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[bool]]:
        if not question or not context:
            raise Exception("Inputs required: question, context")

        async for item in super().run_stream(
            question=question,
            context=context,
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, str]],
        config: Optional[RunnableConfig] = None,
    ) -> list[bool]:
        return super().run_batch(
            inputs=[
                {
                    "question": i[0],
                    "context": i[1],
                }
                for i in inputs
            ],
            config=config,
        )
