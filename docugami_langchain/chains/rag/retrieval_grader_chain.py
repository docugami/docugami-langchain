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
                    "document_summary",
                    "DOCUMENT SUMMARY",
                    "Summary of a document, from which a chunk was retrieved.",
                ),
                RunnableSingleParameter(
                    "retrieved_chunk",
                    "RETRIEVED CHUNK",
                    "A retrieved chunk, which you need to grade.",
                ),
            ],
            output=RunnableSingleParameter(
                "is_relevant",
                "IS RELEVANT",
                "A boolean (true/false) value indicating whether the retrieved chunk (along with the document summary) is relevant to the question.",
            ),
            task_description="acts as a grader assessing relevance of a retrieved chunk and its associated document summary to a user question",
            additional_instructions=[
                "- The output must be a boolean (true/false) judgment only, with no preamble or other explanation.",
                "- If the chunk or document summary contain information or keywords related to the user question, grade it as relevant (true)."
                "- It does not need to be a stringent test. The goal is to filter out erroneous retrievals.",
            ],
            stop_sequences=["<|eot_id|>"],
            additional_runnables=[TruthyOutputParser()],
            include_output_instruction_suffix=True,
        )

    def run(  # type: ignore[override]
        self,
        question: str,
        document_summary: str,
        retrieved_chunk: str,
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[bool]:
        if not question or not document_summary or not retrieved_chunk:
            raise Exception(
                "Inputs required: question, document_summary, retrieved_chunk"
            )

        return super().run(
            question=question,
            document_summary=document_summary,
            retrieved_chunk=retrieved_chunk,
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        question: str,
        document_summary: str,
        retrieved_chunk: str,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[bool]]:
        if not question or not document_summary or not retrieved_chunk:
            raise Exception(
                "Inputs required: question, document_summary, retrieved_chunk"
            )

        async for item in super().run_stream(
            question=question,
            document_summary=document_summary,
            retrieved_chunk=retrieved_chunk,
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, str, str]],
        config: Optional[RunnableConfig] = None,
    ) -> list[bool]:
        return super().run_batch(
            inputs=[
                {
                    "question": i[0],
                    "document_summary": i[1],
                    "retrieved_chunk": i[2],
                }
                for i in inputs
            ],
            config=config,
        )
