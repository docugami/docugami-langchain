from typing import AsyncIterator, Optional

from langchain_docugami.chains.base import BaseDocugamiChain, TracedChainResponse
from langchain_docugami.chains.params import ChainParameters, ChainSingleParameter


class AnswerChain(BaseDocugamiChain[str]):
    def chain_params(self) -> ChainParameters:
        return ChainParameters(
            inputs=[
                ChainSingleParameter(
                    "question", "QUESTION", "A question from the user."
                )
            ],
            output=ChainSingleParameter(
                "answer",
                "ANSWER",
                "A helpful answer, aligned with the rules outlined above",
            ),
            task_description="answers general questions",
            additional_instructions=["- Shorter answers are better."],
            stop_sequences=["\n"],
            key_finding_output_parse=False,  # set to False for streaming
        )

    def run(self, question: str, config: Optional[dict] = None) -> str:  # type: ignore[override]
        if not question:
            raise Exception("Input required: question")

        return super().run(
            question=question,
            config=config,
        )

    def run_stream(  # type: ignore[override]
        self, question: str, config: Optional[dict] = None
    ) -> AsyncIterator[TracedChainResponse[str]]:
        if not question:
            raise Exception("Input required: question")

        return super().run_stream(
            question=question,
            config=config,
        )

    def run_batch(self, inputs: list[str], config: Optional[dict] = None) -> list[str]:  # type: ignore[override]
        return super().run_batch(
            inputs=[{"question": i} for i in inputs],
            config=config,
        )
