from typing import AsyncIterator, Literal, Optional, Tuple

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base_chain import BaseChainRunnable
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class ElaborateChunkChain(BaseChainRunnable[str]):
    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "contents",
                    "CONTENTS",
                    "Contents of the chunk that needs to be elaborated",
                ),
                RunnableSingleParameter(
                    "format",
                    "FORMAT",
                    "Format of the contents, and expected elaborated output.",
                ),
            ],
            output=RunnableSingleParameter(
                "elaboration",
                "ELABORATION",
                "Elaboration generated per the given rules.",
            ),
            task_description="elaborates some given text, while minimizing loss of key details",
            stop_sequences=["CONTENTS:", "FORMAT:"],
            additional_instructions=[
                "- Your generated elaboration should be in the same format as the given document, using the same overall schema.",
                "- Only elaborate, don't try to change any facts in the chunk even if they appear incorrect to you.",
                "- Include as many facts and data points from the original chunk as you can, in your elaboration.",
                "- Pay special attention to key facts like monetary amounts, dates, addresses, names of people and companies, etc and include in your summary.",
                "- Aim for the elaboration to be twice as long as the given text. Be as descriptive as possible within these limits.",
            ],
        )

    def run(  # type: ignore[override]
        self,
        contents: str,
        format: Literal["xml", "text"] = "text",
        config: Optional[dict] = None,
    ) -> str:
        if not contents or not format:
            raise Exception("Inputs required: contents, format")

        return super().run(
            contents=contents,
            format=format,
            config=config,
        )

    def run_stream(  # type: ignore[override]
        self,
        contents: str,
        format: str,
        config: Optional[dict] = None,
    ) -> AsyncIterator[TracedResponse[str]]:
        if not contents or not format:
            raise Exception("Inputs required: contents, format")

        return super().run_stream(
            contents=contents,
            format=format,
            config=config,
        )

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[Tuple[str, str]],
        config: Optional[dict] = None,
    ) -> list[str]:
        return super().run_batch(
            inputs=[
                {
                    "contents": i[0],
                    "format": i[1],
                }
                for i in inputs
            ],
            config=config,
        )
