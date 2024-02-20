from typing import AsyncIterator, Literal, Optional, Tuple

from docugami_langchain.chains.base import BaseDocugamiChain, TracedChainResponse
from docugami_langchain.chains.params import ChainParameters, ChainSingleParameter


class ElaborateChunkChain(BaseDocugamiChain[str]):
    def chain_params(self) -> ChainParameters:
        return ChainParameters(
            inputs=[
                ChainSingleParameter(
                    "contents",
                    "CONTENTS",
                    "Contents of the chunk that needs to be elaborated",
                ),
                ChainSingleParameter(
                    "format",
                    "FORMAT",
                    "Format of the contents, and expected elaborated output.",
                ),
            ],
            output=ChainSingleParameter(
                "elaboration",
                "ELABORATION",
                "Elaboration generated per the given rules.",
            ),
            task_description="elaborates some given text, while minimizing loss of key details",
            additional_instructions=[
                "- Your generated elaboration should be in the same format as the given document, using the same overall schema.",
                "- Only elaborate, don't try to change any facts in the chunk even if they appear incorrect to you.",
                "- Include as many facts and data points from the original chunk as you can, in your elaboration.",
                "- Pay special attention to monetary amounts, dates, names of people and companies, etc and include in your elaboration.",
                "- Aim for the elaboration to be twice as long as the given text. Be as descriptive as possible within these limits.",
                "- Produce all output as one paragraph, don't include any paragraph breaks. However, if the input contains list or table formatting, try to include that in your output as well.",
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
    ) -> AsyncIterator[TracedChainResponse[str]]:
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
