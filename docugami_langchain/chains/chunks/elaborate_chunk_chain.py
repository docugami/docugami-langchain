from typing import AsyncIterator, Literal, Optional

from langchain_core.runnables import RunnableConfig

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class ElaborateChunkChain(BaseDocugamiChain[str]):
    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "contents",
                    "CONTENTS",
                    "Contents of the chunk that needs to be elaborated.",
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
            stop_sequences=["CONTENTS:", "FORMAT:", "<|eot_id|>"],
            additional_instructions=[
                "- Your generated elaboration should be in the same format as the given document, using the same overall schema.",
                "- Only elaborate, don't try to change any facts in the chunk even if they appear incorrect to you.",
                "- Include as many facts and data points from the original chunk as you can, in your elaboration.",
                "- Pay special attention to unique facts like monetary amounts, dates, time durations, addresses, names of people and companies, "
                "phone numbers, email address, etc and include these in your elaboration to ensure it does not lose semantic value.",
                "- Aim for the elaboration to be twice as long as the given text. Be as descriptive as possible within these limits.",
            ],
        )

    def run(  # type: ignore[override]
        self,
        contents: str,
        format: Literal["xml", "text"] = "text",
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[str]:
        if not contents or not format:
            raise Exception("Inputs required: contents, format")

        return super().run(
            contents=contents,
            format=format,
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        contents: str,
        format: str,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[str]]:
        if not contents or not format:
            raise Exception("Inputs required: contents, format")

        async for item in super().run_stream(
            contents=contents,
            format=format,
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, str]],
        config: Optional[RunnableConfig] = None,
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
