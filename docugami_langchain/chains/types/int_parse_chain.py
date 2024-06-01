from typing import Any, AsyncIterator, Optional, Union

from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
)

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.output_parsers.int import IntOutputParser
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class IntParseChain(BaseDocugamiChain[int]):

    _parser = IntOutputParser()

    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """

        def direct_parse(x: dict) -> int:
            return self._parser.parse(str(x["value_text"]))

        def use_llm(x: dict) -> bool:
            try:
                direct_parse(x)
                return False  # direct parse works, no need for LLM
            except Exception:
                return True

        # Try LLM only if simple parsing not enough
        return RunnableBranch(
            (
                RunnableLambda(use_llm),
                super().runnable(),
            ),
            RunnableLambda(direct_parse),
        )

    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "value_text",
                    "VALUE TEXT",
                    "The value expression that needs to be parsed, in rough natural language with possible typos or OCR glitches.",
                ),
            ],
            output=RunnableSingleParameter(
                "parsed_int",
                "PARSED INT",
                "The result of parsing the value expression, as an integer value.",
            ),
            task_description="parses input text values specified in rough natural language, producing output strictly as an integer value that best represents the input text",
            additional_instructions=[
                "- Produce output as a value in integer format (parseable in python) if you find a value.",
                "- If you cannot find any value that represents the input text, don't output anything",
                "- The input data will sometimes by messy, with typos or non-standard formats. Try to guess the value as best as you can, by trying to ignore typical typos and OCR glitches.",
                "- If the value is ambiguous, assume it is the lowest value it could be.",
                "- If multiple values are specified, pick the first one.",
                "- ONLY output the parsed integer value without any commentary, explanation, or listing any assumptions. Your output must EXACTLY match the python integer format.",
            ],
            additional_runnables=[self._parser],
            include_output_instruction_suffix=True,
        )

    def run(  # type: ignore[override]
        self, value_text: str, config: Optional[dict] = None
    ) -> TracedResponse[int]:
        if not value_text:
            raise Exception("Input required: value_text")

        return super().run(
            value_text=value_text,
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self, **kwargs: Any
    ) -> AsyncIterator[TracedResponse[int]]:
        raise NotImplementedError()

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[str],
        config: Optional[dict] = None,
        return_exceptions: bool = True,
    ) -> list[Union[int, Exception]]:
        return super().run_batch(
            inputs=[
                {
                    "value_text": i,
                }
                for i in inputs
            ],
            config=config,
            return_exceptions=return_exceptions,
        )
