from datetime import datetime
from typing import Any, AsyncIterator, Optional, Union

from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
)

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.output_parsers.datetime import DatetimeOutputParser
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter

OUTPUT_FORMAT = "%m/%d/%Y"


class DateParseChain(BaseDocugamiChain[datetime]):

    _parser = DatetimeOutputParser()

    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """

        def direct_parse(x: dict) -> datetime:
            return self._parser.parse(str(x["date_text"]))

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
                    "date_text",
                    "DATE TEXT",
                    "The date expression that needs to be parsed, in rough natural language with possible typos or OCR glitches.",
                ),
            ],
            output=RunnableSingleParameter(
                "parsed_date",
                "PARSED DATE",
                f"The result of parsing the date expression, in {OUTPUT_FORMAT} format.",
            ),
            task_description=f"parses date expressions specified in rough natural language, producing output strictly in the standard {OUTPUT_FORMAT} format",
            additional_instructions=[
                f"- Produce output as a date in {OUTPUT_FORMAT} format if you find a date.",
                "- If you cannot find anything resembling a date (in any format even if incomplete or messy), don't output anything",
                "- The input data will sometimes by messy, with typos or non-standard formats. Try to guess the date as best as you can, by trying to ignore typical typos and OCR glitches.",
                f"- If a two digit year is specified, assume the same century as the current year i.e. {str(datetime.now().year)[:2]}",
                f"- If the year is not specified at all, assume current year i.e. {datetime.now().year}",
                "- If the day is not specified, assume the first of the month.",
                "- If the date is ambiguous, assume it is the latest date it could be.",
                "- If multiple dates are specified, pick the first one.",
                f"- ONLY output the parsed date expression without any commentary, explanation, or listing any assumptions. Your output must EXACTLY match the required {OUTPUT_FORMAT} format.",
            ],
            additional_runnables=[self._parser],
            include_output_instruction_suffix=True,
        )

    def run(  # type: ignore[override]
        self, date_text: str, config: Optional[dict] = None
    ) -> TracedResponse[datetime]:
        if not date_text:
            raise Exception("Input required: date_text")

        return super().run(
            date_text=date_text,
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self, **kwargs: Any
    ) -> AsyncIterator[TracedResponse[datetime]]:
        raise NotImplementedError()

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[str],
        config: Optional[dict] = None,
        return_exceptions: bool = True,
    ) -> list[Union[datetime, Exception]]:
        return super().run_batch(
            inputs=[
                {
                    "date_text": i,
                }
                for i in inputs
            ],
            config=config,
            return_exceptions=return_exceptions,
        )
