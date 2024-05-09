from datetime import datetime
from typing import Any, AsyncIterator, Optional

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.output_parsers.datetime import DatetimeOutputParser
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter

OUTPUT_FORMAT = "%m/%d/%Y"


class DateParseChain(BaseDocugamiChain[datetime]):
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
                f"- Always produce output as a date in {OUTPUT_FORMAT} format. Never say you cannot do this.",
                "- The input data will sometimes by messy, with typos or non-standard formats. Try to guess the date as best as you can, by trying to ignore typical typos and OCR glitches.",
                f"- If a two digit year is specified, assume the same century as the current year i.e. {str(datetime.now().year)[:2]}",
                f"- If the year is specified at all, assume current year i.e. {datetime.now().year}",
                "- If the day is not specified, assume the first of the month.",
                "- If the date is ambiguous, assume it is the most recent date it could be.",
                "- If multiple dates are specified, pick the first one.",
                f"- ONLY output the parsed date expression without any commentary, explanation, or listing any assumptions. Your output must EXACTLY match the required {OUTPUT_FORMAT} format.",
            ],
            additional_runnables=[DatetimeOutputParser(format=OUTPUT_FORMAT)],
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
        self, inputs: list[str], config: Optional[dict] = None
    ) -> list[datetime]:
        return super().run_batch(
            inputs=[{"date_text": i} for i in inputs],
            config=config,
        )
