from datetime import datetime
from typing import Any, AsyncIterator, Optional, Tuple

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.output_parsers.datetime import DatetimeOutputParser
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter

OUTPUT_FORMAT = "%m/%d/%Y"


class DateAddChain(BaseDocugamiChain[datetime]):
    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "start_date",
                    "START DATE",
                    "The start date, in rough natural language with possible typos or OCR glitches.",
                ),
                RunnableSingleParameter(
                    "end_date_or_duration",
                    "END DATE OR DURATION",
                    "A time duration or end date, in rough natural language with possible typos or OCR glitches.",
                ),
            ],
            output=RunnableSingleParameter(
                "added_date",
                "ADDED DATE",
                f"The result of adding the start date and duration, in {OUTPUT_FORMAT} format. If an end date is specified, just output that in the requested output format. "
                + "Ignore any typos or mistakes in the data and just output the added date as requested.",
            ),
            task_description=f"adds dates and time durations written in rough natural language, calculating the end date and producing output strictly in the {OUTPUT_FORMAT} format",
            additional_instructions=[
                f"- Always produce output as a date in {OUTPUT_FORMAT} format. Never say you cannot do this.",
                "- If the time duration is in fact a date itself, just use that as the output.",
                "- The input data will sometimes by messy, with typos or non-standard formats. "
                + "Try to guess the date or duration as best as you can, by trying to ignore typical typos and OCR glitches.",
            ],
            additional_runnables=[DatetimeOutputParser(format=OUTPUT_FORMAT)],
        )

    def run(  # type: ignore[override]
        self, start_date: str, end_date_or_duration: str, config: Optional[dict] = None
    ) -> TracedResponse[datetime]:
        if not start_date or not end_date_or_duration:
            raise Exception("Inputs required: start_date, end_date_or_duration")

        return super().run(
            start_date=start_date,
            end_date_or_duration=end_date_or_duration,
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self, **kwargs: Any
    ) -> AsyncIterator[TracedResponse[datetime]]:
        raise NotImplementedError()

    def run_batch(  # type: ignore[override]
        self, inputs: list[Tuple[str, str]], config: Optional[dict] = None
    ) -> list[datetime]:
        return super().run_batch(
            inputs=[
                {
                    "start_date": i[0],
                    "end_date_or_duration": i[1],
                }
                for i in inputs
            ],
            config=config,
        )
