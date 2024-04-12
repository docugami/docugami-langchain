from typing import Any, AsyncIterator, Optional

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.output_parsers.timespan import TimeSpan, TimespanOutputParser
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter

OUTPUT_FORMAT = "year:month:day:hour:minute:second"


class TimespanParseChain(BaseDocugamiChain[TimeSpan]):
    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "timespan_text",
                    "TIME SPAN TEXT",
                    "The timespan expression that needs to be parsed, in rough natural language with possible typos or OCR glitches.",
                ),
            ],
            output=RunnableSingleParameter(
                "parsed_timespan",
                "PARSED TIME SPAN",
                f"The result of parsing the timespan expression, in {OUTPUT_FORMAT} format.",
            ),
            task_description=f"parses time span expressions specified in rough natural language, producing output strictly in the {OUTPUT_FORMAT} format",
            additional_instructions=[
                f"- Always produce output as a timespan in {OUTPUT_FORMAT} format. Never say you cannot do this.",
                "- The input data will sometimes by messy, with typos or non-standard formats. Try to guess the timespan as best as you can, by trying to ignore typical typos and OCR glitches.",
            ],
            additional_runnables=[TimespanOutputParser()],
            include_output_instruction_suffix=True,
        )

    def run(  # type: ignore[override]
        self, timespan_text: str, config: Optional[dict] = None
    ) -> TracedResponse[TimeSpan]:
        if not timespan_text:
            raise Exception("Input required: timespan_text")

        return super().run(
            timespan_text=timespan_text,
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self, **kwargs: Any
    ) -> AsyncIterator[TracedResponse[TimeSpan]]:
        raise NotImplementedError()

    def run_batch(  # type: ignore[override]
        self, inputs: list[str], config: Optional[dict] = None
    ) -> list[TimeSpan]:
        return super().run_batch(
            inputs=[{"timespan_text": i} for i in inputs],
            config=config,
        )
