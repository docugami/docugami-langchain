from typing import AsyncIterator, Optional, Union

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.chains.types import DataTypeWithUnit
from docugami_langchain.chains.types.common import DataType
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class DataTypeDetectionChain(BaseDocugamiChain[DataTypeWithUnit]):
    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "text",
                    "TEXT",
                    "The text that needs to be classified by data type and unit, in rough natural language with possible typos or OCR glitches.",
                ),
            ],
            output=RunnableSingleParameter(
                "data_type_json",
                "DATA TYPE JSON",
                "A JSON blob with the most likely data type (`type`) and the optional unit (`unit`) that best represents the given text.",
            ),
            task_description="detects the most likely data type for given text and produces valid JSON output per the given examples",
            additional_instructions=[
                """- Here is an example of a valid JSON blob for your output. Please STRICTLY follow this format:
{
  "type": $TYPE,
  "unit": $UNIT
}""",
                "- $TYPE is the (string) predominant data type of the given text, and must be one of these values: "
                + ", ".join([t.value for t in DataType]),
                "- $UNIT is the unit of the data presented by the given text, for example a currency, meters or square feet. If you cannot detect a unit, leave this blank.",
            ],
            additional_runnables=[PydanticOutputParser(pydantic_object=DataTypeWithUnit)],  # type: ignore
            stop_sequences=["TEXT:", "<|eot_id|>"],
            include_output_instruction_suffix=True,
        )

    def run(  # type: ignore[override]
        self,
        text: str,
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[DataTypeWithUnit]:
        if not text:
            raise Exception("Input required: text")

        return super().run(
            text=text,
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        text: str,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[DataTypeWithUnit]]:
        raise NotImplementedError()

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[str],
        config: Optional[RunnableConfig] = None,
        return_exceptions: bool = True,
    ) -> list[Union[DataTypeWithUnit, Exception]]:
        return super().run_batch(
            inputs=[
                {
                    "text": i,
                }
                for i in inputs
            ],
            config=config,
            return_exceptions=return_exceptions,
        )
