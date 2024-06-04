from typing import AsyncIterator, Optional, Union

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.chains.types.common import ParsedAddress
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class AddressParseChain(BaseDocugamiChain[ParsedAddress]):
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
                "address_type_json",
                "ADDRESS TYPE JSON",
                "A JSON blob representing the address per given instructions.",
            ),
            task_description="detects the from the given text and produces valid JSON output per the given examples",
            additional_instructions=[
                """- Here is an example of a valid JSON blob for your output. Please STRICTLY follow this format:
{
  "street": $STREET,
  "city": $CITY
  "state": $STATE
  "zip": $ZIP
}""",
                "- $STREET is the optional (string) street part of the address"
                "- $CITY is the optional (string) city part of the address"
                "- $STATE is the optional (string) state part of the address" 
                "- $ZIP is the optional (string) zip part of the address"
            ],
            additional_runnables=[PydanticOutputParser(pydantic_object=ParsedAddress)],  # type: ignore
            stop_sequences=["TEXT:", "<|eot_id|>"],
            include_output_instruction_suffix=True,
        )

    def run(  # type: ignore[override]
        self,
        text: str,
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[ParsedAddress]:
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
    ) -> AsyncIterator[TracedResponse[ParsedAddress]]:
        raise NotImplementedError()

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[str],
        config: Optional[RunnableConfig] = None,
        return_exceptions: bool = True,
    ) -> list[Union[ParsedAddress, Exception]]:
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
