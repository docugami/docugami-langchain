from typing import AsyncIterator, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.chains.types import DocugamiDataType
from docugami_langchain.chains.types.common import DataTypes
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class DataTypeDetectionChain(BaseDocugamiChain[DocugamiDataType]):
    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "text_items",
                    "TEXT ITEMS",
                    "The list of text items that needs to be classified by predominant data type, in rough natural language with possible typos or OCR glitches.",
                ),
            ],
            output=RunnableSingleParameter(
                "data_type_json",
                "DATA TYPE JSON",
                "A JSON blob with the predominant data type (`type`) and the optional unit (`unit`) that best represents the given list of text items.",
            ),
            task_description="detects the predominant data type from a list of text items and produces valid JSON output per the given examples",
            additional_instructions=[
                """- Here is an example of a valid JSON blob for your output. Please STRICTLY follow this format:
{{
  "type": $TYPE,
  "unit": $UNIT
}}""",
                "- $TYPE is the (string) predominant data type of the given text items, and must be one of these values: "
                + ", ".join([t.value for t in DataTypes]),
                "- $UNIT is the predominant unit of the data presented by the given text items. If there is no unit, just use the date type value here as well.",
            ],
            additional_runnables=[PydanticOutputParser(pydantic_object=DocugamiDataType)],  # type: ignore
            stop_sequences=["TEXT ITEMS:", "<|eot_id|>"],
            include_output_instruction_suffix=True,
        )

    def run(  # type: ignore[override]
        self,
        text_items: list[str],
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[DocugamiDataType]:
        if not text_items:
            raise Exception("Input required: text_items")

        text_items_numbered = []
        for i, item in enumerate(text_items):
            text_items_numbered.append(f"{i+1}. {item.strip()}")

        return super().run(
            text_items="\n".join(text_items_numbered),
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        text_items: list[str],
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[DocugamiDataType]]:
        raise NotImplementedError()

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[list[str]],
        config: Optional[RunnableConfig] = None,
    ) -> list[DocugamiDataType]:
        raise NotImplementedError()
