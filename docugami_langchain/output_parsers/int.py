from langchain_core.output_parsers import BaseOutputParser


class IntOutputParser(BaseOutputParser[int]):
    """Parse the output of an LLM call as an integer."""

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "int_output_parser"

    def parse(self, text: str) -> int:
        """Parse the output of an LLM call."""
        text = text.strip().replace(",", "")

        return int(text)
