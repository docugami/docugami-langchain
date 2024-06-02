from langchain_core.output_parsers import BaseOutputParser


class FloatOutputParser(BaseOutputParser[float]):
    """Parse the output of an LLM call as a float."""

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "float_output_parser"

    def parse(self, text: str) -> float:
        """Parse the output of an LLM call."""
        text = text.strip().replace(",", "")

        return float(text)
