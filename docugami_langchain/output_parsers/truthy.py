from langchain_core.output_parsers import BaseOutputParser


class TruthyOutputParser(BaseOutputParser[bool]):
    """Parse the output of an LLM call as a boolean."""

    TRUTHY_STRINGS = ["true", "yes"]

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "truthy_output_parser"

    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call."""
        text = text.lower()
        return any(substring in text for substring in self.TRUTHY_STRINGS)
