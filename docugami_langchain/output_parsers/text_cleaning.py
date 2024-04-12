from langchain_core.output_parsers import BaseOutputParser

from docugami_langchain.utils.string_cleanup import clean_text


class TextCleaningOutputParser(BaseOutputParser[str]):
    """Clean the output of an LLM call to make it more parseable."""

    protect_nested_strings: bool = False
    """Indicates whether nested strings should be left alone."""

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "text_cleaning_output_parser"

    def parse(self, text: str) -> str:
        """Parse the output of an LLM call."""

        text = text.strip()
        if not text:
            return ""

        return clean_text(text, self.protect_nested_strings)
