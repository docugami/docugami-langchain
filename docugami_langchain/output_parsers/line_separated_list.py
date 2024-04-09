import re

from langchain_core.output_parsers.list import ListOutputParser

from docugami_langchain.utils.string_cleanup import clean_text


class LineSeparatedListOutputParser(ListOutputParser):
    """Parse the output of an LLM call as a line-separated list."""

    strip_ordinals: bool = True
    """Indicates whether list ordinals should be stripped"""

    ignore_pleasantry: bool = True
    """Indicates whether initial pleasantry should be ignored"""

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "line_separated_list_output_parser"

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call."""

        text = text.strip()
        if not text:
            return []

        text = clean_text(text)

        if self.ignore_pleasantry and "\n" in text and text.startswith("Sure"):
            # remove any pleasantries header
            text = "\n".join(text.splitlines()[1:])

        items = []
        for line in text.splitlines():
            if self.strip_ordinals:
                line = re.sub(r"^\d+\.\s*", "", line.strip()).strip()
            if line:
                items.append(line)

        return items
