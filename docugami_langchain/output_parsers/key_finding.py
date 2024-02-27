from typing import Optional

from langchain_core.output_parsers import BaseOutputParser


class KeyfindingOutputParser(BaseOutputParser[str]):
    """Parse the output of an LLM call using an output key and other options to
    find the correct output."""

    output_key: Optional[str] = None
    """The output key that marks the output."""

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "key_finding_output_parser"

    def parse(self, text: str) -> str:
        """Parse the output of an LLM call."""

        if "\n" in text and text.startswith("Sure"):
            # remove any pleasantries header
            text = "\n".join(text.splitlines()[1:])

        if self.output_key:
            # If output key specified, only use text after the key if in output
            key_marker = self.output_key + ":"
            key_pos = text.lower().find(key_marker.lower())
            if key_pos != -1:
                text = text[key_pos + len(key_marker) :].strip()

        return text.strip()
