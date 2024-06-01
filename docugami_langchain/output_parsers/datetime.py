# Source: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/output_parsers/datetime.py


from datetime import datetime

from dateutil.parser import parse
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser


class DatetimeOutputParser(BaseOutputParser[datetime]):
    """Parse the output of an LLM call to a datetime."""

    def parse(self, response: str) -> datetime:
        try:
            return parse(response.strip())
        except ValueError as e:
            raise OutputParserException(
                f"Could not parse datetime string: {response}"
            ) from e

    @property
    def _type(self) -> str:
        return "datetime"
