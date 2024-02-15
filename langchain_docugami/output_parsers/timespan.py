from __future__ import annotations

import re

import wordtodigits
from dateutil.relativedelta import relativedelta
from langchain_core.output_parsers import BaseOutputParser

ORDINAL_MAP = {
    "first": "1st",
    "second": "2nd",
    "third": "3rd",
    "fourth": "4th",
    "fifth": "5th",
    "sixth": "6th",
    "seventh": "7th",
    "eighth": "8th",
    "ninth": "9th",
    "tenth": "10th",
}

NULL_TIMESPAN = "0:0:0:0:0:0"
CANONICAL_TIMESPAN_COMPONENT_COUNT = len(NULL_TIMESPAN.split(":"))


class TimeSpan:
    delta = relativedelta()

    def __str__(self) -> str:
        d = self.delta.normalized()
        return str(f"{d.years}:{d.months}:{d.days}:{d.hours}:{d.minutes}:{d.seconds}")  # type: ignore

    def __eq__(self, __value: object) -> bool:
        return str(self) == str(__value)

    def __repr__(self) -> str:
        return self.__str__()

    def __init__(self, text: str) -> None:
        self.delta = TimeSpan._parse_canonical(text)

        try:
            # First, see if the request string is already in canonical format
            self.delta = TimeSpan._parse_canonical(text)
        except Exception:
            # Next, see if we can search for a timespan inside the request string
            # using regex
            parsed_regex_timespan = TimeSpan.search_string(text)
            if parsed_regex_timespan:
                self.delta = parsed_regex_timespan.delta

        # If we were unable to set, fail
        if not self.delta or self.delta == relativedelta():
            raise Exception(f"Failed to parse given timespan string: {text}")

    def is_zero(self) -> bool:
        return str(self) == NULL_TIMESPAN

    @staticmethod
    def search_string(text: str) -> TimeSpan | None:
        # normalize text
        text = wordtodigits.convert(text)
        text = TimeSpan._normalize_ordinals(text)

        years = TimeSpan._find_value_by_unit(text, ["year", "years", "anniversary"])
        months = TimeSpan._find_value_by_unit(text, ["month", "months"])
        days = TimeSpan._find_value_by_unit(text, ["day", "days"])
        hours = TimeSpan._find_value_by_unit(text, ["hour", "hours"])
        minutes = TimeSpan._find_value_by_unit(text, ["minute", "minutes"])
        seconds = TimeSpan._find_value_by_unit(text, ["second", "seconds"])

        result = TimeSpan(f"{years}:{months}:{days}:{hours}:{minutes}:{seconds}")
        if result.delta == relativedelta():
            # Nothing was found, result is empty
            return None
        else:
            return result

    @staticmethod
    def _find_value_by_unit(
        text: str, unit_variations: list[str], max_sep: int = 10
    ) -> int:
        for unit in unit_variations:
            regex = r"(\d+).{0," + str(max_sep) + r"}\b" + unit + r"\b"
            match = re.search(regex, text, flags=re.IGNORECASE)
            if match:
                return int(match.groups()[0])

        # not found
        return 0

    @staticmethod
    def _normalize_ordinals(text: str) -> str:
        for key, value in ORDINAL_MAP.items():
            text = re.sub(r"\b" + key + r"\b", f" {value} ", text)

        return text

    @staticmethod
    def _parse_canonical(canonical_text: str) -> relativedelta:
        components = canonical_text.split(":")

        # check we have the right # of components
        if len(components) != CANONICAL_TIMESPAN_COMPONENT_COUNT:
            raise Exception(
                f"Provided text does not have the expected {CANONICAL_TIMESPAN_COMPONENT_COUNT} number of components: "
                + canonical_text
            )

        # make sure all components are integers
        for c in components:
            if not c.isnumeric():
                raise Exception(f"Component is non-numeric: {c}")

        # build dateutil relativedelta
        years, months, days, hours, minutes, seconds = components
        return relativedelta(
            years=int(years),
            months=int(months),
            days=int(days),
            hours=int(hours),
            minutes=int(minutes),
            seconds=int(seconds),
        ).normalized()


class TimespanOutputParser(BaseOutputParser[TimeSpan]):
    """Parse the output of an LLM call as a timestamp."""

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "timespan_output_parser"

    def parse(self, text: str) -> TimeSpan:
        """Parse the output of an LLM call."""
        return TimeSpan(text)
