import re
import string

UN_ESCAPE_MAP = {
    "\\\\": "\\",
    "\\'": "'",
    '\\"': '"',
    "\\b": "\b",
    "\\f": "\f",
    "\\n": "\n",
    "\\r": "\r",
    "\\t": "\t",
    "\\(": "(",
    "\\)": ")",
    "\\[": "[",
    "\\]": "]",
    "\\_": "_",
}


def _replace_null_outside_quotes(text: str) -> str:
    """
    Looks for null outside quotes, and if found replaces it with "".
    """

    text = text.strip()
    if not text:
        return ""

    def replacement(match: re.Match) -> str:
        before = text[: match.start()]
        if before.count('"') % 2 == 0:  # Even number of quotes before 'null'
            return '""'
        else:
            return str(match.group(0))  # 'null' is inside quotes, don't replace

    return re.sub(r"\bnull\b", replacement, text, flags=re.IGNORECASE)


def _escape_non_escape_sequence_backslashes(text: str) -> str:
    """
    Escape backslashes that are not part of a known escape sequence.

    Looks for a backslash that is not a part of any known escape characters ('n', 'r', 't', 'f', '"', "'", '\\'), and escapes it.
    """
    return re.sub(r'\\(?![nrtf"\'\\])', r"\\\\", text)


def _unescape_escaped_chars_outside_quoted_strings(text: str) -> str:
    """
    Unescapes unnecessary escaped characters outside of quoted strings, e.g., in a SQL query.

    Assumes that a quoted string starts and ends with the same type of quote
    (single or double) and does not contain mixed types.
    """
    text = text.strip()
    if not text:
        return ""

    def replacement(match: re.Match) -> str:
        before = text[: match.start()]
        # Count the occurrences of unescaped single and double quotes before the match
        single_quotes_count = len(re.findall(r"(?<!\\)'", before))
        double_quotes_count = len(re.findall(r'(?<!\\)"', before))

        matched_sequence = match.group(0)
        # Determine if the match is outside quotes based on the counts
        if single_quotes_count % 2 == 0 and double_quotes_count % 2 == 0:
            # If both counts are even, we are outside quotes, so unescape
            if matched_sequence in UN_ESCAPE_MAP:
                return UN_ESCAPE_MAP[matched_sequence]
            else:
                return matched_sequence[1:]  # Skip the backslash to unescape
        else:
            # Inside quotes, keep the original escaped character
            return matched_sequence

    # This regex looks for any escaped character
    return re.sub(r"\\(.)", replacement, text)


def _unescaped_all_escape_sequences(text: str) -> str:
    """
    Unescapes over-escaped text, i.e. where valid escape sequences are escaped.
    """

    text = text.strip()
    if not text:
        return ""

    for sequence in UN_ESCAPE_MAP:
        text = text.replace(sequence, UN_ESCAPE_MAP[sequence])

    return text


def clean_text(text: str, protect_nested_strings: bool = False) -> str:
    """
    Cleans the given text.
    """

    text = text.strip()
    if not text:
        return ""

    text = "".join(filter(lambda x: x in string.printable, text))  # non-printable chars
    text = _replace_null_outside_quotes(text)

    if protect_nested_strings:
        text = _escape_non_escape_sequence_backslashes(text)
        text = _unescape_escaped_chars_outside_quoted_strings(text)
    else:
        text = _unescaped_all_escape_sequences(text)

    return text
