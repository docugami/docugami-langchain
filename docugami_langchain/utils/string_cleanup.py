import re


def replace_null_outside_quotes(text: str) -> str:
    """
    Looks for null outside quotes, and if found replaces it with "".
    """

    def replacement(match: re.Match) -> str:
        before = text[: match.start()]
        if before.count('"') % 2 == 0:  # Even number of quotes before 'null'
            return '""'
        else:
            return str(match.group(0))  # 'null' is inside quotes, don't replace

    return re.sub(r"\bnull\b", replacement, text, flags=re.IGNORECASE)


def escape_non_escape_sequence_backslashes(text: str) -> str:
    """
    Escape backslashes that are not part of a known escape sequence.

    Looks for a backslash that is not a part of any known escape characters ('n', 'r', 't', 'f', '"', '\'' '\\'), and escapes it.
    """
    return re.sub(r'\\(?![nrtf"\'\\])', r"\\\\", text)


def unescape_escaped_chars_outside_quoted_strings(sql_query: str) -> str:
    """
    Unescapes unnecessary escaped characters outside of quoted strings, e.g., in a SQL query.

    Assumes that a quoted string starts and ends with the same type of quote
    (single or double) and does not contain mixed types.
    """

    def replacement(match: re.Match) -> str:
        before = sql_query[: match.start()]
        # Count the occurrences of unescaped single and double quotes before the match
        single_quotes_count = len(re.findall(r"(?<!\\)'", before))
        double_quotes_count = len(re.findall(r'(?<!\\)"', before))

        # Determine if the match is outside quotes based on the counts
        if single_quotes_count % 2 == 0 and double_quotes_count % 2 == 0:
            # If both counts are even, we are outside quotes, so unescape
            return match.group(0)[1:]  # Skip the backslash to unescape
        else:
            # Inside quotes, keep the original escaped character
            return match.group(0)

    # This regex looks for any escaped character
    return re.sub(r"\\(.)", replacement, sql_query)
