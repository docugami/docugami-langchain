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


def escape_non_escaped_backslashes(text: str) -> str:
    """
    Escape backslashes that are not part of a known escape sequence.

    Looks for a backslash that is not a part of any known escape characters ('n', 'r', 't', 'f', '"', '\\'), and escapes it.
    """
    return re.sub(r'\\(?![nrtf"\\])', r"\\\\", text)


def unescape_quotes_outside_strings(sql_query: str) -> str:
    """
    Unescapes double quotes outside of quoted strings, e.g. in a SQL query.

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
            return '"'
        else:
            # Inside quotes, keep the original escaped quote
            return match.group(0)

    # This regex looks for escaped double quotes
    return re.sub(r'\\(")', replacement, sql_query)
