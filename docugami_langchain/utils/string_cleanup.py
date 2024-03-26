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
