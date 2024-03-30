import pytest

from docugami_langchain.utils.string_cleanup import (
    escape_non_escape_sequence_backslashes,
    replace_null_outside_quotes,
    unescape_escaped_chars_outside_quoted_strings,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This is a test.", "This is a test."),  # Pass through normal strings
        (
            'Quote \\" Line break\\nNew line\\nTab\\t',
            'Quote \\" Line break\\nNew line\\nTab\\t',
        ),  # Known escape sequences are not escaped
        (
            "This \\is a test.",
            "This \\\\is a test.",
        ),  # Non-escape backslashes are escaped
        ("\\a\\b\\c", "\\\\a\\\\b\\\\c"),  # Multiple non-escape backslashes are escaped
        ("Line\\nBreak \\and tab\\t", "Line\\nBreak \\\\and tab\\t"),  # Mixed cases
        ("", ""),  # Empty string
        ("\\This is a test", "\\\\This is a test"),  # Backslash at the start is handled
        ("This is a test\\", "This is a test\\\\"),  # Backslash at the end is handled
    ],
)
def test_escape_non_escaped_backslashes(text: str, expected: str) -> None:
    assert escape_non_escape_sequence_backslashes(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Just some text.", "Just some text."),  # No "null"
        (
            "only full word null not substringnullsofstring replaced",
            'only full word "" not substringnullsofstring replaced',
        ),  # Substring null
        (
            "null should become empty.",
            '"" should become empty.',
        ),  # "null" outside quotes
        ('"null" should stay.', '"null" should stay.'),  # "null" inside quotes
        (
            'Outside null and "inside null".',
            'Outside "" and "inside null".',
        ),  # Mix inside and outside
        ("", ""),  # Empty string
        ("NuLl", '""'),  # Case-insensitive "null"
        (
            'Multiple nulls null "null" null.',
            'Multiple nulls "" "null" "".',
        ),  # Multiple instances
        (
            '"Just some quoted text."',
            '"Just some quoted text."',
        ),  # Edge case with quotes but no "null"
    ],
)
def test_replace_null_outside_quotes(text: str, expected: str) -> None:
    assert replace_null_outside_quotes(text) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("abc def", "abc def"),  # No quotes
        (
            'abc \\"def',
            'abc "def',
        ),  # An escaped double quote outside a quoted string should be unescaped
        (
            "abc \\'def",
            "abc 'def",
        ),  # An escaped single quote outside a quoted string should be unescaped
        (
            'abc \\"def\\" ghi',
            'abc "def" ghi',
        ),  # If the double quotes marking a quoted string are escaped, the should be unescaped
        (
            "abc \\'def\\' ghi",
            "abc 'def' ghi",
        ),  # If the single quotes marking a quoted string are escaped, the should be unescaped
        (
            'abc "d\\"ef" ghi',
            'abc "d\\"ef" ghi',
        ),  # A properly escaped double quote inside a double quoted string should NOT be unescaped
        (
            'SELECT AVG(CAST(\\"Crime Insurance Limit\\" AS REAL)) FROM \\"1.Spreadsheet Services Agreement Luis\\"',
            'SELECT AVG(CAST("Crime Insurance Limit" AS REAL)) FROM "1.Spreadsheet Services Agreement Luis"',
        ),  # An example SQL Query with outer quotes that should be correctly un-escaped
        (
            'SELECT COUNT(\*) FROM "1.Spreadsheet Services Agreement Luis"',
            'SELECT COUNT(*) FROM "1.Spreadsheet Services Agreement Luis"',
        ),  # An example SQL Query with an escaped asterisk that should be un-escaped
    ],
)
def test_unescape_outside_strings(query: str, expected: str) -> None:
    assert unescape_escaped_chars_outside_quoted_strings(query) == expected
