import pytest

from docugami_langchain.utils.string_cleanup import (
    _escape_non_escape_sequence_backslashes,
    _unescape_escaped_chars_outside_quoted_strings,
    clean_text,
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
    assert _escape_non_escape_sequence_backslashes(text) == expected


@pytest.mark.parametrize(
    "text,expected",
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
            "abc \\ndef",
            "abc \ndef",
        ),  # An escaped newline outside a quoted string should be unescaped
        (
            'abc \\"def\\" ghi',
            'abc "def" ghi',
        ),  # If the double quotes marking a quoted string are escaped, they should be unescaped
        (
            "abc \\'def\\' ghi",
            "abc 'def' ghi",
        ),  # If the single quotes marking a quoted string are escaped, they should be unescaped
        (
            'abc "d\\"ef" ghi',
            'abc "d\\"ef" ghi',
        ),  # A properly escaped double quote inside a double quoted string should NOT be unescaped
        (
            'abc "d\\nef" ghi',
            'abc "d\\nef" ghi',
        ),  # A properly escaped newline inside a double quoted string should NOT be unescaped
        (
            'SELECT AVG(CAST(\\"Crime Insurance Limit\\" AS REAL)) FROM \\"1.Spreadsheet Services Agreement \\"',
            'SELECT AVG(CAST("Crime Insurance Limit" AS REAL)) FROM "1.Spreadsheet Services Agreement "',
        ),  # An example SQL Query with outer quotes that should be correctly un-escaped
        (
            'SELECT COUNT(\*) FROM "1.Spreadsheet Services Agreement"',
            'SELECT COUNT(*) FROM "1.Spreadsheet Services Agreement"',
        ),  # An example SQL Query with an escaped asterisk that should be un-escaped
    ],
)
def test_unescape_outside_strings(text: str, expected: str) -> None:
    assert _unescape_escaped_chars_outside_quoted_strings(text) == expected


@pytest.mark.parametrize(
    "text,protect_nested_strings,expected",
    [
        (
            '{\n  "key_1": "val1",\n  "key\\_2": true\n}',
            True,
            '{\n  "key_1": "val1",\n  "key\\\\_2": true\n}',
        ),  # an unnecessarily escaped underscore, but protected in nested string to make sure we retain the escape
        (
            '{\n  "key_1": "val1",\n  "key\\_2": true\n}',
            False,
            '{\n  "key_1": "val1",\n  "key_2": true\n}',
        ),  # an unnecessarily escaped underscore, but protected in nested string
    ],
)
def test_clean_text(text: str, protect_nested_strings: bool, expected: str) -> None:
    assert clean_text(text, protect_nested_strings) == expected
