import pytest

from docugami_langchain.utils.string_cleanup import (
    escape_non_escaped_backslashes,
    replace_null_outside_quotes,
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
    assert escape_non_escaped_backslashes(text) == expected


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
