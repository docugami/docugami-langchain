from langchain_docugami import __all__


EXPECTED_DOCUMENT_LOADERS = [
    "DocugamiLoader",
]

EXPECTED_OUTPUT_PARSERS = [
    "KeyfindingOutputParser",
    "LineSeparatedListOutputParser",
    "SoftReActJsonSingleInputOutputParser",
    "SQLFindingOutputParser",
    "TimeSpan",
    "TimespanOutputParser",
]

EXPECTED_ALL = EXPECTED_DOCUMENT_LOADERS + EXPECTED_OUTPUT_PARSERS


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
