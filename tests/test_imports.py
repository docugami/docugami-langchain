from langchain_docugami import __all__

EXPECTED_CHAINS = [
    "TracedChainResponse",
    "BaseDocugamiChain",
    "AnswerChain",
    "ChainSingleParameter",
    "ChainParameters",
]

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

EXPECTED_PROMPTS = [
    "SYSTEM_MESSAGE_CORE",
    "ASSISTANT_SYSTEM_MESSAGE",
    "CREATE_CHUNK_SUMMARY_PROMPT",
    "CREATE_CHUNK_SUMMARY_SYSTEM_MESSAGE",
    "CREATE_FULL_DOCUMENT_SUMMARY_PROMPT",
    "CREATE_FULL_DOCUMENT_SUMMARY_SYSTEM_MESSAGE",
    "CREATE_DIRECT_RETRIEVAL_TOOL_DESCRIPTION_PROMPT",
    "CREATE_DIRECT_RETRIEVAL_TOOL_SYSTEM_MESSAGE",
]

EXPECTED_RETRIEVERS = ["SearchType", "FusedSummaryRetriever"]

EXPECTED_ALL = (
    EXPECTED_CHAINS
    + EXPECTED_DOCUMENT_LOADERS
    + EXPECTED_OUTPUT_PARSERS
    + EXPECTED_PROMPTS
    + EXPECTED_RETRIEVERS
)


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
