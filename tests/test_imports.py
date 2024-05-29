from docugami_langchain import __all__

EXPECTED_BASE = [
    "TracedResponse",
    "BaseRunnable",
    "RunnableParameters",
    "RunnableSingleParameter",
]

EXPECTED_AGENTS = [
    "AgentState",
    "BaseDocugamiAgent",
    "Citation",
    "CitedAnswer",
    "Invocation",
    "StepState",
    "ReActAgent",
    "ToolRouterAgent",
]

EXPECTED_CHAINS = [
    "BaseDocugamiChain",
    "AnswerChain",
    "ElaborateChunkChain",
    "SummarizeChunkChain",
    "SummarizeDocumentChain",
    "DescribeDocumentSetChain",
    "DocugamiExplainedSQLQueryChain",
    "SQLFixupChain",
    "SQLQueryExplainerChain",
    "SQLResultChain",
    "SQLResultExplainerChain",
    "SuggestedQuestionsChain",
    "SuggestedReportChain",
    "ToolFinalAnswerChain",
    "ToolOutputGraderChain",
    "SimpleRAGChain",
    "StandaloneQuestionChain",
    "DataTypeDetectionChain",
    "DataTypes",
    "DocugamiDataType",
    "DateAddChain",
    "DateParseChain",
    "FloatParseChain",
    "TimespanParseChain",
]

EXPECTED_DOCUMENT_LOADERS = [
    "DocugamiLoader",
]

EXPECTED_OUTPUT_PARSERS = [
    "CustomReActJsonSingleInputOutputParser",
    "DatetimeOutputParser",
    "FloatOutputParser",
    "KeyfindingOutputParser",
    "LineSeparatedListOutputParser",
    "SQLFindingOutputParser",
    "TextCleaningOutputParser",
    "TimeSpan",
    "TimespanOutputParser",
    "TruthyOutputParser",
]

EXPECTED_RETRIEVERS = ["SearchType", "FusedSummaryRetriever"]

EXPECTED_TOOLS = [
    "ChatBotTool",
    "CustomReportRetrievalTool",
]


EXPECTED_ALL = (
    EXPECTED_BASE
    + EXPECTED_AGENTS
    + EXPECTED_CHAINS
    + EXPECTED_DOCUMENT_LOADERS
    + EXPECTED_OUTPUT_PARSERS
    + EXPECTED_RETRIEVERS
    + EXPECTED_TOOLS
)


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
