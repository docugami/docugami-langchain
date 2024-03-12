from docugami_langchain import __all__

EXPECTED_BASE = [
    "TracedResponse",
    "CitationLink",
    "Citation",
    "CitedAnswer",
    "BaseRunnable",
    "RunnableParameters",
    "RunnableSingleParameter",
]

EXPECTED_AGENTS = [
    "ReWOOAgent",
    "ReActAgent",
    "ToolRouterAgent",
]

EXPECTED_CHAINS = [
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
    "SimpleRAGChain",
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

EXPECTED_RETRIEVERS = ["SearchType", "FusedSummaryRetriever"]

EXPECTED_TOOLS = [
    "SmallTalkTool",
    "GeneralKnowlegeTool",
    "HumanInterventionTool",
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
