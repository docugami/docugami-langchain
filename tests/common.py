import os
import random
import warnings
from pathlib import Path
from typing import Any, Optional

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from rerankers.models.ranker import BaseRanker

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.config import DEFAULT_RETRIEVER_K, MAX_FULL_DOCUMENT_TEXT_LENGTH
from docugami_langchain.document_loaders.docugami import DocugamiLoader
from docugami_langchain.retrievers.fused_summary import (
    FusedRetrieverKeyValueFetchCallback,
    FusedSummaryRetriever,
    SearchType,
)
from docugami_langchain.retrievers.mappings import (
    build_chunk_summary_mappings,
    build_doc_maps_from_chunks,
    build_full_doc_summary_mappings,
)
from docugami_langchain.tools.common import get_generic_tools
from docugami_langchain.tools.reports import (
    connect_to_excel,
    get_retrieval_tool_for_report,
    report_details_to_report_query_tool_description,
    report_name_to_report_query_tool_function_name,
)
from docugami_langchain.tools.retrieval import (
    docset_name_to_direct_retriever_tool_function_name,
    get_retrieval_tool_for_docset,
    summaries_to_direct_retriever_tool_description,
)
from docugami_langchain.utils.sql import get_table_info

TEST_DATA_DIR = Path(__file__).parent / "testdata"
RAG_TEST_DGML_DOCSET_NAME = "NTSB Aviation Incident Reports"
RAG_TEST_DGML_DATA_DIR = TEST_DATA_DIR / "dgml_samples" / RAG_TEST_DGML_DOCSET_NAME
EXAMPLES_PATH = TEST_DATA_DIR / "examples"

CHARTERS_SUMMARY_DATA_FILE = TEST_DATA_DIR / "xlsx/Charters Summary.xlsx"
CHARTERS_SUMMARY_TABLE_NAME = "Corporate Charters"

SAAS_CONTRACTS_DATA_FILE = TEST_DATA_DIR / "xlsx/SaaS Contracts Report.xlsx"
SAAS_CONTRACTS_TABLE_NAME = "SaaS Contracts"

FINANCIAL_SAMPLE_DATA_FILE = TEST_DATA_DIR / "xlsx/Financial Sample.xlsx"
FINANCIAL_SAMPLE_TABLE_NAME = "Financial Data"

AVIATION_INCIDENTS_DATA_FILE = TEST_DATA_DIR / "xlsx/Aviation Incidents Report.xlsx"
AVIATION_INCIDENTS_TABLE_NAME = "Aviation Incidents Report"

DEMO_MSA_SERVICES_DATA_FILE = TEST_DATA_DIR / "xlsx/Report Services_preview.xlsx"
DEMO_MSA_SERVICES_TABLE_NAME = "Service Agreements Summary"

GENERAL_KNOWLEDGE_QUESTION = "Who formulated the theory of special relativity?"
GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS = ["einstein"]

GENERAL_KNOWLEDGE_CHAT_HISTORY = [
    (
        "Who formulated the theory of special relativity?",
        "Albert Einstein",
    ),
    (
        "What about the theory of natural selection?",
        "The theory of natural selection was formulated by Charles Darwin and Alfred Russel Wallace.",
    ),
]
GENERAL_KNOWLEDGE_QUESTION_WITH_HISTORY = "When were they all born?"
GENERAL_KNOWLEDGE_ANSWER_WITH_HISTORY_FRAGMENTS = ["1879", "1809", "1823"]


RAG_QUESTION = "What is the accident number for the incident in madill, oklahoma?"
RAG_ANSWER_FRAGMENTS = ["DFW08CA044"]

RAG_CHAT_HISTORY = [
    (
        "What is the county seat of Marshall county, OK?",
        "Madill is a city in and the county seat of Marshall County, Oklahoma, United States.",
    ),
    (
        "Do you know who it was named after?",
        "It was named in honor of George Alexander Madill, an attorney for the St. Louis-San Francisco Railway.",
    ),
]
RAG_QUESTION_WITH_HISTORY = "List the accident numbers for any aviation incidents that happened at this location"
RAG_ANSWER_WITH_HISTORY_FRAGMENTS = ["DFW08CA044"]


def is_core_tests_only_mode() -> bool:
    core_tests_env_var = os.environ.get("DOCUGAMI_ONLY_CORE_TESTS")
    if not core_tests_env_var:
        return False
    else:
        if isinstance(core_tests_env_var, bool):
            return core_tests_env_var
        else:
            return str(core_tests_env_var).lower() == "true"


def verify_value(
    value: Any,
    match_fragment_str_options: list[str] = [],
    empty_ok: bool = False,
) -> None:
    value = str(value)
    assert value
    if match_fragment_str_options:
        output_match = False
        for fragment in match_fragment_str_options:
            output_match = output_match or fragment.lower() in value.lower()

        assert (
            output_match
        ), f"{value} does not contain one of the expected output substrings {match_fragment_str_options}"

    # Check guardrails and warn if any violations detected based on string checks
    for banned_word in ["sql", "context"]:
        if banned_word.lower() in value.lower():
            warnings.warn(
                UserWarning(f"Output contains banned word {banned_word}: {value}")
            )


def verify_traced_response(
    response: TracedResponse[Any],
    match_fragment_str_options: list[str] = [],
    empty_ok: bool = False,
) -> None:
    assert response.run_id
    if empty_ok and not response.value:
        return

    return verify_value(response.value, match_fragment_str_options, empty_ok)


def build_test_common_tools(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> list[BaseTool]:
    """
    Builds common tools for test purposes
    """
    return get_generic_tools(
        llm=llm,
        embeddings=embeddings,
        answer_examples_file=EXAMPLES_PATH / "test_answer_examples.yaml",
    )


def build_test_query_tool(llm: BaseLanguageModel, embeddings: Embeddings) -> BaseTool:
    """
    Builds a query tool over a test database
    """
    xlsx = AVIATION_INCIDENTS_DATA_FILE
    name = AVIATION_INCIDENTS_TABLE_NAME
    db = connect_to_excel(xlsx, name)
    description = report_details_to_report_query_tool_description(
        name, get_table_info(db)
    )
    tool = get_retrieval_tool_for_report(
        local_xlsx_path=xlsx,
        report_name=name,
        retrieval_tool_function_name=report_name_to_report_query_tool_function_name(
            name
        ),
        retrieval_tool_description=description,
        sql_llm=llm,
        embeddings=embeddings,
        sql_fixup_examples_file=EXAMPLES_PATH / "test_sql_fixup_examples.yaml",
        sql_examples_file=EXAMPLES_PATH / "test_sql_examples.yaml",
    )
    if not tool:
        raise Exception("Could not create test query tool")

    return tool


def build_test_retrieval_artifacts(
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    data_dir: Path = RAG_TEST_DGML_DATA_DIR,
    data_files_glob: str = "*.xml",
) -> tuple[
    VectorStore,
    dict[str, Document],
    FusedRetrieverKeyValueFetchCallback,
    FusedRetrieverKeyValueFetchCallback,
]:
    """
    Builds a vector store pre-populated with chunks from test documents
    using the given embeddings, document summaries, and callbacks used for retrieval tests.
    """
    test_dgml_files = list(data_dir.rglob(data_files_glob))
    loader = DocugamiLoader(file_paths=test_dgml_files, parent_hierarchy_levels=2)
    chunks = loader.load()
    full_docs_by_id, parent_chunks_by_id = build_doc_maps_from_chunks(chunks)

    full_doc_summaries_by_id = build_full_doc_summary_mappings(
        docs_by_id=full_docs_by_id,
        llm=llm,
        embeddings=embeddings,
        include_xml_tags=False,
        summarize_document_examples_file=EXAMPLES_PATH
        / "test_summarize_document_examples.yaml",
    )
    chunk_summaries_by_id = build_chunk_summary_mappings(
        docs_by_id=parent_chunks_by_id,
        llm=llm,
        embeddings=embeddings,
        include_xml_tags=False,
        summarize_chunk_examples_file=EXAMPLES_PATH
        / "test_summarize_chunk_examples.yaml",
    )

    vector_store = FAISS.from_documents(
        documents=list(
            chunk_summaries_by_id.values()
        ),  # embed chunk summaries for small to big retrieval
        embedding=embeddings,
    )

    def _fetch_parent_doc_callback(key: str) -> Optional[str]:
        parent_chunk_doc = parent_chunks_by_id.get(key)
        if parent_chunk_doc:
            return parent_chunk_doc.page_content
        return None

    def _fetch_full_doc_summary_callback(key: str) -> Optional[str]:
        full_doc_summary_doc = full_doc_summaries_by_id.get(key)
        if full_doc_summary_doc:
            return full_doc_summary_doc.page_content
        return None

    return (
        vector_store,
        full_doc_summaries_by_id,
        _fetch_parent_doc_callback,
        _fetch_full_doc_summary_callback,
    )


def build_test_fused_retriever(
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    re_ranker: BaseRanker,
    data_dir: Path = RAG_TEST_DGML_DATA_DIR,
    data_files_glob: str = "*.xml",
) -> FusedSummaryRetriever:
    """
    Builds a vector store pre-populated with chunks from test documents
    using the given embeddings, and returns a retriever off it.
    """
    (
        vector_store,
        _,
        _fetch_parent_doc_callback,
        _fetch_full_doc_summary_callback,
    ) = build_test_retrieval_artifacts(llm, embeddings, data_dir, data_files_glob)

    return FusedSummaryRetriever(
        vectorstore=vector_store,
        re_ranker=re_ranker,
        fetch_parent_doc_callback=_fetch_parent_doc_callback,
        fetch_full_doc_summary_callback=_fetch_full_doc_summary_callback,
        retriever_k=DEFAULT_RETRIEVER_K,
        search_type=SearchType.mmr,
    )


def build_test_retrieval_tool(
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    re_ranker: BaseRanker,
    data_dir: Path = RAG_TEST_DGML_DATA_DIR,
    data_files_glob: str = "*.xml",
) -> BaseTool:
    """
    Builds a vector store pre-populated with chunks from test documents
    using the given embeddings, and returns a retriever tool off it.
    """
    (
        vector_store,
        full_doc_summaries_by_id,
        _fetch_parent_doc_callback,
        _fetch_full_doc_summary_callback,
    ) = build_test_retrieval_artifacts(llm, embeddings, data_dir, data_files_glob)

    retrieval_tool_description = summaries_to_direct_retriever_tool_description(
        name=RAG_TEST_DGML_DOCSET_NAME,
        summaries=random.sample(
            list(full_doc_summaries_by_id.values()),
            min(len(full_doc_summaries_by_id), 3),
        ),  # give 3 randomly selected summaries summaries
        llm=llm,
        embeddings=embeddings,
        max_sample_documents_cutoff_length=MAX_FULL_DOCUMENT_TEXT_LENGTH,
        describe_document_set_examples_file=EXAMPLES_PATH
        / "test_describe_document_set_examples.yaml",
    )
    tool = get_retrieval_tool_for_docset(
        chunk_vectorstore=vector_store,
        re_ranker=re_ranker,
        retrieval_tool_function_name=docset_name_to_direct_retriever_tool_function_name(
            RAG_TEST_DGML_DOCSET_NAME
        ),
        retrieval_tool_description=retrieval_tool_description,
        fetch_parent_doc_callback=_fetch_parent_doc_callback,
        fetch_full_doc_summary_callback=_fetch_full_doc_summary_callback,
        retrieval_k=DEFAULT_RETRIEVER_K,
    )
    if not tool:
        raise Exception("Failed to create retrieval tool")

    return tool
