import os
import random
import warnings
from pathlib import Path
from typing import Any, Optional

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.config import MAX_FULL_DOCUMENT_TEXT_LENGTH, RETRIEVER_K
from docugami_langchain.document_loaders.docugami import DocugamiLoader
from docugami_langchain.retrievers.mappings import (
    build_chunk_summary_mappings,
    build_doc_maps_from_chunks,
    build_full_doc_summary_mappings,
)
from docugami_langchain.tools.retrieval import (
    docset_name_to_direct_retriever_tool_function_name,
    get_retrieval_tool_for_docset,
    summaries_to_direct_retriever_tool_description,
)

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

DEMO_MSA_SERVICES_DATA_FILE = TEST_DATA_DIR / "xlsx/Report Services_preview.xlsx"
DEMO_MSA_SERVICES_TABLE_NAME = "Service Agreements Summary"

GENERAL_KNOWLEDGE_QUESTION = "Who formulated the theory of special relativity?"
GENERAL_KNOWLEDGE_ANSWER_FRAGMENTS = ["einstein"]


def is_core_tests_only_mode() -> bool:
    core_tests_env_var = os.environ.get("DOCUGAMI_ONLY_CORE_TESTS")
    if not core_tests_env_var:
        return False
    else:
        if isinstance(core_tests_env_var, bool):
            return core_tests_env_var
        else:
            return str(core_tests_env_var).lower() == "true"


def verify_response(
    response: TracedResponse[Any],
    match_fragment_str_options: list[str] = [],
    empty_ok: bool = False,
) -> None:
    assert response.run_id
    if empty_ok and not response.value:
        return

    value = str(response.value)
    assert value
    if match_fragment_str_options:
        output_match = False
        for fragment in match_fragment_str_options:
            output_match = output_match or fragment.lower() in value.lower()

        assert (
            output_match
        ), f"{response} does not contain one of the expected output substrings {match_fragment_str_options}"

    # Check guardrails and warn if any violations detected based on string checks
    for banned_word in ["sql", "context"]:
        if banned_word.lower() in value.lower():
            warnings.warn(
                UserWarning(f"Output contains banned word {banned_word}: {value}")
            )


def build_test_query_tool(llm: BaseLanguageModel, embeddings: Embeddings) -> BaseTool:
    """
    Builds a query tool over a test database
    """
    raise Exception()


def build_test_search_tool(
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    data_dir: Path = RAG_TEST_DGML_DATA_DIR,
    data_files_glob: str = "*.xml",
) -> BaseTool:
    """
    Builds a vector store pre-populated with chunks from test documents
    using the given embeddings, and returns a retriever tool off it.
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
        retrieval_tool_function_name=docset_name_to_direct_retriever_tool_function_name(
            RAG_TEST_DGML_DOCSET_NAME
        ),
        retrieval_tool_description=retrieval_tool_description,
        fetch_parent_doc_callback=_fetch_parent_doc_callback,
        fetch_full_doc_summary_callback=_fetch_full_doc_summary_callback,
        retrieval_k=RETRIEVER_K,
    )
    if not tool:
        raise Exception("Failed to create retrieval tool")

    return tool
