import warnings
from pathlib import Path
from typing import Any, Optional

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.rag.retrieval_grader_chain import RetrievalGraderChain
from docugami_langchain.config import DEFAULT_RETRIEVER_K
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

TEST_DATA_DIR = Path(__file__).parent / "testdata"
EXAMPLES_PATH = TEST_DATA_DIR / "examples"

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


def verify_output_list(
    values: list[str],
    match_fragment_str_options: list[str] = [],
    empty_ok: bool = False,
) -> None:
    if empty_ok and not values:
        return

    assert values
    if match_fragment_str_options:
        output_match = False
        for value in values:
            for fragment in match_fragment_str_options:
                output_match = output_match or fragment.lower() in value.lower()

            assert (
                output_match
            ), f"The output {value} does not contain one of the expected output substrings {match_fragment_str_options}"

            # Check guardrails and warn if any violations detected based on string checks
            for banned_word in ["sql", "context"]:
                if banned_word.lower() in value.lower():
                    warnings.warn(
                        UserWarning(f"Output contains banned word {banned_word}: {value}")
                )


def verify_output(
    value: str,
    match_fragment_str_options: list[str] = [],
    empty_ok: bool = False,
) -> None:
    if empty_ok and not value:
        return

    verify_output_list([value], match_fragment_str_options, empty_ok)


def verify_traced_response(
    response: TracedResponse[Any],
    match_fragment_str_options: list[str] = [],
    empty_ok: bool = False,
) -> None:
    assert response.run_id
    if empty_ok and not response.value:
        return

    return verify_output(str(response.value), match_fragment_str_options, empty_ok)


def build_test_retrieval_artifacts(
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    data_dir: Path,
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
    loader = DocugamiLoader(file_paths=test_dgml_files, parent_chunk_hierarchy_levels=2)
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
    data_dir: Path,
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

    grader_chain = RetrievalGraderChain(llm=llm, embeddings=embeddings)
    grader_chain.load_examples(
        EXAMPLES_PATH / "test_retrieval_grader_examples.yaml",
    )

    return FusedSummaryRetriever(
        vectorstore=vector_store,
        fetch_parent_doc_callback=_fetch_parent_doc_callback,
        fetch_full_doc_summary_callback=_fetch_full_doc_summary_callback,
        retriever_k=DEFAULT_RETRIEVER_K,
        grader_chain=grader_chain,
        search_type=SearchType.mmr,
    )
