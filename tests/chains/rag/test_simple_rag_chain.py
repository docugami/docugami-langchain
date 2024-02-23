import os
from typing import Optional

import pytest
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever

from docugami_langchain.chains import SimpleRAGChain
from docugami_langchain.config import RETRIEVER_K
from docugami_langchain.document_loaders.docugami import DocugamiLoader
from docugami_langchain.retrievers.fused_summary import (
    FusedSummaryRetriever,
    SearchType,
)
from docugami_langchain.retrievers.mappings import (
    build_chunk_summary_mappings,
    build_doc_maps_from_chunks,
    build_full_doc_summary_mappings,
)
from tests.common import EXAMPLES_PATH, RAG_TEST_DGML_DATA_DIR, verify_chain_response

TEST_QUESTION = "What is the accident number for the incident in madill, oklahoma?"
TEST_ANSWER_OPTIONS = ["DFW08CA044"]


def build_retriever(llm: BaseLanguageModel, embeddings: Embeddings) -> BaseRetriever:
    """
    Builds a vector store pre-populated with chunks from test documents
    using the given embeddings, and returns a retriever off it.
    """
    test_dgml_files = list(RAG_TEST_DGML_DATA_DIR.rglob("*.xml"))
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

    return FusedSummaryRetriever(
        vectorstore=vector_store,
        fetch_parent_doc_callback=_fetch_parent_doc_callback,
        fetch_full_doc_summary_callback=_fetch_full_doc_summary_callback,
        search_kwargs={"k": RETRIEVER_K},
        search_type=SearchType.mmr,
    )


@pytest.fixture()
def huggingface_retriever(
    fireworksai_mixtral: BaseLanguageModel, huggingface_minilm: Embeddings
) -> BaseRetriever:
    return build_retriever(llm=fireworksai_mixtral, embeddings=huggingface_minilm)


@pytest.fixture()
def openai_retriever(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> BaseRetriever:
    return build_retriever(llm=openai_gpt35, embeddings=openai_ada)


@pytest.fixture()
def fireworksai_mixtral_simple_rag_chain(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    huggingface_retriever: BaseRetriever,
) -> SimpleRAGChain:
    """
    Fireworks AI chain to do simple RAG queries using mixtral.
    """
    chain = SimpleRAGChain(
        llm=fireworksai_mixtral,
        embeddings=huggingface_minilm,
        retriever=huggingface_retriever,
    )
    return chain


@pytest.fixture()
def openai_gpt35_simple_rag_chain(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
    openai_retriever: BaseRetriever,
) -> SimpleRAGChain:
    """
    OpenAI chain to do simple RAG queries using GPT 3.5.
    """
    chain = SimpleRAGChain(
        llm=openai_gpt35,
        embeddings=openai_ada,
        retriever=openai_retriever,
    )
    return chain


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_simple_rag(
    fireworksai_mixtral_simple_rag_chain: SimpleRAGChain,
) -> None:
    answer = fireworksai_mixtral_simple_rag_chain.run(TEST_QUESTION)
    verify_chain_response(answer, TEST_ANSWER_OPTIONS)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_simple_rag(openai_gpt35_simple_rag_chain: SimpleRAGChain) -> None:
    answer = openai_gpt35_simple_rag_chain.run(TEST_QUESTION)
    verify_chain_response(answer, TEST_ANSWER_OPTIONS)
