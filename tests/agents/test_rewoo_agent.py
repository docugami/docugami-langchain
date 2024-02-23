import os
import random
from typing import Optional

import pytest
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from docugami_langchain.agents import ReWOOAgent
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
from tests.common import (
    EXAMPLES_PATH,
    RAG_TEST_DGML_DATA_DIR,
    RAG_TEST_DGML_DOCSET_NAME,
    verify_chain_response,
)

TEST_QUESTION = "What is the accident number for the incident in madill, oklahoma?"
TEST_ANSWER_OPTIONS = ["DFW08CA044"]


def build_retrieval_tool(llm: BaseLanguageModel, embeddings: Embeddings) -> BaseTool:
    """
    Builds a vector store pre-populated with chunks from test documents
    using the given embeddings, and returns a retriever tool off it.
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


@pytest.fixture()
def huggingface_retrieval_tool(
    fireworksai_mixtral: BaseLanguageModel, huggingface_minilm: Embeddings
) -> BaseTool:
    return build_retrieval_tool(llm=fireworksai_mixtral, embeddings=huggingface_minilm)


@pytest.fixture()
def openai_retrieval_tool(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> BaseTool:
    return build_retrieval_tool(llm=openai_gpt35, embeddings=openai_ada)


@pytest.fixture()
def fireworksai_mixtral_rewoo_agent(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    huggingface_retrieval_tool: BaseTool,
) -> ReWOOAgent:
    """
    Fireworks AI ReWOO Agent using mixtral.
    """
    chain = ReWOOAgent(
        llm=fireworksai_mixtral,
        embeddings=huggingface_minilm,
        tools=[huggingface_retrieval_tool],
    )
    return chain


@pytest.fixture()
def openai_gpt35_rewoo_agent(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
    openai_retrieval_tool: BaseTool,
) -> ReWOOAgent:
    """
    OpenAI ReWOO Agent using GPT 3.5.
    """
    chain = ReWOOAgent(
        llm=openai_gpt35, embeddings=openai_ada, tools=[openai_retrieval_tool]
    )
    return chain


@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_rewoo(
    fireworksai_mixtral_rewoo_agent: ReWOOAgent,
) -> None:
    response = fireworksai_mixtral_rewoo_agent.run(TEST_QUESTION)
    result = response.get("result")
    assert result
    verify_chain_response(result, TEST_ANSWER_OPTIONS)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_rewoo(openai_gpt35_rewoo_agent: ReWOOAgent) -> None:

    # test general LLM response from agent
    response = openai_gpt35_rewoo_agent.run("Who formulated the theory of special relativity?")
    result = response.get("result")
    assert result
    verify_chain_response(result, ["einstein"])

    # test retrieval response from agent
    response = openai_gpt35_rewoo_agent.run(TEST_QUESTION)
    result = response.get("result")
    assert result
    verify_chain_response(result, TEST_ANSWER_OPTIONS)
