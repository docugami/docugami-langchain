import os

import pytest
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

from langchain_docugami.chains import SimpleRAGChain
from langchain_docugami.document_loaders.docugami import DocugamiLoader
from tests.conftest import TEST_DATA_DIR, is_core_tests_only_mode, verify_chain_response

TEST_QUESTION = "What is the accident number for the incident in madill, oklahoma?"
TEST_ANSWER_OPTIONS = ["DFW08CA044"]


RAG_TEST_DGML_DATA_DIR = TEST_DATA_DIR / "dgml_samples"
if is_core_tests_only_mode():
    # RAG over fewer files when in core tests mode (to speed things up)
    RAG_TEST_DGML_DATA_DIR = RAG_TEST_DGML_DATA_DIR / "NTSB Aviation Incident Reports"


def build_vectorstore(embeddings: Embeddings) -> VectorStore:
    """
    Builds a vector store pre-populated with chunks from test documents
    using the given embeddings.
    """
    test_dgml_files = list(RAG_TEST_DGML_DATA_DIR.rglob("*.xml"))
    loader = DocugamiLoader(file_paths=test_dgml_files)
    chunks = loader.load()
    return FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )


@pytest.fixture()
def huggingface_minilm_vectorstore(huggingface_minilm: Embeddings) -> VectorStore:
    return build_vectorstore(huggingface_minilm)


@pytest.fixture()
def openai_ada_vectorstore(openai_ada: Embeddings) -> VectorStore:
    return build_vectorstore(openai_ada)


@pytest.fixture()
def fireworksai_mixtral_simple_rag_chain(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    huggingface_minilm_vectorstore: VectorStore,
) -> SimpleRAGChain:
    """
    Fireworks AI chain to do simple RAG queries using mixtral.
    """
    chain = SimpleRAGChain(
        llm=fireworksai_mixtral,
        embeddings=huggingface_minilm,
        chunk_vectorstore=huggingface_minilm_vectorstore,
    )
    return chain


@pytest.fixture()
def openai_gpt35_simple_rag_chain(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
    openai_ada_vectorstore: VectorStore,
) -> SimpleRAGChain:
    """
    OpenAI chain to do simple RAG queries using GPT 3.5.
    """
    chain = SimpleRAGChain(
        llm=openai_gpt35,
        embeddings=openai_ada,
        chunk_vectorstore=openai_ada_vectorstore,
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
