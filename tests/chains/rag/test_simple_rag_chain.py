import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever

from docugami_langchain.chains import SimpleRAGChain
from tests.common import RAG_ANSWER_FRAGMENTS, RAG_QUESTION, build_test_fused_retriever, verify_traced_response


@pytest.fixture()
def huggingface_retriever(
    fireworksai_mixtral: BaseLanguageModel, huggingface_minilm: Embeddings
) -> BaseRetriever:
    return build_test_fused_retriever(
        llm=fireworksai_mixtral, embeddings=huggingface_minilm
    )


@pytest.fixture()
def openai_retriever(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> BaseRetriever:
    return build_test_fused_retriever(llm=openai_gpt35, embeddings=openai_ada)


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
    answer = fireworksai_mixtral_simple_rag_chain.run(RAG_QUESTION)
    verify_traced_response(answer, RAG_ANSWER_FRAGMENTS)


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_simple_rag(openai_gpt35_simple_rag_chain: SimpleRAGChain) -> None:
    answer = openai_gpt35_simple_rag_chain.run(RAG_QUESTION)
    verify_traced_response(answer, RAG_ANSWER_FRAGMENTS)
