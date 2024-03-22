import os
from pathlib import Path

import pytest
from langchain_community.cache import SQLiteCache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_fireworks.chat_models import ChatFireworks
from langchain_fireworks.llms import Fireworks
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rerankers import Reranker
from rerankers.models.ranker import BaseRanker

from tests.common import (
    build_test_common_tools,
    build_test_query_tool,
    build_test_retrieval_tool,
)

# Turn on caching
LOCAL_LLM_CACHE_DB_FILE = os.environ.get(
    "LOCAL_LLM_CACHE", "/tmp/docugami/.langchain.db"
)
os.makedirs(Path(LOCAL_LLM_CACHE_DB_FILE).parent, exist_ok=True)
set_llm_cache(SQLiteCache(database_path=LOCAL_LLM_CACHE_DB_FILE))


@pytest.fixture()
def fireworksai_mistral_7b() -> BaseLanguageModel:
    """
    Mistral_7b model hosted on fireworksai.
    """
    return Fireworks(
        model="accounts/fireworks/models/mistral-7b",
        cache=True,
        temperature=0,
        max_tokens=2 * 1024,  # includes input and output tokens
        model_kwargs={
            "context_length_exceeded_behavior": "truncate",
        },
    )


@pytest.fixture()
def fireworksai_mixtral() -> BaseLanguageModel:
    """
    Mixtral8x7b model hosted on fireworksai.
    """
    return ChatFireworks(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        streaming=True,
        cache=True,
        temperature=0,
        max_tokens=32 * 1024,  # includes input and output tokens
        model_kwargs={
            "context_length_exceeded_behavior": "truncate",
        },
    )


@pytest.fixture()
def huggingface_minilm() -> Embeddings:
    """
    MiniLM-L6-v2 embeddings running locally using huggingface.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


@pytest.fixture()
def mxbai_re_rank() -> BaseRanker:
    return Reranker("mixedbread-ai/mxbai-rerank-base-v1", verbose=0)


@pytest.fixture()
def openai_gpt35() -> BaseLanguageModel:
    """
    GPT 3.5 model by OpenAI.
    """
    return ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        streaming=True,
        cache=True,
        temperature=0,
        max_tokens=2 * 1024,  # only output tokens
    )


@pytest.fixture()
def openai_ada() -> Embeddings:
    """
    Ada embeddings by OpenAI.
    """
    return OpenAIEmbeddings(model="text-embedding-ada-002", client=None)


@pytest.fixture()
def openai_gpt35_re_rank() -> BaseRanker:
    return Reranker("rankgpt3", api_key=os.environ.get("OPENAI_API_KEY"), verbose=0)


@pytest.fixture()
def huggingface_common_tools(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> list[BaseTool]:
    return build_test_common_tools(
        llm=fireworksai_mixtral,
        embeddings=huggingface_minilm,
    )


@pytest.fixture()
def openai_common_tools(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> list[BaseTool]:
    return build_test_common_tools(
        llm=openai_gpt35,
        embeddings=openai_ada,
    )


@pytest.fixture()
def huggingface_query_tool(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> BaseTool:
    return build_test_query_tool(
        llm=fireworksai_mixtral,
        embeddings=huggingface_minilm,
    )


@pytest.fixture()
def openai_query_tool(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
) -> BaseTool:
    return build_test_query_tool(
        llm=openai_gpt35,
        embeddings=openai_ada,
    )


@pytest.fixture()
def huggingface_retrieval_tool(
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
    mxbai_re_rank: BaseRanker,
) -> BaseTool:
    return build_test_retrieval_tool(
        llm=fireworksai_mixtral,
        embeddings=huggingface_minilm,
        re_ranker=mxbai_re_rank,
    )


@pytest.fixture()
def openai_retrieval_tool(
    openai_gpt35: BaseLanguageModel,
    openai_ada: Embeddings,
    openai_gpt35_re_rank: BaseRanker,
) -> BaseTool:
    return build_test_retrieval_tool(
        llm=openai_gpt35,
        embeddings=openai_ada,
        re_ranker=openai_gpt35_re_rank,
    )
