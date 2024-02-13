import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_docugami.chains.documents import SummarizeDocumentChain
from tests.conftest import TEST_DATA_DIR, verify_chain_response
from tests.testdata.dg_samples.dg_samples_test_data import (
    DG_SAMPLE_TEST_DATA,
    DGSamplesTestData,
)


@pytest.fixture()
def fireworksai_mixtral_summarize_document_chain(
    fireworksai_mixtral: BaseChatModel, huggingface_minilm: Embeddings
) -> SummarizeDocumentChain:
    """
    Fireworks AI chain to do document summarize using mixtral.
    """
    return SummarizeDocumentChain(
        llm=fireworksai_mixtral, embeddings=huggingface_minilm
    )


@pytest.fixture()
def openai_gpt35_summarize_document_chain(
    openai_gpt35: BaseChatModel, openai_ada: Embeddings
) -> SummarizeDocumentChain:
    """
    OpenAI chain to do document summarize using GPT 3.5.
    """
    return SummarizeDocumentChain(llm=openai_gpt35, embeddings=openai_ada)


def _runtest(
    chain: SummarizeDocumentChain,
    test_data: DGSamplesTestData,
) -> Any:
    data_dir: Path = TEST_DATA_DIR / "dg_samples" / test_data.test_data_dir

    for md_file in data_dir.rglob("*.md"):
        # Read and process the contents of each file
        with open(md_file, "r", encoding="utf-8") as file:
            contents = file.read()
            summary = chain.run(contents)
            verify_chain_response(summary)
            assert summary
            assert len(summary) < len(contents)


@pytest.mark.parametrize("test_data", DG_SAMPLE_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_summarize_document(
    fireworksai_mixtral_summarize_document_chain: SummarizeDocumentChain,
    test_data: DGSamplesTestData,
) -> Any:
    _runtest(fireworksai_mixtral_summarize_document_chain, test_data)


@pytest.mark.parametrize("test_data", DG_SAMPLE_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_summarize_document(
    openai_gpt35_summarize_document_chain: SummarizeDocumentChain,
    test_data: DGSamplesTestData,
) -> Any:
    _runtest(openai_gpt35_summarize_document_chain, test_data)
