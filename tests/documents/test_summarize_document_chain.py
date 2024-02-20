import os
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.documents import SummarizeDocumentChain
from tests.conftest import TEST_DATA_DIR, verify_chain_response
from tests.testdata.dgml_samples.dgml_samples_test_data import (
    DG_SAMPLE_TEST_DATA,
    DGSamplesTestData,
)


@pytest.fixture()
def fireworksai_mixtral_summarize_document_chain(
    fireworksai_mixtral: BaseLanguageModel, huggingface_minilm: Embeddings
) -> SummarizeDocumentChain:
    """
    Fireworks AI chain to do document summarize using mixtral.
    """
    chain = SummarizeDocumentChain(
        llm=fireworksai_mixtral, embeddings=huggingface_minilm
    )
    chain.load_examples(
        TEST_DATA_DIR / "examples/test_summarize_document_examples.yaml"
    )
    return chain


@pytest.fixture()
def openai_gpt35_summarize_document_chain(
    openai_gpt35: BaseLanguageModel, openai_ada: Embeddings
) -> SummarizeDocumentChain:
    """
    OpenAI chain to do document summarize using GPT 3.5.
    """
    chain = SummarizeDocumentChain(llm=openai_gpt35, embeddings=openai_ada)
    chain.load_examples(
        TEST_DATA_DIR / "examples/test_summarize_document_examples.yaml"
    )
    return chain


def _runtest_serial(
    chain: SummarizeDocumentChain,
    test_data: DGSamplesTestData,
) -> None:
    data_dir: Path = TEST_DATA_DIR / "dgml_samples" / test_data.test_data_dir

    for md_file in data_dir.rglob("*.md"):
        # Read and process the contents of each file
        with open(md_file, "r", encoding="utf-8") as file:
            contents = file.read()
            summary = chain.run(contents)
            verify_chain_response(summary)
            assert summary
            assert len(summary) < len(contents)


def _runtest_batched(
    chain: SummarizeDocumentChain,
    test_data: DGSamplesTestData,
) -> None:
    data_dir: Path = TEST_DATA_DIR / "dgml_samples" / test_data.test_data_dir

    all_contents = []
    for md_file in data_dir.rglob("*.md"):
        # Read and process the contents of each file
        with open(md_file, "r", encoding="utf-8") as file:
            contents = file.read()
            all_contents.append(contents)

    all_summaries = chain.run_batch([(c, "text") for c in all_contents])
    assert len(all_summaries) == len(all_contents)

    for idx in range(len(all_contents)):
        summary = all_summaries[idx]
        verify_chain_response(summary)
        assert summary
        assert len(summary) < len(all_contents[idx])


@pytest.mark.parametrize("test_data", DG_SAMPLE_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_summarize_document(
    fireworksai_mixtral_summarize_document_chain: SummarizeDocumentChain,
    test_data: DGSamplesTestData,
) -> None:
    _runtest_batched(fireworksai_mixtral_summarize_document_chain, test_data)


@pytest.mark.parametrize("test_data", DG_SAMPLE_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_summarize_document(
    openai_gpt35_summarize_document_chain: SummarizeDocumentChain,
    test_data: DGSamplesTestData,
) -> None:
    _runtest_serial(openai_gpt35_summarize_document_chain, test_data)
