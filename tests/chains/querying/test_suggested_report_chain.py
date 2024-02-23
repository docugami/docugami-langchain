import os
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains import SuggestedReportChain, SummarizeDocumentChain
from tests.common import TEST_DATA_DIR
from tests.testdata.dgml_samples.dgml_samples_test_data import (
    DG_SAMPLE_TEST_DATA,
    DGSamplesTestData,
)

SUGGESTED_REPORT_EXAMPLES_FILE = (
    TEST_DATA_DIR / "examples/test_suggested_report_examples.yaml"
)

SUMMARIZE_DOCUMENT_EXAMPLES_FILE = (
    TEST_DATA_DIR / "examples/test_summarize_document_examples.yaml"
)


@pytest.fixture()
def fireworksai_mixtral_suggested_report_chain(
    fireworksai_mixtral: BaseLanguageModel, huggingface_minilm: Embeddings
) -> SuggestedReportChain:
    """
    Fireworks AI chain to generate suggested report columns using mixtral.
    """
    chain = SuggestedReportChain(llm=fireworksai_mixtral, embeddings=huggingface_minilm)
    chain.load_examples(SUGGESTED_REPORT_EXAMPLES_FILE)
    return chain


@pytest.fixture()
def openai_gpt35_suggested_report_chain(
    openai_gpt35: BaseLanguageModel, openai_ada: Embeddings
) -> SuggestedReportChain:
    """
    OpenAI chain to generate suggested report columns using GPT 3.5.
    """
    chain = SuggestedReportChain(llm=openai_gpt35, embeddings=openai_ada)
    chain.load_examples(SUGGESTED_REPORT_EXAMPLES_FILE)
    return chain


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
    return chain


@pytest.fixture()
def openai_gpt35_summarize_document_chain(
    openai_gpt35: BaseLanguageModel, openai_ada: Embeddings
) -> SummarizeDocumentChain:
    """
    OpenAI chain to do document summarize using GPT 3.5.
    """
    chain = SummarizeDocumentChain(llm=openai_gpt35, embeddings=openai_ada)
    return chain


def _runtest(
    suggested_report_chain: SuggestedReportChain,
    summarize_document_chain: SummarizeDocumentChain,
    test_data: DGSamplesTestData,
) -> None:
    data_dir: Path = TEST_DATA_DIR / "dgml_samples" / test_data.test_data_dir

    test_data_full_docs: list[Document] = []
    for md_file in data_dir.rglob("*.md"):
        # Read and process the contents of each file
        with open(md_file, "r", encoding="utf-8") as file:
            contents = file.read()
            test_data_full_docs.append(
                Document(
                    page_content=contents,
                    metadata={
                        "source": str(md_file.absolute()),
                    },
                )
            )

    page_contents = [(d.page_content, "text") for d in test_data_full_docs]
    summaries = summarize_document_chain.run_batch(page_contents)  # type: ignore
    assert len(summaries) == len(test_data_full_docs)

    test_data_summarized_docs = []
    for doc, summary in zip(test_data_full_docs, summaries):
        doc.page_content = summary
        test_data_summarized_docs.append(doc)

    answers = suggested_report_chain.run(test_data_summarized_docs)
    assert answers
    assert len(answers) > 0


@pytest.mark.parametrize("test_data", DG_SAMPLE_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_suggested_report(
    fireworksai_mixtral_suggested_report_chain: SuggestedReportChain,
    fireworksai_mixtral_summarize_document_chain: SummarizeDocumentChain,
    test_data: DGSamplesTestData,
) -> None:
    _runtest(
        fireworksai_mixtral_suggested_report_chain,
        fireworksai_mixtral_summarize_document_chain,
        test_data,
    )


@pytest.mark.parametrize("test_data", DG_SAMPLE_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_suggested_report(
    openai_gpt35_suggested_report_chain: SuggestedReportChain,
    openai_gpt35_summarize_document_chain: SummarizeDocumentChain,
    test_data: DGSamplesTestData,
) -> None:
    _runtest(
        openai_gpt35_suggested_report_chain,
        openai_gpt35_summarize_document_chain,
        test_data,
    )
