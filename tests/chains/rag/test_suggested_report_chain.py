import os
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains import SuggestedReportChain, SummarizeDocumentChain
from tests.common import TEST_DATA_DIR
from tests.testdata.docsets.docset_test_data import (
    DOCSET_TEST_DATA,
    DocsetTestData,
)


def init_suggested_report_chain(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> SuggestedReportChain:
    chain = SuggestedReportChain(llm=llm, embeddings=embeddings)
    chain.load_examples(TEST_DATA_DIR / "examples/test_suggested_report_examples.yaml")
    return chain


def init_summarize_document_chain(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> SummarizeDocumentChain:
    chain = SummarizeDocumentChain(llm=llm, embeddings=embeddings)
    chain.load_examples(
        TEST_DATA_DIR / "examples/test_summarize_document_examples.yaml"
    )
    return chain


def _runtest(
    suggested_report_chain: SuggestedReportChain,
    summarize_document_chain: SummarizeDocumentChain,
    test_data: DocsetTestData,
) -> None:
    data_dir: Path = TEST_DATA_DIR / "docsets" / test_data.name

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
    summaries: list[str] = summarize_document_chain.run_batch(page_contents)  # type: ignore
    for summary in summaries:
        if isinstance(summary, Exception):
            raise summary

    assert len(summaries) == len(test_data_full_docs)

    test_data_summarized_docs = []
    for doc, summary in zip(test_data_full_docs, summaries):
        doc.page_content = summary
        test_data_summarized_docs.append(doc)

    suggestions = suggested_report_chain.run(test_data_summarized_docs)
    assert len(suggestions.value) > 0


@pytest.mark.parametrize("test_data", DOCSET_TEST_DATA)
@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks API token not set"
)
def test_fireworksai_llama3_suggested_report(
    test_data: DocsetTestData,
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    suggested_report_chain = init_suggested_report_chain(
        fireworksai_llama3, huggingface_minilm
    )
    summarize_document_chain = init_summarize_document_chain(
        fireworksai_llama3, huggingface_minilm
    )
    _runtest(
        suggested_report_chain,
        summarize_document_chain,
        test_data,
    )


@pytest.mark.parametrize("test_data", DOCSET_TEST_DATA)
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OpenAI API token not set"
)
def test_openai_gpt4_suggested_report(
    test_data: DocsetTestData, openai_gpt4: BaseLanguageModel, openai_ada: Embeddings
) -> None:
    suggested_report_chain = init_suggested_report_chain(openai_gpt4, openai_ada)
    summarize_document_chain = init_summarize_document_chain(openai_gpt4, openai_ada)
    _runtest(
        suggested_report_chain,
        summarize_document_chain,
        test_data,
    )
