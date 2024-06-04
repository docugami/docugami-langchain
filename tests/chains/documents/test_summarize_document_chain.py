import os
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.documents import SummarizeDocumentChain
from tests.common import TEST_DATA_DIR, verify_traced_response
from tests.testdata.docsets.docset_test_data import (
    DOCSET_TEST_DATA,
    DocsetTestData,
)


def init_chain(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> SummarizeDocumentChain:
    chain = SummarizeDocumentChain(llm=llm, embeddings=embeddings)
    chain.load_examples(
        TEST_DATA_DIR / "examples/test_summarize_document_examples.yaml"
    )
    return chain


def _runtest_serial(
    chain: SummarizeDocumentChain,
    test_data: DocsetTestData,
) -> None:
    data_dir: Path = TEST_DATA_DIR / "docsets" / test_data.name

    for md_file in data_dir.rglob("*.md"):
        # Read and process the contents of each file
        with open(md_file, "r", encoding="utf-8") as file:
            contents = file.read()
            summary = chain.run(contents)
            verify_traced_response(summary)
            assert len(summary.value) < len(contents)


def _runtest_batched(
    chain: SummarizeDocumentChain,
    test_data: DocsetTestData,
) -> None:
    data_dir: Path = TEST_DATA_DIR / "docsets" / test_data.name

    all_contents: list[str] = []
    for md_file in data_dir.rglob("*.md"):
        # Read and process the contents of each file
        with open(md_file, "r", encoding="utf-8") as file:
            contents = file.read()
            all_contents.append(contents)

    all_summaries: list[str] = chain.run_batch([(c, "text") for c in all_contents])  # type: ignore
    for summary in all_summaries:
        if isinstance(summary, Exception):
            raise summary

    assert len(all_summaries) == len(all_contents)

    for idx in range(len(all_contents)):
        summary = all_summaries[idx]
        assert len(summary) < len(all_contents[idx])


@pytest.mark.parametrize("test_data", DOCSET_TEST_DATA)
@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks API token not set"
)
def test_fireworksai_llama3_summarize_document(
    test_data: DocsetTestData,
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    chain = init_chain(fireworksai_llama3, huggingface_minilm)
    _runtest_batched(chain, test_data)


@pytest.mark.parametrize("test_data", DOCSET_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_gpt4_summarize_document(
    test_data: DocsetTestData, openai_gpt4: BaseLanguageModel, openai_ada: Embeddings
) -> None:
    chain = init_chain(openai_gpt4, openai_ada)
    _runtest_serial(chain, test_data)
