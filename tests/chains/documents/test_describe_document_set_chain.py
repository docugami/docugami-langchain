import os
import random
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains.documents import (
    DescribeDocumentSetChain,
    SummarizeDocumentChain,
)
from docugami_langchain.config import DEFAULT_EXAMPLES_PER_PROMPT
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


@pytest.fixture()
def fireworksai_mixtral_describe_document_set_chain(
    fireworksai_mixtral: BaseLanguageModel, huggingface_minilm: Embeddings
) -> DescribeDocumentSetChain:
    """
    Fireworks AI chain to describe document sets using mixtral.
    """
    chain = DescribeDocumentSetChain(
        llm=fireworksai_mixtral, embeddings=huggingface_minilm
    )
    chain.load_examples(
        TEST_DATA_DIR / "examples/test_describe_document_set_examples.yaml"
    )
    return chain


@pytest.fixture()
def openai_gpt35_describe_document_set_chain(
    openai_gpt35: BaseLanguageModel, openai_ada: Embeddings
) -> DescribeDocumentSetChain:
    """
    OpenAI chain to describe document sets using GPT 3.5.
    """
    chain = DescribeDocumentSetChain(llm=openai_gpt35, embeddings=openai_ada)
    chain.load_examples(
        TEST_DATA_DIR / "examples/test_describe_document_set_examples.yaml"
    )
    return chain


def _runtest(
    summarize_chain: SummarizeDocumentChain,
    describe_document_sets_chain: DescribeDocumentSetChain,
    test_data: DGSamplesTestData,
) -> None:
    data_dir: Path = TEST_DATA_DIR / "dgml_samples" / test_data.test_data_dir

    all_contents = []
    for md_file in data_dir.rglob("*.md"):
        # Read and process the contents of each file
        with open(md_file, "r", encoding="utf-8") as file:
            contents = file.read()
            all_contents.append(contents)

    all_summaries = summarize_chain.run_batch([(c, "text") for c in all_contents])
    selected_summaries = random.sample(
        all_summaries, min(len(all_summaries), DEFAULT_EXAMPLES_PER_PROMPT)
    )
    selected_summary_docs = [Document(s) for s in selected_summaries]
    description = describe_document_sets_chain.run(
        summaries=selected_summary_docs,
        docset_name=test_data.test_data_dir,
    )
    verify_chain_response(description, empty_ok=False)


@pytest.mark.parametrize("test_data", DG_SAMPLE_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_describe_document_set(
    fireworksai_mixtral_summarize_document_chain: SummarizeDocumentChain,
    fireworksai_mixtral_describe_document_set_chain: DescribeDocumentSetChain,
    test_data: DGSamplesTestData,
) -> None:
    _runtest(
        fireworksai_mixtral_summarize_document_chain,
        fireworksai_mixtral_describe_document_set_chain,
        test_data,
    )


@pytest.mark.parametrize("test_data", DG_SAMPLE_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_describe_document_set(
    openai_gpt35_summarize_document_chain: SummarizeDocumentChain,
    openai_gpt35_describe_document_set_chain: DescribeDocumentSetChain,
    test_data: DGSamplesTestData,
) -> None:
    _runtest(
        openai_gpt35_summarize_document_chain,
        openai_gpt35_describe_document_set_chain,
        test_data,
    )
