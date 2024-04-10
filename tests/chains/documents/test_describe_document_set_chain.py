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
from tests.common import TEST_DATA_DIR, verify_traced_response
from tests.testdata.docsets.docset_test_data import (
    DOCSET_TEST_DATA,
    DocsetTestData,
)


def init_summarize_document_chain(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> SummarizeDocumentChain:
    chain = SummarizeDocumentChain(llm=llm, embeddings=embeddings)
    chain.load_examples(
        TEST_DATA_DIR / "examples/test_summarize_document_examples.yaml"
    )
    return chain


def init_describe_document_set_chain(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> DescribeDocumentSetChain:
    chain = DescribeDocumentSetChain(llm=llm, embeddings=embeddings)
    chain.load_examples(
        TEST_DATA_DIR / "examples/test_describe_document_set_examples.yaml"
    )
    return chain


def _runtest(
    summarize_document_chain: SummarizeDocumentChain,
    describe_document_set_chain: DescribeDocumentSetChain,
    test_data: DocsetTestData,
) -> None:
    data_dir: Path = TEST_DATA_DIR / "docsets" / test_data.name

    all_contents = []
    for md_file in data_dir.rglob("*.md"):
        # Read and process the contents of each file
        with open(md_file, "r", encoding="utf-8") as file:
            contents = file.read()
            all_contents.append(contents)

    all_summaries = summarize_document_chain.run_batch(
        [(c, "text") for c in all_contents]
    )
    selected_summaries = random.sample(
        all_summaries, min(len(all_summaries), DEFAULT_EXAMPLES_PER_PROMPT)
    )
    selected_summary_docs = [Document(s) for s in selected_summaries]
    description = describe_document_set_chain.run(
        summaries=selected_summary_docs,
        docset_name=test_data.name,
    )
    verify_traced_response(description, empty_ok=False)


@pytest.mark.parametrize("test_data", DOCSET_TEST_DATA)
@pytest.mark.skipif(
    "FIREWORKS_API_KEY" not in os.environ, reason="Fireworks API token not set"
)
def test_fireworksai_describe_document_set(
    test_data: DocsetTestData,
    fireworksai_mixtral: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:
    summarize_document_chain = init_summarize_document_chain(
        fireworksai_mixtral, huggingface_minilm
    )
    describe_document_set_chain = init_describe_document_set_chain(
        fireworksai_mixtral, huggingface_minilm
    )
    _runtest(
        summarize_document_chain,
        describe_document_set_chain,
        test_data,
    )


@pytest.mark.parametrize("test_data", DOCSET_TEST_DATA)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="OpenAI API token not set"
)
def test_openai_gpt4_describe_document_set(
    test_data: DocsetTestData,
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    summarize_document_chain = init_summarize_document_chain(openai_gpt4, openai_ada)
    describe_document_set_chain = init_describe_document_set_chain(
        openai_gpt4, openai_ada
    )
    _runtest(
        summarize_document_chain,
        describe_document_set_chain,
        test_data,
    )
