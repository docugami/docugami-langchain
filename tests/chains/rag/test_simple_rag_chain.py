import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains import SimpleRAGChain
from docugami_langchain.chains.rag.standalone_question_chain import (
    StandaloneQuestionChain,
)
from tests.common import (
    TEST_DATA_DIR,
    build_test_fused_retriever,
    verify_traced_response,
)
from tests.testdata.docsets.docset_test_data import (
    DOCSET_TEST_DATA,
    DocsetTestData,
)


def init_simple_rag_chain(
    test_data: DocsetTestData, llm: BaseLanguageModel, embeddings: Embeddings
) -> SimpleRAGChain:
    retriever = build_test_fused_retriever(
        llm=llm,
        embeddings=embeddings,
        data_dir=TEST_DATA_DIR / "docsets" / test_data.name,
    )

    standalone_questions_chain = StandaloneQuestionChain(
        llm=llm,
        embeddings=embeddings,
    )
    standalone_questions_chain.load_examples(
        TEST_DATA_DIR / "examples/test_standalone_question_examples.yaml"
    )

    return SimpleRAGChain(
        llm=llm,
        embeddings=embeddings,
        retriever=retriever,
    )


@pytest.mark.parametrize("test_data", DOCSET_TEST_DATA)
@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"), reason="Fireworks API token not set"
)
def test_fireworksai_llama3_simple_rag(
    test_data: DocsetTestData,
    fireworksai_llama3: BaseLanguageModel,
    huggingface_minilm: Embeddings,
) -> None:

    fireworksai_llama3_simple_rag_chain = init_simple_rag_chain(
        test_data=test_data,
        llm=fireworksai_llama3,
        embeddings=huggingface_minilm,
    )

    for question in test_data.questions:
        if not question.requires_report and not question.chat_history:
            answer = fireworksai_llama3_simple_rag_chain.run(
                question=question.question
            )
            verify_traced_response(answer, question.acceptable_answer_fragments)


@pytest.mark.parametrize("test_data", DOCSET_TEST_DATA)
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OpenAI API token not set"
)
def test_openai_gpt4_simple_rag(
    test_data: DocsetTestData,
    openai_gpt4: BaseLanguageModel,
    openai_ada: Embeddings,
) -> None:
    openai_gpt4_simple_rag_chain = init_simple_rag_chain(
        test_data=test_data,
        llm=openai_gpt4,
        embeddings=openai_ada,
    )

    for question in test_data.questions:
        if not question.requires_report and not question.chat_history:
            answer = openai_gpt4_simple_rag_chain.run(
                question=question.question,
            )
            verify_traced_response(answer, question.acceptable_answer_fragments)
