import warnings
from typing import Optional

import pytest
import torch
from langchain_community.chat_models import ChatFireworks, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

CUDA_DEVICE = "cpu"
if torch.cuda.is_available():
    CUDA_DEVICE = "cuda"


def verify_chain_response(
    response: Optional[str],
    match_fragment_str_options: list[str] = [],
    empty_ok: bool = False,
) -> None:
    if empty_ok and not response:
        return

    assert response
    if match_fragment_str_options:
        output_match = False
        for fragment in match_fragment_str_options:
            output_match = output_match or fragment.lower() in response.lower()

        assert output_match, (
            f"{response} does not contain one of the expected output "
            + "substrings {match_fragment_str_options}"
        )

    # Check guardrails and warn if any violations detected based on string checks
    for banned_word in ["sql", "table"]:
        if banned_word.lower() in response.lower():
            warnings.warn(
                UserWarning(f"Output contains banned word {banned_word}: {response}")
            )


@pytest.fixture()
def fireworksai_mixtral() -> BaseChatModel:
    """
    Mixtral8x7b model hosted on fireworksai
    """
    return ChatFireworks(name="accounts/fireworks/models/mixtral-8x7b-instruct")


@pytest.fixture()
def huggingface_minilm() -> Embeddings:
    """
    MiniLM-L6-v2 embeddings running locally using huggingface
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": CUDA_DEVICE},
    )


@pytest.fixture()
def openai_gpt35() -> BaseChatModel:
    """
    GPT 3.5 model by OpenAI
    """
    return ChatOpenAI(model="gpt-3.5-turbo-16k")


@pytest.fixture()
def openai_ada() -> Embeddings:
    """
    Ada embeddings by OpenAI
    """
    return OpenAIEmbeddings(model="text-embedding-ada-002", client=None)
