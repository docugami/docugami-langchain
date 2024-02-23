"""Test DocugamiLoader."""

import pytest

from docugami_langchain.document_loaders.docugami import DocugamiLoader
from tests.common import TEST_DATA_DIR


@pytest.mark.requires("dgml_utils")
def test_docugami_loader_local() -> None:
    """Test DocugamiLoader."""
    loader = DocugamiLoader(file_paths=[TEST_DATA_DIR / "simple-dgml.xml"])
    docs = loader.load()

    assert len(docs) == 25

    assert "/docset:DisclosingParty" in docs[1].metadata["xpath"]
    assert "h1" in docs[1].metadata["structure"]
    assert "DisclosingParty" in docs[1].metadata["tag"]
    assert docs[1].page_content.startswith("Disclosing")


def test_docugami_initialization() -> None:
    """Test correct initialization in remote mode."""
    DocugamiLoader(access_token="test", docset_id="123")
