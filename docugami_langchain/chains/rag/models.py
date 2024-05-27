from typing import TypedDict

from langchain_core.documents import Document


class ExtendedRAGResult(TypedDict):
    question: str
    answer: str
    source_docs: list[Document]
