from typing import TypedDict


class ExtendedRAGResult(TypedDict):
    question: str
    answer: str
    sources: list[str]
