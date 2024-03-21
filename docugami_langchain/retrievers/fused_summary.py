from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from rerankers import Reranker
from rerankers.models.ranker import BaseRanker

from docugami_langchain.config import DEFAULT_RETRIEVER_K

PARENT_DOC_ID_KEY = "doc_id"
FULL_DOC_SUMMARY_ID_KEY = "full_doc_id"
SOURCE_KEY = "source"

FusedRetrieverKeyValueFetchCallback = Callable[[str], Optional[str]]


class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""


@dataclass
class FusedDocumentElements:
    rank: int
    summary: str
    fragments: list[str]
    source: str


DOCUMENT_SUMMARY_TEMPLATE: str = """
--------------------------------
**** DOCUMENT NAME: {doc_name}

**** DOCUMENT SUMMARY:
{summary}

**** RELEVANT FRAGMENTS:
{fragments}
--------------------------------
"""


class FusedSummaryRetriever(BaseRetriever):
    """
    Retrieves a fused document that includes pre-calculated summaries
    for the full-document as well as individual chunks. Specifically:

    - Full document summaries are included in the fused document to give
      broader context to the LLM, which may not be in the retrieved chunks

    - Chunk summaries are using to improve retrieval, i.e. "big-to-small"
      retrieval which is a common use case with the [multi-vector retriever](./)

    """

    vectorstore: VectorStore
    """The underlying vectorstore to use to store small chunks
    and their embedding vectors."""

    retriever_k: int = DEFAULT_RETRIEVER_K
    """The number of chunks the retriever tries to get from the vectorstore."""

    re_rank_enabled: bool = True
    """Set to False to disable re_rank."""

    re_rank_filter_percentile: float = 80
    """Results above this percentile are kept, others are rejected (0 means keep them all, 90 means keep only very good ones, 100 means keep the top one, etc.)."""

    re_rank_model_id: str = "mixedbread-ai/mxbai-rerank-base-v1"
    """The model ID to use for re_rank. See https://github.com/AnswerDotAI/rerankers for details."""

    re_ranker: Optional[BaseRanker] = None
    """Re-ranker is automatically created on init (based on the other re_rank properties)."""

    parent_id_key: str = PARENT_DOC_ID_KEY
    """Metadata key for parent doc ID (maps chunk summaries in the vector store to parent / unsummarized chunks)."""

    fetch_parent_doc_callback: Optional[FusedRetrieverKeyValueFetchCallback] = None
    """Callback to fetch parent docs by ID key."""

    full_doc_summary_id_key: str = FULL_DOC_SUMMARY_ID_KEY
    """Metadata key for full doc summary ID (maps chunk summaries in the vector store to full doc summaries)."""

    fetch_full_doc_summary_callback: Optional[FusedRetrieverKeyValueFetchCallback] = (
        None
    )
    """Callback to fetch full doc summaries by ID key."""

    source_key: str = SOURCE_KEY
    """Metadata key for source document of chunks."""

    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    search_type: SearchType = SearchType.mmr
    """Type of search to perform (similarity / mmr)"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if self.re_rank_enabled:
            self.re_ranker = Reranker(self.re_rank_model_id, verbose=0)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """

        if self.search_kwargs:
            if "k" not in self.search_kwargs:
                self.search_kwargs["k"] = self.retriever_k

        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        if self.re_rank_enabled and self.re_ranker:
            # Re-rank
            doc_contents = [doc.page_content for doc in sub_docs]
            ranked_results = self.re_ranker.rank(
                query=query, docs=doc_contents, doc_ids=list(range(len(sub_docs)))
            )
            scores_by_ranker_id = {
                result.doc_id: result.score for result in ranked_results.results
            }
            score_threshold = np.percentile(
                [float(s) for s in scores_by_ranker_id.values()],
                self.re_rank_filter_percentile, 
            )
            filtered_sub_docs = [
                doc
                for idx, doc in enumerate(sub_docs)
                if scores_by_ranker_id[idx] >= score_threshold
            ]
            sub_docs = filtered_sub_docs

        fused_doc_elements: dict[str, FusedDocumentElements] = {}
        for i, sub_doc in enumerate(sub_docs):
            parent_id = sub_doc.metadata.get(self.parent_id_key)
            full_doc_summary_id = sub_doc.metadata.get(self.full_doc_summary_id_key)
            parent: Optional[str] = None
            full_doc_summary: Optional[str] = None

            if parent_id and self.fetch_parent_doc_callback:
                parent = self.fetch_parent_doc_callback(parent_id)

            if full_doc_summary_id and self.fetch_full_doc_summary_callback:
                full_doc_summary = self.fetch_full_doc_summary_callback(
                    full_doc_summary_id
                )

            source: str = sub_doc.metadata.get(self.source_key, "")
            key = full_doc_summary_id if full_doc_summary_id else "-1"

            if key not in fused_doc_elements:
                fused_doc_elements[key] = FusedDocumentElements(
                    rank=i,
                    summary=(full_doc_summary if full_doc_summary else ""),
                    fragments=[parent if parent else sub_doc.page_content],
                    source=source,
                )
            else:
                fused_doc_elements[key].fragments.append(
                    parent if parent else sub_doc.page_content
                )

        fused_docs: list[Document] = []
        for element in sorted(fused_doc_elements.values(), key=lambda x: x.rank):
            fragments_str: str = "\n\n".join([d.strip() for d in element.fragments])
            fused_docs.append(
                Document(
                    page_content=DOCUMENT_SUMMARY_TEMPLATE.format(
                        doc_name=element.source,
                        summary=element.summary,
                        fragments=fragments_str,
                    )
                )
            )

        return fused_docs
