import hashlib
from pathlib import Path
from typing import Dict, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from langchain_docugami.chains import (
    SummarizeChunkChain,
    SummarizeDocumentChain,
)
from langchain_docugami.config import (
    MAX_CHUNK_TEXT_LENGTH,
    MAX_FULL_DOCUMENT_TEXT_LENGTH,
    MIN_LENGTH_TO_SUMMARIZE,
)
from langchain_docugami.retrievers.fused_summary import PARENT_DOC_ID_KEY


def _build_summary_mappings(
    docs_by_id: Dict[str, Document],
    chain: Union[SummarizeChunkChain, SummarizeDocumentChain],
    include_xml_tags: bool = False,
) -> Dict[str, Document]:
    """
    Build summaries for all the given documents.
    """
    summaries: Dict[str, Document] = {}
    format: str = (
        "text"
        if not include_xml_tags
        else "semantic XML without any namespaces or attributes"
    )

    batch_input = [(doc.page_content, format) for _, doc in docs_by_id.items()]
    batch_summaries = chain.run_batch(batch_input)  # type: ignore

    # Assigning summaries to the respective document IDs
    for (id, doc), summary in zip(docs_by_id.items(), batch_summaries):
        summary_id = hashlib.md5(summary.encode()).hexdigest()
        meta = doc.metadata
        meta["id"] = summary_id
        meta[PARENT_DOC_ID_KEY] = id

        summaries[id] = Document(
            page_content=summary,
            metadata=meta,
        )

    return summaries


def build_full_doc_summary_mappings(
    docs_by_id: Dict[str, Document],
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    min_length_to_summarize: int = MIN_LENGTH_TO_SUMMARIZE,
    max_length_cutoff: int = MAX_FULL_DOCUMENT_TEXT_LENGTH,
    summarize_document_examples_file: Optional[Path] = None,
) -> Dict[str, Document]:
    """
    Build summary mappings for all the given full documents.
    """

    chain = SummarizeDocumentChain(llm=llm, embeddings=embeddings)
    chain.min_length_to_summarize = min_length_to_summarize
    chain.input_params_max_length_cutoff = max_length_cutoff
    if summarize_document_examples_file:
        chain.load_examples(summarize_document_examples_file)

    return _build_summary_mappings(docs_by_id=docs_by_id, chain=chain)


def build_chunk_summary_mappings(
    docs_by_id: Dict[str, Document],
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    min_length_to_summarize: int = MIN_LENGTH_TO_SUMMARIZE,
    max_length_cutoff: int = MAX_CHUNK_TEXT_LENGTH,
    summarize_chunks_examples_file: Optional[Path] = None,
) -> Dict[str, Document]:
    """
    Build summary mappings for all the given chunks.
    """

    chain = SummarizeChunkChain(llm=llm, embeddings=embeddings)
    chain.min_length_to_summarize = min_length_to_summarize
    chain.input_params_max_length_cutoff = max_length_cutoff
    if summarize_chunks_examples_file:
        chain.load_examples(summarize_chunks_examples_file)

    return _build_summary_mappings(docs_by_id=docs_by_id, chain=chain)
