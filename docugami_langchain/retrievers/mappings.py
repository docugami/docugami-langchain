import hashlib
from pathlib import Path
from typing import Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.chains import SummarizeChunkChain, SummarizeDocumentChain
from docugami_langchain.config import (
    BATCH_SIZE,
    INCLUDE_XML_TAGS,
    MAX_CHUNK_TEXT_LENGTH,
    MAX_FULL_DOCUMENT_TEXT_LENGTH,
    MIN_LENGTH_TO_SUMMARIZE,
)
from docugami_langchain.retrievers.fused_summary import (
    FULL_DOC_SUMMARY_ID_KEY,
    PARENT_DOC_ID_KEY,
    SOURCE_KEY,
)


def _build_summary_mappings(
    docs_by_id: dict[str, Document],
    chain: Union[SummarizeChunkChain, SummarizeDocumentChain],
    include_xml_tags: bool,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Document]:
    """
    Build summaries for all the given documents in batches of a specified size.
    """
    summaries: dict[str, Document] = {}
    format: str = (
        "text"
        if not include_xml_tags
        else "semantic XML without any namespaces or attributes"
    )

    # Create batches of input tuples
    items = list(docs_by_id.items())
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    for batch in batches:
        batch_input = [(doc.page_content, format) for _, doc in batch]
        batch_summaries = chain.run_batch(batch_input)  # type: ignore

        # Assigning summaries to the respective document IDs
        for (id, doc), summary in zip(batch, batch_summaries):
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
    docs_by_id: dict[str, Document],
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    include_xml_tags: bool = INCLUDE_XML_TAGS,
    min_length_to_summarize: int = MIN_LENGTH_TO_SUMMARIZE,
    max_length_cutoff: int = MAX_FULL_DOCUMENT_TEXT_LENGTH,
    summarize_document_examples_file: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Document]:
    """
    Build summary mappings for all the given full documents.
    """

    chain = SummarizeDocumentChain(llm=llm, embeddings=embeddings)
    chain.min_length_to_summarize = min_length_to_summarize
    chain.input_params_max_length_cutoff = max_length_cutoff
    if summarize_document_examples_file:
        chain.load_examples(summarize_document_examples_file)

    return _build_summary_mappings(
        docs_by_id=docs_by_id,
        chain=chain,
        include_xml_tags=include_xml_tags,
        batch_size=batch_size,
    )


def build_chunk_summary_mappings(
    docs_by_id: dict[str, Document],
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    include_xml_tags: bool = INCLUDE_XML_TAGS,
    min_length_to_summarize: int = MIN_LENGTH_TO_SUMMARIZE,
    max_length_cutoff: int = MAX_CHUNK_TEXT_LENGTH,
    summarize_chunk_examples_file: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Document]:
    """
    Build summary mappings for all the given chunks.
    """

    chain = SummarizeChunkChain(llm=llm, embeddings=embeddings)
    chain.min_length_to_summarize = min_length_to_summarize
    chain.input_params_max_length_cutoff = max_length_cutoff
    if summarize_chunk_examples_file:
        chain.load_examples(summarize_chunk_examples_file)

    return _build_summary_mappings(
        docs_by_id=docs_by_id,
        chain=chain,
        include_xml_tags=include_xml_tags,
        batch_size=batch_size,
    )


def build_doc_maps_from_chunks(
    chunks: list[Document],
    chunk_id_key: str = "id",
    parent_id_key: str = PARENT_DOC_ID_KEY,
    full_doc_summary_id_key: str = FULL_DOC_SUMMARY_ID_KEY,
    source_key: str = SOURCE_KEY,
) -> tuple[dict[str, Document], dict[str, Document]]:
    """Build separate maps of full docs and parent chunks (by individual chunk id)"""
    # Build separate maps of chunks, and parents
    parent_chunks_by_id: dict[str, Document] = {}
    chunks_by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        chunk_id = str(chunk.metadata.get(chunk_id_key))
        chunk_source = str(chunk.metadata.get(source_key))
        parent_chunk_id = chunk.metadata.get(parent_id_key)

        if chunk_source not in chunks_by_source:
            chunks_by_source[chunk_source] = []

        chunks_by_source[chunk_source].append(chunk.page_content)

        if not parent_chunk_id:
            # parent chunk, we will use this (for expanded context) as our chunk
            parent_chunks_by_id[chunk_id] = chunk

    # Build up the full docs by concatenating all the child chunks from a source
    full_docs_by_id: dict[str, Document] = {}
    full_doc_ids_by_source: dict[str, str] = {}
    for source in chunks_by_source:
        chunks_from_source = chunks_by_source[source]
        full_doc_text = "\n".join([c for c in chunks_from_source])
        full_doc_id = hashlib.md5(full_doc_text.encode()).hexdigest()
        full_doc_ids_by_source[source] = full_doc_id
        full_docs_by_id[full_doc_id] = Document(
            page_content=full_doc_text, metadata={chunk_id_key: full_doc_id}
        )

    # Associate parent chunks with full docs
    for parent_chunk_id in parent_chunks_by_id:
        parent_chunk = parent_chunks_by_id[parent_chunk_id]
        parent_chunk_source = parent_chunk.metadata.get(source_key)
        if parent_chunk_source:
            full_doc_id = full_doc_ids_by_source.get(parent_chunk_source, "")
            if full_doc_id:
                parent_chunk.metadata[full_doc_summary_id_key] = full_doc_id

    return full_docs_by_id, parent_chunks_by_id
