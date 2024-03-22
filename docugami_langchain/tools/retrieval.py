import re
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool, Tool
from langchain_core.vectorstores import VectorStore
from rerankers.models.ranker import BaseRanker

from docugami_langchain.chains.documents.describe_document_set_chain import (
    DescribeDocumentSetChain,
)
from docugami_langchain.config import DEFAULT_RETRIEVER_K, MAX_FULL_DOCUMENT_TEXT_LENGTH
from docugami_langchain.retrievers.fused_summary import (
    FusedRetrieverKeyValueFetchCallback,
    FusedSummaryRetriever,
    SearchType,
)


class RetrieverInput(BaseModel):
    """Input to the retriever."""

    question: str = Field(
        description="question to look up in retriever, to find relevant chunks that might contain answers"
    )


def docset_name_to_direct_retriever_tool_function_name(name: str) -> str:
    """
    Converts a docset name to a direct retriever tool function name.

    Direct retriever tool function names follow these conventions:
    1. Retrieval tool function names always start with "retrieval_".
    2. The rest of the name should be a lowercased string, with underscores
       for whitespace.
    3. Exclude any characters other than a-z (lowercase) from the function
       name, replacing them with underscores.
    4. The final function name should not have more than one underscore together.

    >>> docset_name_to_direct_retriever_tool_function_name('Earnings Calls')
    'retrieval_earnings_calls'
    >>> docset_name_to_direct_retriever_tool_function_name('COVID-19   Statistics')
    'retrieval_covid_19_statistics'
    >>> docset_name_to_direct_retriever_tool_function_name('2023 Market Report!!!')
    'retrieval_2023_market_report'
    """
    # Replace non-letter characters with underscores and remove extra whitespaces
    name = re.sub(r"[^a-z\d]", "_", name.lower())
    # Replace whitespace with underscores and remove consecutive underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")

    return f"retrieval_{name}"


def summaries_to_direct_retriever_tool_description(
    name: str,
    summaries: list[Document],
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    max_sample_documents_cutoff_length: int = MAX_FULL_DOCUMENT_TEXT_LENGTH,
    describe_document_set_examples_file: Optional[Path] = None,
) -> str:
    """
    Converts a set of chunks to a direct retriever tool description.
    """
    chain = DescribeDocumentSetChain(llm=llm, embeddings=embeddings)
    chain.input_params_max_length_cutoff = max_sample_documents_cutoff_length
    if describe_document_set_examples_file:
        chain.load_examples(describe_document_set_examples_file)

    description = chain.run(summaries=summaries, docset_name=name)
    return (
        "Pass the user's question, after rewriting it to be self-contained based on chat history, as input directly to this tool. "
        + f"Internally, it has logic to retrieve relevant chunks from {name} documents that might contain answers to the question. "
        + "Use this tool if you think the answer is likely to come from one or a few of these documents, and can be synthesized from "
        + "retrieved chunks. "
        + description.value
    )


def get_retrieval_tool_for_docset(
    chunk_vectorstore: VectorStore,
    retrieval_tool_function_name: str,
    retrieval_tool_description: str,
    re_ranker: BaseRanker,
    fetch_full_doc_summary_callback: FusedRetrieverKeyValueFetchCallback,
    fetch_parent_doc_callback: FusedRetrieverKeyValueFetchCallback,
    retrieval_k: int = DEFAULT_RETRIEVER_K,
) -> Optional[BaseTool]:
    """
    Gets a retrieval tool for an agent.
    """

    # Instantiate FusedSummaryRetriever with callback functions
    retriever = FusedSummaryRetriever(
        vectorstore=chunk_vectorstore,
        re_ranker=re_ranker,
        fetch_parent_doc_callback=fetch_parent_doc_callback,
        fetch_full_doc_summary_callback=fetch_full_doc_summary_callback,
        retriever_k=retrieval_k,
        search_type=SearchType.mmr,
    )

    if not retriever:
        return None

    def wrapped_get_relevant_documents(
        question: str,
        callbacks: Any = None,
        tags: Any = None,
        metadata: Any = None,
        run_name: Any = None,
        **kwargs: Any,
    ) -> str:
        docs: list[Document] = retriever.get_relevant_documents(
            query=question,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            run_name=run_name,
            **kwargs,
        )
        return "\n\n".join([doc.page_content for doc in docs])

    async def awrapped_get_relevant_documents(
        question: str,
        callbacks: Any = None,
        tags: Any = None,
        metadata: Any = None,
        run_name: Any = None,
        **kwargs: Any,
    ) -> str:
        docs: list[Document] = await retriever.aget_relevant_documents(
            query=question,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            run_name=run_name,
            **kwargs,
        )
        return "\n\n".join([doc.page_content for doc in docs])

    return Tool(
        name=retrieval_tool_function_name,
        description=retrieval_tool_description,
        func=wrapped_get_relevant_documents,
        coroutine=awrapped_get_relevant_documents,
        args_schema=RetrieverInput,
    )
