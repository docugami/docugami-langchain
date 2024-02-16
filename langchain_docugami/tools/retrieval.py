import re
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.stores import BaseStore
from langchain_core.tools import BaseTool, Tool
from langchain_core.vectorstores import VectorStore

from langchain_docugami.chains.documents.describe_document_set_chain import (
    DescribeDocumentSetChain,
)
from langchain_docugami.config import MAX_CHUNK_TEXT_LENGTH, RETRIEVER_K
from langchain_docugami.retrievers.fused_summary import (
    FusedSummaryRetriever,
    SearchType,
)


class RetrieverInput(BaseModel):
    """Input to the retriever."""

    query: str = Field(description="query to look up in retriever")


def docset_name_to_direct_retriever_tool_function_name(name: str) -> str:
    """
    Converts a docset name to a direct retriever tool function name.

    Direct retriever tool function names follow these conventions:
    1. Retrieval tool function names always start with "search_".
    2. The rest of the name should be a lowercased string, with underscores
       for whitespace.
    3. Exclude any characters other than a-z (lowercase) from the function
       name, replacing them with underscores.
    4. The final function name should not have more than one underscore together.

    >>> docset_name_to_direct_retriever_tool_function_name('Earnings Calls')
    'search_earnings_calls'
    >>> docset_name_to_direct_retriever_tool_function_name('COVID-19   Statistics')
    'search_covid_19_statistics'
    >>> docset_name_to_direct_retriever_tool_function_name('2023 Market Report!!!')
    'search_2023_market_report'
    """
    # Replace non-letter characters with underscores and remove extra whitespaces
    name = re.sub(r"[^a-z\d]", "_", name.lower())
    # Replace whitespace with underscores and remove consecutive underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")

    return f"search_{name}"


def chunks_to_direct_retriever_tool_description(
    name: str,
    chunks: List[Document],
    llm: BaseChatModel,
    embeddings: Embeddings,
    max_chunk_text_length: int = MAX_CHUNK_TEXT_LENGTH,
) -> str:
    """
    Converts a set of chunks to a direct retriever tool description.
    """
    texts = [c.page_content for c in chunks[:100]]
    sample = "\n".join(texts)[:max_chunk_text_length]

    chain = DescribeDocumentSetChain(llm=llm, embeddings=embeddings)
    description = chain.run(sample=sample, docset_name=name)

    return f"Given a single input 'query' parameter, searches for and returns chunks from {name} documents. {description}"


def get_retrieval_tool_for_docset(
    chunk_vectorstore: VectorStore,
    retrieval_tool_function_name: str,
    retrieval_tool_description: str,
    full_doc_summary_store: BaseStore[str, Document],
    parent_doc_store: BaseStore[str, Document],
    retrieval_k: int = RETRIEVER_K,
) -> Optional[BaseTool]:
    """
    Gets a retrieval tool for an agent.
    """

    retriever = FusedSummaryRetriever(
        vectorstore=chunk_vectorstore,
        parent_doc_store=parent_doc_store,
        full_doc_summary_store=full_doc_summary_store,
        search_kwargs={"k": retrieval_k},
        search_type=SearchType.mmr,
    )

    if not retriever:
        return None

    def wrapped_get_relevant_documents(
        query: str,
        callbacks: Any = None,
        tags: Any = None,
        metadata: Any = None,
        run_name: Any = None,
        **kwargs: Any,
    ) -> str:
        docs: List[Document] = retriever.get_relevant_documents(
            query,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            run_name=run_name,
            **kwargs,
        )
        return "\n\n".join([doc.page_content for doc in docs])

    async def awrapped_get_relevant_documents(
        query: str,
        callbacks: Any = None,
        tags: Any = None,
        metadata: Any = None,
        run_name: Any = None,
        **kwargs: Any,
    ) -> str:
        docs: List[Document] = await retriever.aget_relevant_documents(
            query,
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
