import re
from pathlib import Path
from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from rerankers.models.ranker import BaseRanker

from docugami_langchain.agents.models import Citation, CitedAnswer, Invocation
from docugami_langchain.chains.documents.describe_document_set_chain import (
    DescribeDocumentSetChain,
)
from docugami_langchain.chains.rag.simple_rag_chain import SimpleRAGChain
from docugami_langchain.config import DEFAULT_RETRIEVER_K, MAX_FULL_DOCUMENT_TEXT_LENGTH
from docugami_langchain.retrievers.fused_summary import (
    FULL_DOC_SUMMARY_ID_KEY,
    FusedRetrieverKeyValueFetchCallback,
    FusedSummaryRetriever,
    SearchType,
)
from docugami_langchain.tools.common import NOT_FOUND, BaseDocugamiTool


class CustomDocsetRetrievalTool(BaseDocugamiTool):
    """A Tool that knows how to do retrieval over a docset."""

    chain: SimpleRAGChain
    name: str = "document_answer_tool"
    description: str = ""

    def to_human_readable(self, invocation: Invocation) -> str:
        """Converts tool invocation to human readable string."""
        return f"Searching documents: {invocation.tool_input}"

    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> CitedAnswer:  # type: ignore
        """Use the tool."""

        if not question:
            return CitedAnswer(
                source=self.name,
                answer="Please specify a question that you want to answer from this docset",
            )

        try:
            config = None
            if run_manager:
                config = RunnableConfig(
                    run_name=self.__class__.__name__,
                    callbacks=run_manager,
                )
            chain_response = self.chain.run(
                question=question,
                config=config,
            )
            if chain_response.value:
                answer = chain_response.value.get("answer")
                sources = chain_response.value.get("sources", [])
                if answer:
                    return CitedAnswer(
                        source=self.name,
                        answer=answer,
                        citations=[Citation(label=s) for s in sources]
                    )

            return CitedAnswer(source=self.name, answer=NOT_FOUND)
        except Exception as exc:
            return CitedAnswer(
                source=self.name,
                answer=f"There was an error. Please try a different question, or a different tool. Details: {exc}",
            )


def docset_name_to_direct_retrieval_tool_function_name(name: str) -> str:
    """
    Converts a docset name to a direct retriever tool function name.

    Direct retriever tool function names follow these conventions:
    1. Retrieval tool function names always start with "document_answer_tool".
    2. The rest of the name should be a lowercased string, with underscores
       for whitespace.
    3. Exclude any characters other than a-z (lowercase) from the function
       name, replacing them with underscores.
    4. The final function name should not have more than one underscore together.

    >>> docset_name_to_direct_retrieval_tool_function_name('Earnings Calls')
    'document_answer_tool_earnings_calls'
    >>> docset_name_to_direct_retrieval_tool_function_name('COVID-19   Statistics')
    'document_answer_tool_covid_19_statistics'
    >>> docset_name_to_direct_retrieval_tool_function_name('2023 Market Report!!!')
    'document_answer_tool_2023_market_report'
    """
    # Replace non-letter characters with underscores and remove extra whitespaces
    name = re.sub(r"[^a-z\d]", "_", name.lower())
    # Replace whitespace with underscores and remove consecutive underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")

    return f"document_answer_tool_{name}"


def docset_details_to_direct_retrieval_tool_description(
    name: str, description: str
) -> str:
    return (
        "Pass the COMPLETE question as input to this tool."
        + f"It implements logic to to answer questions based on information in {name} documents and outputs only the answer to your question. "
        + "Use this tool if you think the answer is likely to come from one or a few of these documents. "
        + description
    )


def summaries_to_direct_retrieval_tool_description(
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

    return docset_details_to_direct_retrieval_tool_description(name, description.value)


def get_retrieval_tool_for_docset(
    chunk_vectorstore: VectorStore,
    retrieval_tool_function_name: str,
    retrieval_tool_description: str,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    re_ranker: Optional[BaseRanker] = None,
    fetch_full_doc_summary_callback: Optional[
        FusedRetrieverKeyValueFetchCallback
    ] = None,
    fetch_parent_doc_callback: Optional[FusedRetrieverKeyValueFetchCallback] = None,
    retrieval_k: int = DEFAULT_RETRIEVER_K,
    full_doc_summary_id_key: str = FULL_DOC_SUMMARY_ID_KEY,
) -> Optional[BaseDocugamiTool]:
    """
    Gets a retrieval tool for an agent.
    """

    # Instantiate FusedSummaryRetriever with callback functions
    retriever = FusedSummaryRetriever(
        vectorstore=chunk_vectorstore,
        re_ranker=re_ranker,
        fetch_parent_doc_callback=fetch_parent_doc_callback,
        full_doc_summary_id_key=full_doc_summary_id_key,
        fetch_full_doc_summary_callback=fetch_full_doc_summary_callback,
        retriever_k=retrieval_k,
        search_type=SearchType.mmr,
    )

    simple_rag_chain = SimpleRAGChain(
        llm=llm,
        embeddings=embeddings,
        retriever=retriever,
    )

    return CustomDocsetRetrievalTool(
        chain=simple_rag_chain,
        name=retrieval_tool_function_name,
        description=retrieval_tool_description,
    )
