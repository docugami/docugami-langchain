import logging
import random
import time
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from docugami_langchain.agents.base import AgentState, BaseDocugamiAgent
from docugami_langchain.agents.models import CitationType, CitedAnswer
from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.config import DEFAULT_RETRIEVER_K, MAX_FULL_DOCUMENT_TEXT_LENGTH
from docugami_langchain.tools.common import BaseDocugamiTool, get_generic_tools
from docugami_langchain.tools.reports import (
    connect_to_excel,
    get_retrieval_tool_for_report,
    report_details_to_report_query_tool_description,
    report_name_to_report_query_tool_function_name,
)
from docugami_langchain.tools.retrieval import (
    docset_name_to_direct_retrieval_tool_function_name,
    get_retrieval_tool_for_docset,
    summaries_to_direct_retrieval_tool_description,
)
from docugami_langchain.utils.sql import get_table_info_as_list
from tests.common import (
    EXAMPLES_PATH,
    TEST_DATA_DIR,
    build_test_retrieval_artifacts,
    verify_output,
    verify_output_list,
)
from tests.testdata.docsets.docset_test_data import DocsetTestData
from tests.testdata.xlsx.query_test_data import TestReportData


def build_test_query_tool(
    report: TestReportData, llm: BaseLanguageModel, embeddings: Embeddings
) -> BaseDocugamiTool:
    """
    Builds a query tool over a test database
    """
    db = connect_to_excel(report.data_file, report.name)
    description = report_details_to_report_query_tool_description(
        report.name, get_table_info_as_list(db)
    )
    tool = get_retrieval_tool_for_report(
        local_xlsx_path=report.data_file,
        report_name=report.name,
        retrieval_tool_function_name=report_name_to_report_query_tool_function_name(
            report.name
        ),
        retrieval_tool_description=description,
        sql_llm=llm,
        general_llm=llm,
        embeddings=embeddings,
        sql_fixup_examples_file=EXAMPLES_PATH / "test_sql_fixup_examples.yaml",
        sql_examples_file=EXAMPLES_PATH / "test_sql_examples.yaml",
        data_type_detection_examples_file=EXAMPLES_PATH
        / "test_data_type_detection_examples.yaml",
        date_parse_examples_file=EXAMPLES_PATH / "test_date_parse_examples.yaml",
        float_parse_examples_file=EXAMPLES_PATH / "test_float_parse_examples.yaml",
    )

    if not tool:
        raise Exception("Could not create test query tool")

    return tool


def build_test_common_tools(
    llm: BaseLanguageModel, embeddings: Embeddings
) -> list[BaseDocugamiTool]:
    """
    Builds common tools for test purposes
    """
    return get_generic_tools(
        llm=llm,
        embeddings=embeddings,
        answer_examples_file=EXAMPLES_PATH / "test_answer_examples.yaml",
    )


def build_test_retrieval_tool(
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    docset: DocsetTestData,
    data_files_glob: str = "*.xml",
) -> BaseDocugamiTool:
    """
    Builds a vector store pre-populated with chunks from test documents
    using the given embeddings, and returns a retriever tool off it.
    """
    (
        vector_store,
        full_doc_summaries_by_id,
        _fetch_parent_doc_callback,
        _fetch_full_doc_summary_callback,
    ) = build_test_retrieval_artifacts(
        llm,
        embeddings,
        TEST_DATA_DIR / "docsets" / docset.name,
        data_files_glob,
    )

    retrieval_tool_description = summaries_to_direct_retrieval_tool_description(
        name=docset.name,
        summaries=random.sample(
            list(full_doc_summaries_by_id.values()),
            min(len(full_doc_summaries_by_id), 3),
        ),  # give 3 randomly selected summaries summaries
        llm=llm,
        embeddings=embeddings,
        max_sample_documents_cutoff_length=MAX_FULL_DOCUMENT_TEXT_LENGTH,
        describe_document_set_examples_file=EXAMPLES_PATH
        / "test_describe_document_set_examples.yaml",
    )
    tool = get_retrieval_tool_for_docset(
        chunk_vectorstore=vector_store,
        retrieval_tool_function_name=docset_name_to_direct_retrieval_tool_function_name(
            docset.name
        ),
        retrieval_tool_description=retrieval_tool_description,
        llm=llm,
        embeddings=embeddings,
        fetch_parent_doc_callback=_fetch_parent_doc_callback,
        fetch_full_doc_summary_callback=_fetch_full_doc_summary_callback,
        retrieval_grader_examples_file=EXAMPLES_PATH
        / "test_retrieval_grader_examples.yaml",
        retrieval_k=DEFAULT_RETRIEVER_K,
    )
    if not tool:
        raise Exception("Failed to create retrieval tool")

    return tool


def run_agent_test(
    agent: BaseDocugamiAgent,
    question: str,
    answer_options: list[str] = [],
    chat_history: list[tuple[str, str]] = [],
    citation_label_options: list[str] = [],
) -> None:

    response = agent.run(
        question=question,
        chat_history=chat_history,
    )

    assert response.run_id
    cited_answer: Optional[CitedAnswer] = response.value.get("cited_answer")

    assert cited_answer
    assert cited_answer.is_final
    verify_output(cited_answer.answer, answer_options)

    if citation_label_options:
        assert cited_answer.citations

        for c in cited_answer.citations:
            assert c.citation_type
            if c.citation_type == CitationType.DOCUMENT:
                assert c.document_id
            elif c.citation_type == CitationType.REPORT:
                assert c.report_query

        verify_output_list(
            [c.label for c in cited_answer.citations],
            citation_label_options,
            empty_ok=False,
        )


async def run_streaming_agent_test(
    agent: BaseDocugamiAgent,
    question: str,
    answer_options: list[str],
    chat_history: list[tuple[str, str]] = [],
    citation_label_options: list[str] = [],
) -> None:
    last_response = TracedResponse[AgentState](value={})  # type: ignore

    streamed_answers: list = []
    step_deltas: list = []
    start_time = time.time()  # Start timing before the test begins

    async for incremental_response in agent.run_stream(
        question=question,
        chat_history=chat_history,
    ):
        if incremental_response.value:
            streamed_answer = incremental_response.value.get("cited_answer")
            if streamed_answer:
                streamed_answers.append(streamed_answer)
                current_time = time.time()
                step_deltas.append(
                    (
                        streamed_answer.answer,
                        round(
                            current_time
                            - start_time
                            - sum([s[1] for s in step_deltas]),
                            2,
                        ),
                    )
                )

        last_response = incremental_response

    # Log the spacing between steps
    if len(step_deltas) > 1:
        logging.info("Streamed answers, with time deltas")
        for s in step_deltas:
            logging.info(f"{s[0]}|{s[1]}")

    assert streamed_answers
    assert last_response.run_id
    cited_answer: Optional[CitedAnswer] = last_response.value.get("cited_answer")

    assert cited_answer
    assert cited_answer.is_final
    verify_output(cited_answer.answer, answer_options)

    if citation_label_options:
        assert cited_answer.citations

        for c in cited_answer.citations:
            assert c.citation_type
            if c.citation_type == CitationType.DOCUMENT:
                assert c.document_id
            elif c.citation_type == CitationType.REPORT:
                assert c.report_query

        verify_output_list(
            [c.label for c in cited_answer.citations],
            citation_label_options,
            empty_ok=False,
        )
