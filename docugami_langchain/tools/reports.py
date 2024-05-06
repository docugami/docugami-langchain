import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from langchain_community.tools.sql_database.tool import BaseSQLDatabaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig

from docugami_langchain.agents.models import Citation, CitedAnswer, Invocation
from docugami_langchain.chains.querying import (
    DocugamiExplainedSQLQueryChain,
    SQLFixupChain,
    SQLQueryExplainerChain,
    SQLResultChain,
)
from docugami_langchain.config import MAX_PARAMS_CUTOFF_LENGTH_CHARS
from docugami_langchain.tools.common import NOT_FOUND, BaseDocugamiTool


class CustomReportRetrievalTool(BaseSQLDatabaseTool, BaseDocugamiTool):
    db: SQLDatabase
    chain: DocugamiExplainedSQLQueryChain
    name: str = "report_answer_tool"
    description: str = ""

    def _is_sql_like(self, question: str) -> bool:
        question = question.lower().strip()
        return question.startswith("select") and "from" in question

    def to_human_readable(self, invocation: Invocation) -> str:
        if self._is_sql_like(invocation.tool_input):
            return "Querying report."
        else:
            return f"Querying report: {invocation.tool_input}"

    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> CitedAnswer:  # type: ignore
        """Use the tool."""

        if self._is_sql_like(question):
            return CitedAnswer(
                source=self.name,
                answer="Looks like you passed in a SQL query. This tool takes natural language questions, and automatically translates them to SQL queries. "
                "Please try again with a natural language version of this question.",
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
                results = chain_response.value.get("results")
                if results:
                    sql_result = results.get("sql_result", "")  # type: ignore
                    sql_query = results.get("sql_query", "")  # type: ignore
                    explained_sql_query = results.get("explained_sql_query", "")  # type: ignore
                    if sql_result:
                        return CitedAnswer(
                            source=self.name,
                            answer=sql_result,
                            citations=(
                                [
                                    Citation(
                                        label="Query",
                                        details=explained_sql_query,
                                        link=sql_query if sql_query else "",
                                    )
                                ]
                                if explained_sql_query
                                else []
                            ),
                        )

            return CitedAnswer(source=self.name, answer=NOT_FOUND)
        except Exception as exc:
            return CitedAnswer(
                source=self.name,
                answer=f"There was an error. Please try a different question, or a different tool. Details: {exc}",
            )


def report_name_to_report_query_tool_function_name(name: str) -> str:
    """
    Converts a report name to a report query tool function name.

    Report query tool function names follow these conventions:
    1. Report tool function names always start with "report_answer_tool_".
    2. The rest of the name should be a lowercased string, with underscores
       for whitespace.
    3. Exclude any characters other than a-z (lowercase) from the function name,
       replacing them with underscores.
    4. The final function name should not have more than one underscore together.

    >>> report_name_to_report_query_tool_function_name('Earnings Calls')
    'report_answer_tool_earnings_calls'
    >>> report_name_to_report_query_tool_function_name('COVID-19   Statistics')
    'report_answer_tool_covid_19_statistics'
    >>> report_name_to_report_query_tool_function_name('2023 Market Report!!!')
    'report_answer_tool_2023_market_report'
    """
    # Replace non-letter characters with underscores and remove extra whitespaces
    name = re.sub(r"[^a-z\d]", "_", name.lower())
    # Replace whitespace with underscores and remove consecutive underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")

    return f"report_answer_tool_{name}"


def report_details_to_report_query_tool_description(name: str, table_info: str) -> str:
    """
    Converts a set of chunks to a direct retriever tool description.
    """
    description = (
        "Pass the COMPLETE question as input to this tool. "
        + f"It implements logic to to answer questions by querying the {name} report and outputs only the answer to your question. "
        + f"Use this tool if you think the answer can be calculated from the information in this report via standard data operations like counting, sorting, averaging or summing.\n\n{table_info}"
    )

    # Cap to avoid runaway tool descriptions.
    return description[:MAX_PARAMS_CUTOFF_LENGTH_CHARS]


def excel_to_sqlite_connection(
    file_path: Union[Path, str], table_name: str
) -> sqlite3.Connection:
    # Create a temporary SQLite database in memory
    conn = sqlite3.connect(":memory:")

    # Verify the file path
    file_path = Path(file_path)
    if not (file_path.exists() and file_path.suffix.lower() == ".xlsx"):
        raise Exception(f"Invalid file path: {file_path}")

    # Read the Excel file using pandas (only the first sheet)
    df = pd.read_excel(file_path, sheet_name=0)

    # Ignore non-informational columns
    DROP_COLUMNS = ["FileId", "File", "Link to Document"]
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Write the table to the SQLite database
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    return conn


def connect_to_db(conn: sqlite3.Connection) -> SQLDatabase:
    temp_db_file = tempfile.NamedTemporaryFile(suffix=".sqlite")
    with sqlite3.connect(temp_db_file.name) as disk_conn:
        conn.backup(disk_conn)  # dumps the connection to disk
    return SQLDatabase.from_uri(
        f"sqlite:///{temp_db_file.name}",
        sample_rows_in_table_info=0,  # We select and insert sample rows using custom logic
    )


def connect_to_excel(file_path: Union[Path, str], table_name: str) -> SQLDatabase:
    return connect_to_db(excel_to_sqlite_connection(file_path, table_name))


def get_retrieval_tool_for_report(
    local_xlsx_path: Path,
    report_name: str,
    retrieval_tool_function_name: str,
    retrieval_tool_description: str,
    sql_llm: BaseLanguageModel,
    explainer_llm: BaseLanguageModel,
    embeddings: Embeddings,
    sql_fixup_examples_file: Optional[Path] = None,
    sql_examples_file: Optional[Path] = None,
) -> Optional[BaseDocugamiTool]:
    if not local_xlsx_path.exists():
        return None

    db = connect_to_excel(local_xlsx_path, report_name)

    fixup_chain = SQLFixupChain(llm=sql_llm, embeddings=embeddings)
    if sql_fixup_examples_file:
        fixup_chain.load_examples(sql_fixup_examples_file)

    sql_result_chain = SQLResultChain(
        llm=sql_llm,
        embeddings=embeddings,
        db=db,
        sql_fixup_chain=fixup_chain,
    )
    if sql_examples_file:
        sql_result_chain.load_examples(sql_examples_file)
    sql_result_chain.optimize()

    sql_query_explainer_chain = SQLQueryExplainerChain(
        llm=explainer_llm,
        embeddings=embeddings,
    )
    if sql_examples_file:
        sql_query_explainer_chain.load_examples(sql_examples_file)

    return CustomReportRetrievalTool(
        db=db,
        chain=DocugamiExplainedSQLQueryChain(
            llm=explainer_llm,
            embeddings=embeddings,
            sql_result_chain=sql_result_chain,
            sql_query_explainer_chain=sql_query_explainer_chain,
        ),
        name=retrieval_tool_function_name,
        description=retrieval_tool_description,
    )
