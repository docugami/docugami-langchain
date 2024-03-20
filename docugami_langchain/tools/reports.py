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
from langchain_core.tools import BaseTool

from docugami_langchain.chains.querying.sql_fixup_chain import SQLFixupChain
from docugami_langchain.chains.querying.sql_result_chain import SQLResultChain

NOT_FOUND = "Not found, please consider trying a different query"


class CustomReportRetrievalTool(BaseSQLDatabaseTool, BaseTool):
    db: SQLDatabase
    chain: SQLResultChain
    name: str = "query_report"
    description: str = ""

    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:  # type: ignore
        """Use the tool."""

        if "select" in question.lower() and "from" in question.lower():
            return "Looks like you passed in a SQL query. This tool takes natural language inputs, and automatically translates them to SQL queries. Please try passing in a natural language query."

        chain_response = self.chain.run(question=question)
        if chain_response.value:
            sql_result = chain_response.value.get("sql_result")
            if sql_result:
                return str(sql_result)

        return NOT_FOUND


def report_name_to_report_query_tool_function_name(name: str) -> str:
    """
    Converts a report name to a report query tool function name.

    Report query tool function names follow these conventions:
    1. Retrieval tool function names always start with "query_".
    2. The rest of the name should be a lowercased string, with underscores
       for whitespace.
    3. Exclude any characters other than a-z (lowercase) from the function name,
       replacing them with underscores.
    4. The final function name should not have more than one underscore together.

    >>> report_name_to_report_query_tool_function_name('Earnings Calls')
    'query_earnings_calls'
    >>> report_name_to_report_query_tool_function_name('COVID-19   Statistics')
    'query_covid_19_statistics'
    >>> report_name_to_report_query_tool_function_name('2023 Market Report!!!')
    'query_2023_market_report'
    """
    # Replace non-letter characters with underscores and remove extra whitespaces
    name = re.sub(r"[^a-z\d]", "_", name.lower())
    # Replace whitespace with underscores and remove consecutive underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")

    return f"query_{name}"


def report_details_to_report_query_tool_description(name: str, table_info: str) -> str:
    """
    Converts a set of chunks to a direct retriever tool description.
    """
    description = (
        "Pass the user's question, after rewriting it to be self-contained based on chat history, as input directly to this tool. "
        + f"Internally, it has logic to translate the question to a SQL query over the {name} report, run it, and return the result. "
        + "Do NOT pass SQL as input to the tool even if you think you know it (trust the tool to translate the question to SQL). "
        + f"Use this tool if you think the answer can be calculated via SQL query on the following table.\n\n{table_info}"
    )

    return description[:2048]  # cap to avoid failures when the description is too long


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
    DROP_COLUMNS = ["FileId", "Link to Document"]
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
    embeddings: Embeddings,
    sql_fixup_examples_file: Optional[Path] = None,
    sql_examples_file: Optional[Path] = None,
) -> Optional[BaseTool]:
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

    return CustomReportRetrievalTool(
        db=db,
        chain=sql_result_chain,
        name=retrieval_tool_function_name,
        description=retrieval_tool_description,
    )
