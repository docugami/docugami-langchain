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
        result = self.chain.run(question=question).get("sql_result")
        if result:
            return str(result)
        else:
            return ""


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
    table_info = re.sub(r"\s+", " ", table_info)
    description = (
        f"Given a single input 'question' parameter, generates and runs a SQL query over the {name}"
        + f" report, represented internally as the following SQL Table:\n\n{table_info}"
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

    # Write the table to the SQLite database
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    return conn


def connect_to_db(
    conn: sqlite3.Connection,
    sample_rows_in_table_info: int = 0,
) -> SQLDatabase:
    temp_db_file = tempfile.NamedTemporaryFile(suffix=".sqlite")
    with sqlite3.connect(temp_db_file.name) as disk_conn:
        conn.backup(disk_conn)  # dumps the connection to disk
    return SQLDatabase.from_uri(
        f"sqlite:///{temp_db_file.name}",
        sample_rows_in_table_info=sample_rows_in_table_info,
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

    return CustomReportRetrievalTool(
        db=db,
        chain=sql_result_chain,
        name=retrieval_tool_function_name,
        description=retrieval_tool_description,
        return_direct=True,
    )
