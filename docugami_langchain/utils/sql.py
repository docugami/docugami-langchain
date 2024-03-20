from typing import Any, Optional

import sqlglot
import sqlglot.expressions as exp
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.embeddings import Embeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.vectorstores import VectorStore
from sqlalchemy import Table, exc, select, text
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.schema import CreateTable
from tabulate import tabulate

from docugami_langchain.config import (
    DEFAULT_SAMPLE_ROWS_GRID_FORMAT,
    DEFAULT_SAMPLE_ROWS_IN_TABLE_INFO,
    DEFAULT_TABLE_AS_TEXT_CELL_MAX_WIDTH,
)


def first_table(db: SQLDatabase) -> Table:
    """Gets a reference to the first table in the db underlying this chain."""

    meta = db._metadata
    if not meta.sorted_tables:
        raise Exception("No tables in db")

    return meta.sorted_tables[0]


def sanitize_example_value(val: Any) -> str:
    clean_val = str(val) or ""
    clean_val = clean_val.strip()

    return clean_val


def create_example_selector(
    db: SQLDatabase,
    embeddings: Embeddings,
    examples_vectorstore_cls: type[VectorStore],
) -> SemanticSimilarityExampleSelector:
    """Reads all rows from the first table of the db and indexes them for few shot retrieval."""
    table = first_table(db)
    select_all_query = select(table)

    with db._engine.connect() as connection:
        result = connection.execute(select_all_query)

        example_rows = []
        for row in result:
            sanitized_row_dict = {}
            raw_row_dict = row._asdict()
            for key in raw_row_dict:
                sanitized_row_dict[key] = sanitize_example_value(raw_row_dict[key])
            example_rows.append(sanitized_row_dict)

        return SemanticSimilarityExampleSelector.from_examples(
            examples=example_rows,
            embeddings=embeddings,
            vectorstore_cls=examples_vectorstore_cls,
        )


def sample_rows(
    db: SQLDatabase,
    question: Optional[str] = None,
    example_selector: Optional[SemanticSimilarityExampleSelector] = None,
    included_sample_rows: int = DEFAULT_SAMPLE_ROWS_IN_TABLE_INFO,
    sample_row_grid_format: str = DEFAULT_SAMPLE_ROWS_GRID_FORMAT,
) -> str:
    """Gets sample rows from the given table, optionally via few shot comparison to the user question."""
    table = first_table(db)
    sample_rows = []
    with db._engine.connect() as connection:
        if example_selector and question:
            # Optimized, select via few shot selector
            example_selector.k = included_sample_rows
            similar_rows = example_selector.select_examples(
                {
                    "question": question,
                }
            )
            for row_dict in similar_rows:
                sample_rows.append(
                    [sanitize_example_value(row_dict[key]) for key in row_dict]
                )
        else:
            # Not-optimized, select top N
            select_command = select(table).limit(included_sample_rows)
            sample_rows_result = connection.execute(select_command)  # type: ignore
            for row in sample_rows_result:
                sample_rows.append([sanitize_example_value(val) for val in row])

    columns_names = [col.name for col in table.columns]
    grid_str = tabulate(
        sample_rows,
        headers=columns_names,
        tablefmt=sample_row_grid_format,
        maxcolwidths=DEFAULT_TABLE_AS_TEXT_CELL_MAX_WIDTH,
    )

    return (
        f"{len(sample_rows)} example rows from the table (consider these to better craft and format queries e.g. considering "
        + f"the format of typical values in the data when crafting LIKE statements):\n {grid_str}\n"
    )


def get_table_info(
    db: SQLDatabase,
    question: Optional[str] = None,
    example_selector: Optional[SemanticSimilarityExampleSelector] = None,
    included_sample_rows: int = DEFAULT_SAMPLE_ROWS_IN_TABLE_INFO,
    grid_format: str = DEFAULT_SAMPLE_ROWS_GRID_FORMAT,
    override_table_name: Optional[str] = None,
) -> str:
    """Gets the table info for the first table in the db underlying this chain."""

    table = first_table(db)
    if override_table_name:
        table.name = override_table_name

    create_table = str(CreateTable(table).compile(db._engine))
    table_info_str = f"{create_table.rstrip()}"
    sample_rows_str = sample_rows(
        db=db,
        question=question,
        example_selector=example_selector,
        included_sample_rows=included_sample_rows,
        sample_row_grid_format=grid_format,
    )

    table_info_str += "\n\n" + sample_rows_str

    return table_info_str


def ensure_query(db: SQLDatabase, sql_query: str) -> str:
    """Ensures the given query is syntactically correct, and contains columns and tables that actually exist in the db."""

    # Use sqlglot to parse and extract columns and tables from the query
    parsed_query = sqlglot.parse_one(sql_query)
    select_stmt = parsed_query.find(exp.Select)
    if not select_stmt:
        raise ValueError("Only SELECT statements are supported.")

    sql_query_columns = []
    for expression in select_stmt.args["expressions"]:
        if isinstance(expression, exp.Alias):
            sql_query_columns.append(expression.text("alias"))
        elif isinstance(expression, exp.Column):
            sql_query_columns.append(expression.text("this"))

    inspector = Inspector.from_engine(db._engine)

    # Check if tables and columns exist in the database
    for table_name in inspector.get_table_names():
        table_columns = [col["name"] for col in inspector.get_columns(table_name)]
        for col in sql_query_columns:
            if col not in table_columns:
                raise ValueError(
                    f"SQL Query column '{col}' does not exist in table '{table_name}' or is not accessible."
                )

    # Perform a dry-run to check syntax and table/column existence
    stmt = text(sql_query)
    with db._engine.connect() as conn:
        # We use a transaction and roll it back to avoid any side effects
        trans = conn.begin()
        try:
            conn.execute(stmt.execution_options(autocommit=False)).fetchall()
        except exc.DBAPIError as e:
            raise ValueError(f"Query failed due to database error: {e}")
        finally:
            trans.rollback()

    # If all checks pass, the query is valid against the db
    return sql_query
