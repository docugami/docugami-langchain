from __future__ import annotations

import sqlparse
from langchain_core.documents import Document


def format_document_list(docs: list[Document], document_content_key: str) -> str:
    if not docs:
        raise Exception("No docs provided")

    formatted_output = ""
    for doc in docs:
        source = None
        if doc.metadata:
            source = doc.metadata.get("source")
        formatted_output += "\n\n****************"
        if source:
            formatted_output += f"\nDOCUMENT NAME: {source}"
        formatted_output += f"\n{document_content_key}:\n\n{doc.page_content}"

    return formatted_output


def formatted_summaries(docs: list[Document]) -> str:
    return format_document_list(docs, document_content_key="DOCUMENT SUMMARY")


def table_name_from_sql_create(create_sql: str) -> str:
    """
    Given a CREATE TABLE sql statement, returns the name of the table

    Parameters:
    - create_sql (str): The input string containing the create table sql.

    Returns:
    - str: Extracted table name, else empty string.

    >>> table_name_from_sql_create("CREATE TABLE example_table (id INT, name VARCHAR(100))")
    'example_table'

    >>> table_name_from_sql_create('CREATE TABLE "Corporate Charters" ("File" TEXT, "Link to Document" TEXT, "FILED Date" TEXT)')
    'Corporate Charters'
    """

    # Parse the SQL statement
    parsed_statements = sqlparse.parse(create_sql)

    # Assume first statement is the CREATE TABLE statement
    create_stmt = parsed_statements[0]

    # Look for the token representing the table name
    table_keyword_seen = False
    for token in create_stmt.tokens:
        if token.is_whitespace:
            continue

        if token.is_keyword and token.normalized.lower() == "table":
            table_keyword_seen = True
        elif table_keyword_seen and token.normalized.lower() is not None:
            return token.get_real_name()

    return ""


def replace_table_name_in_select(select_sql: str, new_table_name: str) -> str:
    """
    Given a SQL SELECT statement, updates the name of the table in it.

    Parameters:
    - select_sql (str): The input string containing a SELECT statement
    - new_table_name (str): New table name to update in the SELECT statement

    Returns:
    - str: Updated SELECT statement with new table name

    >>> replace_table_name_in_select('SELECT "Client" FROM "Report_Services_preview.xlsx" ORDER BY "Excess Liability Umbrella Coverage" DESC LIMIT 1', "Service Agreements Summary")
    'SELECT "Client" FROM "Service Agreements Summary" ORDER BY "Excess Liability Umbrella Coverage" DESC LIMIT 1'

    >>> replace_table_name_in_select('SELECT "Term Expiry (clean)" FROM "Foo" WHERE LOWER("Client") LIKE "%medcore%', "SaaS Contracts")
    'SELECT "Term Expiry (clean)" FROM "SaaS Contracts" WHERE LOWER("Client") LIKE "%medcore%'
    """

    # Parse the SQL statement
    parsed_statements = sqlparse.parse(select_sql)

    # Assume first statement is the SELECT statement
    select_stmt = parsed_statements[0]

    # Iterate over tokens and replace table name
    from_keyword_seen = False
    for token in select_stmt.tokens:
        if token.is_whitespace:
            continue

        if token.is_keyword and token.normalized.lower() == "from":
            from_keyword_seen = True
        elif from_keyword_seen and token.normalized.lower() is not None:
            token.tokens[0].value = f'"{new_table_name}"'
            break

    return str(select_stmt)
