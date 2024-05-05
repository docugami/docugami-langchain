
from typing import TypedDict


class ExtendedSQLResult(TypedDict):
    question: str
    sql_query: str
    explained_sql_query: str
    sql_result: str
    explained_sql_result: str