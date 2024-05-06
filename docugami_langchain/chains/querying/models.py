from typing import TypedDict


class ExplainedSQLResult(TypedDict):
    sql_query: str
    explained_sql_query: str
    sql_result: str
    explained_sql_result: str


class ExplainedSQLQuestionResult(TypedDict):
    question: str
    result: ExplainedSQLResult
