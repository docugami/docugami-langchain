from enum import Enum
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel


class DataTypes(Enum):
    INT = "integer"  # A predominantly numeric value, with or without text before/after
    DATETIME = "datetime"  # A predominantly date and/or time value, with or without text before/after
    TEXT = "text"  # Generic unstructured text that is not one of the other types


class DocugamiDataType(BaseModel):
    """
    A data type with optional unit
    """

    type: DataTypes

    unit: Optional[str]
