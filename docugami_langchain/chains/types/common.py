from enum import Enum
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel


class DataType(Enum):
    FLOAT = "float"  # A predominantly floating point value
    INTEGER = "integer"  # A predominantly integer value
    DATETIME = "datetime"  # A predominantly date and/or time value
    BOOL = "bool"  # A predominantly boolean (true/false or yes/no) value
    TEXT = "text"  # Generic unstructured text that is not one of the other types


class DataTypeWithUnit(BaseModel):
    """
    A data type with optional unit
    """

    type: DataType

    unit: Optional[str] = None

    def normalized_unit(self) -> str:
        if self.unit:
            return self.unit.strip().lower()
        return ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataTypeWithUnit):
            return NotImplemented

        # Compare type and (normalized) unit for equality
        return (self.type, self.normalized_unit()) == (
            other.type,
            other.normalized_unit(),
        )

    def __hash__(self) -> int:
        # Create a hash based on the type and normalized unit
        return hash((self.type, self.normalized_unit()))


class ParsedAddress(BaseModel):
    street: Optional[str]
    city: Optional[str]
    state: Optional[str]
    zip: Optional[str]
