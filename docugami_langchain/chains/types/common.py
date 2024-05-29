from enum import Enum
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel


class DataType(Enum):
    NUMBER = (
        "number"  # A predominantly numeric value, with or without text before/after
    )
    DATETIME = "datetime"  # A predominantly date and/or time value, with or without text before/after
    TEXT = "text"  # Generic unstructured text that is not one of the other types
    BOOL = "bool"  # A predominantly boolean (true/false or yes/no) value


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
