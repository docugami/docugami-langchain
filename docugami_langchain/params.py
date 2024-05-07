from dataclasses import dataclass, field
from typing import Optional

from langchain_core.runnables import Runnable

from docugami_langchain.config import DEFAULT_EXAMPLES_PER_PROMPT


@dataclass
class RunnableSingleParameter:
    variable: str
    key: str
    description: str


@dataclass
class RunnableParameters:
    inputs: list[RunnableSingleParameter]
    output: RunnableSingleParameter
    task_description: str
    additional_instructions: list[str] = field(default_factory=lambda: [])
    stop_sequences: list[str] = field(default_factory=lambda: ["\n", "<|eot_id|>"])
    num_examples: int = DEFAULT_EXAMPLES_PER_PROMPT
    additional_runnables: Optional[list[Runnable]] = None
    key_finding_output_parse: bool = True
    include_output_instruction_suffix: bool = False


__all__ = ["RunnableSingleParameter", "RunnableParameters"]
