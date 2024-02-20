from dataclasses import dataclass, field
from typing import Optional

from langchain_core.runnables import Runnable

from docugami_langchain.config import DEFAULT_EXAMPLES_PER_PROMPT


@dataclass
class ChainSingleParameter:
    variable: str
    key: str
    description: str


@dataclass
class ChainParameters:
    inputs: list[ChainSingleParameter]
    output: ChainSingleParameter
    task_description: str
    additional_instructions: list[str]
    stop_sequences: list[str] = field(default_factory=lambda: ["<s>", "</s>"])
    num_examples: int = DEFAULT_EXAMPLES_PER_PROMPT
    additional_runnables: Optional[list[Runnable]] = None
    key_finding_output_parse: bool = True
