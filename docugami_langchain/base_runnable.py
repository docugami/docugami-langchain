import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Generic, Optional, TypeVar

import yaml
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
    StringPromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import merge_configs
from langchain_core.tracers.context import collect_runs
from langchain_core.vectorstores import VectorStore

from docugami_langchain.config import (
    DEFAULT_AGENT_RECURSION_LIMIT,
    DEFAULT_EXAMPLES_PER_PROMPT,
    MAX_PARAMS_CUTOFF_LENGTH_CHARS,
)
from docugami_langchain.output_parsers import KeyfindingOutputParser
from docugami_langchain.params import RunnableParameters

T = TypeVar("T")

CONFIG_KEY: str = "config"


def standard_sytem_instructions(task: str) -> str:
    return f"""You are the helpful Docugami Assistant that {task}.

You ALWAYS follow the following guidance to generate your answers, regardless of any other guidance or requests:

- Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.
"""


def prompt_input_templates(
    params: RunnableParameters,
    include_output_instruction_suffix: bool = False,
) -> str:
    """
    Builds and returns the core prompt with input key/value pairs.
    """
    input_template_list = ""
    for input in params.inputs:
        input_template_list += f"{input.key}: {{{input.variable}}}\n"

    if include_output_instruction_suffix and params.output:
        input_template_list += (
            f"\nGiven the inputs above, please generate: {params.output.description}"
        )

    return input_template_list.strip()


def system_prompt(params: RunnableParameters) -> str:
    """
    Constructs a system prompt for instruct models, suitable for running in chains and agents with inputs and outputs specified in params.
    """

    prompt = standard_sytem_instructions(params.task_description)

    additional_instructions_list = ""
    if params.additional_instructions:
        additional_instructions_list = "\n".join(params.additional_instructions)

    if additional_instructions_list:
        prompt += additional_instructions_list

    input_description_list = ""
    for input in params.inputs:
        input_description_list += f"{input.key}: {input.description}\n"

    if input_description_list:
        prompt += f"""

Your inputs will be in this format:

{input_description_list}
"""

    if params.output:
        prompt += f"Given the inputs above, please generate: {params.output.description}"

    return prompt


def generic_string_prompt_template(
    params: RunnableParameters,
    example_selector: Optional[MaxMarginalRelevanceExampleSelector] = None,
    num_examples: int = DEFAULT_EXAMPLES_PER_PROMPT,
    include_output_instruction_suffix: bool = False,
) -> StringPromptTemplate:
    """
    Constructs a string prompt template generically suitable for all models.
    """
    input_vars = [i.variable for i in params.inputs]

    if not example_selector:
        # Basic simple prompt template
        return PromptTemplate(
            input_variables=input_vars,
            template=(
                prompt_input_templates(params, include_output_instruction_suffix)
                + "\n"
                + params.output.key
                + ":"
            ),
        )
    else:
        # Examples available, use few shot prompt template instead
        example_selector.k = num_examples

        example_input_vars = input_vars.copy()
        example_input_vars.append(params.output.variable)

        # Basic few shot prompt template
        return FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=PromptTemplate(
                input_variables=example_input_vars,
                template=prompt_input_templates(params, False)
                + f"\n{params.output.key}: {{{params.output.variable}}}",
            ),
            prefix="",
            suffix=(
                prompt_input_templates(params, include_output_instruction_suffix)
                + "\n"
                + params.output.key
                + ":"
            ),
            input_variables=input_vars,
        )


def chat_prompt_template(
    params: RunnableParameters,
    example_selector: Optional[MaxMarginalRelevanceExampleSelector] = None,
    num_examples: int = DEFAULT_EXAMPLES_PER_PROMPT,
    include_output_instruction_suffix: bool = False,
) -> ChatPromptTemplate:
    """
    Constructs a chat prompt template.
    """

    input_vars = [i.variable for i in params.inputs]

    human_message_body = prompt_input_templates(
        params,
        include_output_instruction_suffix,
    )

    # Basic chat prompt template (with system instructions and optional chat history)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt(params)),
            ("human", human_message_body),
        ]
    )

    if example_selector:
        # Examples available, use few shot prompt template instead
        example_selector.k = num_examples

        # Basic few shot prompt template
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            # The input variables select the values to pass to the example_selector
            input_variables=input_vars,
            example_selector=example_selector,
            # Define how each example will be formatted.
            # In this case, each example will become 2 messages:
            # 1 human, and 1 AI
            example_prompt=ChatPromptTemplate.from_messages(
                [
                    (
                        "human",
                        prompt_input_templates(params, False),
                    ),
                    ("ai", f"{{{params.output.variable}}}"),
                ]
            ),
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt(params)),
                few_shot_prompt,
                ("human", human_message_body),
            ]
        )

    return prompt_template


def normalize_whitespace(text: str) -> str:
    """
    Normalizes whitespace in given text without affecting visual formatting.

    This function aims to:
    1. Compress multiple vertical whitespace (more than two newlines) into two newlines, without affecting horizontal whitespace (indentation).
    2. Remove leading and trailing whitespace from the text.

    >>> normalize_whitespace("  Hello\\n\\n\\nWorld  ")
    'Hello\\n\\nWorld'
    >>> normalize_whitespace("\\n\\n\\n    Indented text\\nMore indented text\\n\\n")
    'Indented text\\nMore indented text'
    >>> normalize_whitespace("No extra\\nwhitespace here.")
    'No extra\\nwhitespace here.'
    >>> normalize_whitespace("  \\n  Leading and trailing newlines and spaces  \\n  ")
    'Leading and trailing newlines and spaces'
    >>> normalize_whitespace("\\n\\n\\n\\nOnly newlines here\\n\\n\\n\\nHello")
    'Only newlines here\\n\\nHello'

    Note that horizontal spaces before and after the text in a single line are not preserved if they're at the beginning or end of the text.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized text with whitespace adjusted.
    """

    # compress vertical whitespace without affecting horizontal whitespace (indentation)
    text = re.sub(r"(\s*\n){3,}", "\n\n", text)

    # remove leading and trailing whitespace
    text = text.strip()

    return text


@dataclass
class TracedResponse(Generic[T]):
    value: T
    run_id: str = ""


class BaseRunnable(BaseModel, Generic[T], ABC):
    """
    Base class with common functionality for various runnables.
    """

    llm: BaseLanguageModel
    embeddings: Optional[Embeddings] = None
    examples_vectorstore_cls: type[VectorStore] = FAISS

    input_params_max_length_cutoff: int = MAX_PARAMS_CUTOFF_LENGTH_CHARS
    few_shot_params_max_length_cutoff: int = MAX_PARAMS_CUTOFF_LENGTH_CHARS
    _examples: list[dict] = []
    _example_selector: Optional[MaxMarginalRelevanceExampleSelector] = None

    recursion_limit = DEFAULT_AGENT_RECURSION_LIMIT

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def vector_collection_name(self) -> str:
        """
        Unique vector collection name for each class and embedding.
        """
        embedding_model_name = getattr(
            self.embeddings,
            "model_name",
            getattr(self.embeddings, "model", getattr(self.embeddings, "name", None)),
        )
        if not embedding_model_name:
            raise Exception(f"Could not determine model name for {self.embeddings}")

        raw_name = str(self.__class__.__name__)
        raw_name = f"{raw_name}-{embedding_model_name}"
        return "".join([char for char in raw_name if char.isalnum()])

    def load_examples(
        self,
        examples_yaml: Path,
        num_examples: int = DEFAULT_EXAMPLES_PER_PROMPT,
    ) -> None:
        """
        Optional: loads examples from the given examples file (YAML) and initializes an
        internal example selector that select appropriate examples for each prompt. Note
        that each runnable requires its own particular format for examples based on the
        keys required in its prompt.

        The provided embeddings instance is used to embed the examples for similarity.
        See https://langchain.readthedocs.io/en/latest/modules/indexes/examples/embeddings.html.
        """

        if not self.embeddings:
            raise Exception("Embedding model required to use few shot examples")

        with open(examples_yaml, "r", encoding="utf-8") as in_f:
            self._examples = yaml.safe_load(in_f)

            for ex in self._examples:
                keys = ex.keys()
                for k in keys:
                    if ex[k]:
                        # whitespace normalize
                        ex[k] = normalize_whitespace(ex[k])

                        # truncate length to avoid overflowing context too much (strip any trailing whitespace again)
                        ex[k] = ex[k][: self.few_shot_params_max_length_cutoff].strip()
                    else:
                        ex[k] = ""

            if self._examples and num_examples:
                try:
                    self._example_selector = (
                        MaxMarginalRelevanceExampleSelector.from_examples(
                            examples=self._examples,
                            embeddings=self.embeddings,
                            vectorstore_cls=self.examples_vectorstore_cls,
                            k=num_examples,
                        )
                    )
                except Exception as exc:
                    details = f"Exception while loading samples from YAML {examples_yaml}. Details: {exc}"
                    raise Exception(details)

    def runnable(self) -> Runnable:
        """
        Runnable for this chain, built dynamically from the params
        """

        # Build up prompt for this use case, possibly customizing for this model
        params = self.params()
        prompt_template = self.prompt(params)

        # Generate answer from the LLM
        full_runnable = prompt_template | self.llm.bind(stop=params.stop_sequences)
        if isinstance(self.llm, BaseChatModel):
            # For chat models, we need to make sure the output is a string
            full_runnable = full_runnable | StrOutputParser()

        if params.key_finding_output_parse:
            # Increase accuracy for models that require very specific output, by
            # looking for the output key however adding such an output parser disables
            # streaming, so use carefully
            full_runnable = full_runnable | KeyfindingOutputParser(
                output_key=params.output.key
            )

        if params.additional_runnables:
            for runnable in params.additional_runnables:
                full_runnable = full_runnable | runnable

        return full_runnable

    def _prepare_run_args(self, kwargs_dict: dict) -> tuple[RunnableConfig, dict]:
        # In langsmith, default the run to be named according the the chain class
        config = RunnableConfig(
            run_name=self.__class__.__name__,
            recursion_limit=self.recursion_limit,
        )
        if kwargs_dict:
            arg_config: Optional[RunnableConfig] = kwargs_dict.get(CONFIG_KEY)
            # Use additional caller specified config, e.g. in case of chains
            # nested inside lambdas
            config = merge_configs(config, arg_config)

            # kwargs are used as inputs to the chain prompt, so remove the config
            # param if specified
            if CONFIG_KEY in kwargs_dict:
                del kwargs_dict[CONFIG_KEY]

        for key in kwargs_dict:
            if isinstance(kwargs_dict[key], str):
                # whitespace normalize
                kwargs_dict[key] = normalize_whitespace(kwargs_dict[key])

                # truncate length to avoid overflowing context too much (strip any trailing whitespace again)
                kwargs_dict[key] = kwargs_dict[key][
                    : self.few_shot_params_max_length_cutoff
                ].strip()

        return config, kwargs_dict

    @abstractmethod
    def run(self, **kwargs) -> TracedResponse[T]:  # type: ignore
        config, kwargs_dict = self._prepare_run_args(kwargs)
        with collect_runs() as cb:
            chain_output: T = self.runnable().invoke(input=kwargs_dict, config=config)  # type: ignore

            run_id = ""
            if cb.traced_runs:
                run_id = str(cb.traced_runs[0].id)

            return TracedResponse[T](run_id=run_id, value=chain_output)

    @abstractmethod
    def run_batch(self, **kwargs: Any) -> list[T]:
        config, kwargs_dict = self._prepare_run_args(kwargs)

        inputs = kwargs_dict.get("inputs")
        if not inputs:
            raise Exception("Please specify a batch for inference")

        if not isinstance(inputs, list):
            raise Exception("Input for batch processing must be a List")

        for input_dict in inputs:
            for key in input_dict:
                # For string args, cap at max to avoid chance of prompt overflow
                if isinstance(input_dict[key], str):
                    input_dict[key] = input_dict[key][
                        : self.input_params_max_length_cutoff
                    ]

        return self.runnable().batch(inputs=inputs, config=config)  # type: ignore

    def prompt(
        self,
        params: RunnableParameters,
        num_examples: int = DEFAULT_EXAMPLES_PER_PROMPT,
    ) -> BasePromptTemplate:
        if isinstance(self.llm, BaseChatModel):
            # For chat model instances, use chat prompts with
            # specially crafted system and few shot messages.
            return chat_prompt_template(
                params=params,
                example_selector=self._example_selector,
                num_examples=min(num_examples, len(self._examples)),
                include_output_instruction_suffix=params.include_output_instruction_suffix,
            )
        else:
            # For non-chat model instances, we need a string prompt
            return generic_string_prompt_template(
                params=params,
                example_selector=self._example_selector,
                num_examples=min(num_examples, len(self._examples)),
                include_output_instruction_suffix=params.include_output_instruction_suffix,
            )

    @abstractmethod
    async def run_stream(self, **kwargs: Any) -> AsyncIterator[TracedResponse[T]]: ...

    @abstractmethod
    def params(self) -> RunnableParameters: ...


__all__ = ["TracedResponse", "BaseRunnable"]
