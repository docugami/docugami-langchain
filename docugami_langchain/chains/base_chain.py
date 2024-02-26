from typing import Optional

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
    StringPromptTemplate,
)

from docugami_langchain.base_runnable import (
    BaseRunnable,
    T,
    prompt_input_templates,
    standard_sytem_instructions,
)
from docugami_langchain.config import DEFAULT_EXAMPLES_PER_PROMPT
from docugami_langchain.params import RunnableParameters


def chain_system_prompt(params: RunnableParameters) -> str:
    """
    Constructs a system prompt for instruct models, suitable for running in chains with inputs and outputs specified in params.
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
        prompt += f"Given these inputs, please generate: {params.output.description}"

    return prompt


def chain_generic_string_prompt_template(
    params: RunnableParameters,
    example_selector: Optional[SemanticSimilarityExampleSelector] = None,
    num_examples: int = DEFAULT_EXAMPLES_PER_PROMPT,
) -> StringPromptTemplate:
    """
    Constructs a string prompt template generically suitable for all models.
    """
    input_vars = [i.variable for i in params.inputs]

    if not example_selector:
        # Basic simple prompt template
        return PromptTemplate(
            input_variables=input_vars,
            template=(prompt_input_templates(params) + "\n" + params.output.key + ":"),
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
                template=prompt_input_templates(params)
                + f"\n{params.output.key}: {{{params.output.variable}}}",
            ),
            prefix="",
            suffix=(prompt_input_templates(params) + "\n" + params.output.key + ":"),
            input_variables=input_vars,
        )


def chain_chat_prompt_template(
    params: RunnableParameters,
    example_selector: Optional[SemanticSimilarityExampleSelector] = None,
    num_examples: int = DEFAULT_EXAMPLES_PER_PROMPT,
) -> ChatPromptTemplate:
    """
    Constructs a chat prompt template.
    """

    input_vars = [i.variable for i in params.inputs]

    human_message_body = prompt_input_templates(params)

    # Basic chat prompt template (with system instructions and optional chat history)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=chain_system_prompt(params)),
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
                        prompt_input_templates(params),
                    ),
                    ("ai", f"{{{params.output.variable}}}"),
                ]
            ),
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=chain_system_prompt(params)),
                few_shot_prompt,
                ("human", human_message_body),
            ]
        )

    return prompt_template


class BaseChainRunnable(BaseRunnable[T]):
    """
    Base class with common functionality for various chains.
    """

    def prompt(
        self,
        params: RunnableParameters,
        num_examples: int = DEFAULT_EXAMPLES_PER_PROMPT,
    ) -> BasePromptTemplate:
        if isinstance(self.llm, BaseChatModel):
            # For chat model instances, use chat prompts with
            # specially crafted system and few shot messages.
            return chain_chat_prompt_template(
                params=params,
                example_selector=self._example_selector,
                num_examples=min(num_examples, len(self._examples)),
            )
        else:
            # For non-chat model instances, we need a string prompt
            return chain_generic_string_prompt_template(
                params=params,
                example_selector=self._example_selector,
                num_examples=min(num_examples, len(self._examples)),
            )
