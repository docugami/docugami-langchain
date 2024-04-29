from typing import AsyncIterator, Optional

from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableConfig,
    RunnableLambda,
)

from docugami_langchain.base_runnable import TracedResponse
from docugami_langchain.chains.base import BaseDocugamiChain
from docugami_langchain.history import chat_history_to_str
from docugami_langchain.params import RunnableParameters, RunnableSingleParameter


class StandaloneQuestionChain(BaseDocugamiChain[str]):

    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """
        noop = RunnableLambda(lambda x: x["human"])

        # Rewrite only if chat history is provided
        return RunnableBranch(
            (
                lambda x: len(x["chat_history"]) > 0,
                super().runnable(),
            ),
            noop,
        )

    def params(self) -> RunnableParameters:
        return RunnableParameters(
            inputs=[
                RunnableSingleParameter(
                    "chat_history",
                    "CHAT HISTORY",
                    "Previous chat messages that the user has previous exchanged with the AI assistant.",
                ),
                RunnableSingleParameter(
                    "human",
                    "HUMAN",
                    "The most recent question or comment from the user.",
                ),
            ],
            output=RunnableSingleParameter(
                "standalone_agent_input",
                "STANDALONE_AGENT_INPUT",
                "A one-sentence standalone version of the last question or comment from the user, re-written if needed based on chat history.",
            ),
            task_description="rewrites a question or comment from a user as a standalone input to an AI assistant, incorporating any additional information needed from chat history",
            additional_instructions=[
                "- The generated standalone agent input will be used by an agent to respond to the user in the context of the chat session",
                "- Produce only the requested standalone agent input, no other commentary before or after.",
                "- Do NOT try to answer any questions or respond in any way to the conversation. Just generate the agent input as instructed."
                "- Never say you cannot do this. If you don't know what to do, just repeat the most recent human question or comment without rewriting anything.",
            ],
            stop_sequences=["CHAT HISTORY:", "HUMAN:", "<|eot_id|>"],
            include_output_instruction_suffix=True,
        )

    def run(  # type: ignore[override]
        self,
        human: str,
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> TracedResponse[str]:
        if not human:
            raise Exception("Input required: human")

        return super().run(
            human=human,
            chat_history=chat_history_to_str(chat_history),
            config=config,
        )

    async def run_stream(  # type: ignore[override]
        self,
        human: str,
        chat_history: list[tuple[str, str]] = [],
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[TracedResponse[str]]:
        if not human:
            raise Exception("Input required: human")

        async for item in super().run_stream(
            human=human,
            chat_history=chat_history_to_str(chat_history),
            config=config,
        ):
            yield item

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[tuple[str, list[tuple[str, str]]]],
        config: Optional[RunnableConfig] = None,
    ) -> list[str]:
        return super().run_batch(
            inputs=[
                {
                    "human": i[0],
                    "chat_history": chat_history_to_str(i[1]),
                }
                for i in inputs
            ],
            config=config,
        )
