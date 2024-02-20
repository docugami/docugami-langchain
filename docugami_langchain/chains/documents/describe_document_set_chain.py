from operator import itemgetter
from typing import AsyncIterator, List, Optional

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from docugami_langchain.chains.base import BaseDocugamiChain, TracedChainResponse
from docugami_langchain.chains.helpers import formatted_summaries
from docugami_langchain.chains.params import ChainParameters, ChainSingleParameter


class DescribeDocumentSetChain(BaseDocugamiChain[str]):
    def runnable(self) -> Runnable:
        """
        Custom runnable for this chain.
        """

        return {
            "summaries": itemgetter("summaries") | RunnableLambda(formatted_summaries),
            "docset_name": itemgetter("docset_name"),
        } | super().runnable()

    def chain_params(self) -> ChainParameters:
        return ChainParameters(
            inputs=[
                ChainSingleParameter(
                    "summaries",
                    "SUMMARIES",
                    "Summaries of representative documents from a set of documents",
                ),
                ChainSingleParameter(
                    "docset_name",
                    "DOCSET NAME",
                    "A user entered description for this type of document",
                ),
            ],
            output=ChainSingleParameter(
                "description",
                "DESCRIPTION",
                "A short general description of the given document type, using the given summaries as a guide",
            ),
            task_description="creates a short description of a document type, given a some sample documents as a guide",
            additional_instructions=[
                "- Make sure your description is text only, regardless of any markup in the given sample documents.",
                "- The generated description must apply to all documents of the given type, similar to the sample documents given, not just the exact same document.",
                "- The generated description will be used to describe this type of document in general in a product. When users ask a question, an AI agent will use the description you produce to "
                + "decide whether the answer for that question is likely to be found in this type of document or not.",
                "- Do NOT include any data or details from these particular sample documents but DO use these sample documents to get a better understanding of what types of information this type of "
                + "document might contain.",
                "- The generated description should be very short and up to 2 sentences max.",
            ],
        )

    def run(  # type: ignore[override]
        self,
        summaries: List[Document],
        docset_name: str,
        config: Optional[dict] = None,
    ) -> str:
        if not summaries or not docset_name:
            raise Exception("Inputs required: summaries, docset_name")

        return super().run(
            summaries=summaries,
            docset_name=docset_name,
            config=config,
        )

    def run_stream(  # type: ignore[override]
        self,
        summaries: List[Document],
        docset_name: str,
        config: Optional[dict] = None,
    ) -> AsyncIterator[TracedChainResponse[str]]:
        if not summaries or not docset_name:
            raise Exception("Inputs required: summaries, docset_name")

        return super().run_stream(
            summaries=summaries,
            docset_name=docset_name,
            config=config,
        )

    def run_batch(  # type: ignore[override]
        self,
    ) -> list[str]:
        raise NotImplementedError()
