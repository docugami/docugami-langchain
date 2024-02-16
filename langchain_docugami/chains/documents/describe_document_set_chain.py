from typing import AsyncIterator, Optional, Tuple

from langchain_docugami.chains.base import BaseDocugamiChain, TracedChainResponse
from langchain_docugami.chains.params import ChainParameters, ChainSingleParameter


class DescribeDocumentSetChain(BaseDocugamiChain[str]):
    def chain_params(self) -> ChainParameters:
        return ChainParameters(
            inputs=[
                ChainSingleParameter(
                    "sample",
                    "SAMPLE",
                    "Snippet from a sample document",
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
                "A short general description of the given document type, using the given sample as a guide",
            ),
            task_description="creates a short description of a document type, given a particular sample document as a guide",
            additional_instructions=[
                "- Make sure your description is text only, regardless of any markup in the given sample document.",
                "- The generated description must apply to all documents of the given type, similar to the sample document given, not just the exact same document.",
                "- The generated description will be used to describe this type of document in general in a product. When users ask a question, an AI agent will use the description you produce to "
                + "decide whether the answer for that question is likely to be found in this type of document or not.",
                "- Do NOT include any data or details from this particular sample document but DO use this sample document to get a better understanding of what types of information this type of "
                + "document might contain.",
                "- The generated description should be very short and up to 2 sentences max.",
            ],
        )

    def run(  # type: ignore[override]
        self,
        sample: str,
        docset_name: str,
        config: Optional[dict] = None,
    ) -> str:
        if not sample or not docset_name:
            raise Exception("Inputs required: sample, docset_name")

        return super().run(
            sample=sample,
            docset_name=docset_name,
            config=config,
        )

    def run_stream(  # type: ignore[override]
        self,
        sample: str,
        docset_name: str,
        config: Optional[dict] = None,
    ) -> AsyncIterator[TracedChainResponse[str]]:
        if not sample or not docset_name:
            raise Exception("Inputs required: sample, docset_name")

        return super().run_stream(
            sample=sample,
            docset_name=docset_name,
            config=config,
        )

    def run_batch(  # type: ignore[override]
        self,
        inputs: list[Tuple[str, str]],
        config: Optional[dict] = None,
    ) -> list[str]:
        return super().run_batch(
            inputs=[
                {
                    "sample": i[0],
                    "docset_name": i[1],
                }
                for i in inputs
            ],
            config=config,
        )
