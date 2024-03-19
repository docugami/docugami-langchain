from langchain_core.documents import Document


def format_document_list(docs: list[Document], document_content_key: str) -> str:
    if not docs:
        raise Exception("No docs provided")

    formatted_output = ""
    for doc in docs:
        source = None
        if doc.metadata:
            source = doc.metadata.get("source")
        formatted_output += "\n\n****************"
        if source:
            formatted_output += f"\nDOCUMENT NAME: {source}"
        formatted_output += f"\n{document_content_key}:\n\n{doc.page_content}"

    return formatted_output


def formatted_summaries(docs: list[Document]) -> str:
    return format_document_list(docs, document_content_key="DOCUMENT SUMMARY")
