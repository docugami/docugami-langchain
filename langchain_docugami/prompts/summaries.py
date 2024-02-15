# flake8: noqa: E501

from langchain_docugami.prompts.core import SYSTEM_MESSAGE_CORE

CREATE_FULL_DOCUMENT_SUMMARY_SYSTEM_MESSAGE = f"""{SYSTEM_MESSAGE_CORE}
You will be asked to summarize documents. You ALWAYS follow these rules when generating summaries:

- Your generated summary should be in the same format as the given document, using the same overall schema.
- The generated summary should be up to 1 page of text in length, or shorter if the original document is short.
- Only summarize, don't try to change any facts in the document even if they appear incorrect to you.
- Include as many facts and data points from the original document as you can, in your summary.
"""

CREATE_FULL_DOCUMENT_SUMMARY_PROMPT = """Here is a document, in {format} format:

{document}

Please write a detailed summary of the given document.

Respond only with the summary and no other language before or after.
"""

CREATE_CHUNK_SUMMARY_SYSTEM_MESSAGE = f"""{SYSTEM_MESSAGE_CORE}
You will be asked to summarize chunks of documents. You ALWAYS follow these rules when generating summaries:

- Your generated summary should be in the same format as the given document, using the same overall schema.
- The generated summary will be embedded and used to retrieve the raw text or table elements from a vector database.
- Only summarize, don't try to change any facts in the chunk even if they appear incorrect to you.
- Include as many facts and data points from the original chunk as you can, in your summary.
- Pay special attention to monetary amounts, dates, names of people and companies, etc and include in your summary.
"""

CREATE_CHUNK_SUMMARY_PROMPT = """Here is a chunk from a document, in {format} format:

{document}

Respond only with the summary and no other language before or after.
"""
