# flake8: noqa: E501

from langchain_docugami.prompts.core import SYSTEM_MESSAGE_CORE

CREATE_DIRECT_RETRIEVAL_TOOL_SYSTEM_MESSAGE = f"""{SYSTEM_MESSAGE_CORE}
You will be asked to write short descriptions of document types, given a particular sample document
as a guide. You ALWAYS follow these rules when generating descriptions:

- Make sure your description is text only, regardless of any markup in the given sample document.
- The generated description must apply to all documents of the given type, similar to the sample
  document given, not just the exact same document.
- The generated description will be used to describe this type of document in general in a product. When users ask
  a question, an AI agent will use the description you produce to decide whether the
  answer for that question is likely to be found in this type of document or not.
- Do NOT include any data or details from this particular sample document but DO use this sample
  document to get a better understanding of what types of information this type of document might contain.
- The generated description should be very short and up to 2 sentences max.

"""

CREATE_DIRECT_RETRIEVAL_TOOL_DESCRIPTION_PROMPT = """Here is a snippet from a sample document of type {docset_name}:

{document}

Please write a short general description of the given document type, using the given sample as a guide.

Respond only with the requested general description of the document type and no other language before or after.
"""
