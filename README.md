# docugami-langchain

This package contains the LangChain integrations for Docugami.

## Installation

To install with pip:

`pip install docugami-langchain`

To install with poetry:

`poetry add docugami-langchain`

## Development and Testing

1. To install with development and testing dependencies, install as `poetry install --with test,lint,typing,codespell`.
1. Add an API key:
    1. Ensure that you have a FIREWORKS_API_KEY set in your .env file.
    1. Ensure that you have OPENAI_API_KEY set in your .env file (platform.openai.com).
1. To send traces to langsmith, add the following to your .env file:
    1. LANGCHAIN_TRACING_V2=true
    1. LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    1. LANGCHAIN_API_KEY=... # get from LangSmith
    1. LANGCHAIN_PROJECT=alias-dev # change to your alias