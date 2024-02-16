# Lengths are in terms of characters, 1 token ~= 4 chars in English
# Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them

# Chunks and docs below this length are not summarized by default
MIN_LENGTH_TO_SUMMARIZE: int = 2048

# When summarizing full docs we cut off input after this by default
MAX_FULL_DOCUMENT_TEXT_LENGTH: int = 1024 * 56  # ~14k tokens,

# When summarizing chunks we cut off input after this by default
MAX_CHUNK_TEXT_LENGTH: int = 1024 * 18  # ~4.5k tokens

MAX_PARAMS_CUTOFF_LENGTH_CHARS: int = 1024 * 8  # ~2k tokens
DEFAULT_EXAMPLES_PER_PROMPT = 3

RETRIEVER_K: int = 9
INCLUDE_XML_TAGS = True
