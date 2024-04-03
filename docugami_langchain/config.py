________SINGLE_TOKEN_LINE________ = "----------------"

# Lengths are in terms of characters, 1 token ~= 4 chars in English
# Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them

# Chunks and docs below this length are not summarized by default
MIN_LENGTH_TO_SUMMARIZE: int = 2048

# When summarizing full docs we cut off input after this by default
MAX_FULL_DOCUMENT_TEXT_LENGTH: int = int(1024 * 4 * 14)  # ~14k tokens,

# When summarizing chunks we cut off input after this by default
MAX_CHUNK_TEXT_LENGTH: int = int(1024 * 4 * 4.5)  # ~4.5k tokens
MAX_PARAMS_CUTOFF_LENGTH_CHARS: int = int(1024 * 4 * 2)  # ~2k tokens
DEFAULT_EXAMPLES_PER_PROMPT = 3

DEFAULT_SAMPLE_ROWS_IN_TABLE_INFO = 3

# Control tabular presentation of rows in SQL query prompts
DEFAULT_SAMPLE_ROWS_GRID_FORMAT = "grid"  # format from the tabulate library
DEFAULT_TABLE_AS_TEXT_CELL_MAX_WIDTH = 64  # cell width
DEFAULT_TABLE_AS_TEXT_CELL_MAX_LENGTH = 64 * 3  # cell content length

DEFAULT_RETRIEVER_K: int = 24
INCLUDE_XML_TAGS = True

DEFAULT_RECURSION_LIMIT = 35
