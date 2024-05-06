________SINGLE_TOKEN_LINE________ = "----------------"

# ************ PARAMETERS TO CONTROL PROMPTS

# Lengths are in terms of characters, 1 token ~= 4 chars in English
# Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them

# Chunks and docs below this length are not summarized by default
MIN_LENGTH_TO_SUMMARIZE: int = 2048

# Change this to improve parallelization, or work around late limiting issues
BATCH_SIZE = 8

# When summarizing full docs we cut off input after this by default
MAX_FULL_DOCUMENT_TEXT_LENGTH: int = int(1024 * 4 * 8)  # ~8k tokens,

# When summarizing chunks we cut off input after this by default
MAX_CHUNK_TEXT_LENGTH: int = int(1024 * 4 * 4.5)  # ~4.5k tokens
MAX_PARAMS_CUTOFF_LENGTH_CHARS: int = int(1024 * 4 * 2)  # ~2k tokens
DEFAULT_EXAMPLES_PER_PROMPT = 3

# ************ PARAMETERS TO CONTROL RETRIEVAL
INCLUDE_XML_TAGS = True
# The number of results retrieved (before re-ranking / filtering)
DEFAULT_RETRIEVER_K: int = 24
# Results above this percentile are kept, others are rejected (0 means keep them all, 90 means keep only very good ones, 100 means keep the top one, etc.)
RE_RANK_FILTER_PERCENTILE: int = 70

# ************ PARAMETERS TO TABULAR REPRESENTATION OF ROWS TO HELP GENERATE SQL
DEFAULT_SAMPLE_ROWS_IN_TABLE_INFO = 3
DEFAULT_SAMPLE_ROWS_GRID_FORMAT = "grid"  # format from the tabulate library
DEFAULT_TABLE_AS_TEXT_CELL_MAX_WIDTH = 64  # cell width
DEFAULT_TABLE_AS_TEXT_CELL_MAX_LENGTH = 64 * 3  # cell content length

# ************ PARAMETERS TO CONTROL AGENT BEHAVIOR
DEFAULT_AGENT_RECURSION_LIMIT = 70
