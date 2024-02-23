import os
import warnings
from pathlib import Path
from typing import Optional

TEST_DATA_DIR = Path(__file__).parent / "testdata"
RAG_TEST_DGML_DOCSET_NAME = "NTSB Aviation Incident Reports"
RAG_TEST_DGML_DATA_DIR = TEST_DATA_DIR / "dgml_samples" / RAG_TEST_DGML_DOCSET_NAME
EXAMPLES_PATH = TEST_DATA_DIR / "examples"

CHARTERS_SUMMARY_DATA_FILE = TEST_DATA_DIR / "xlsx/Charters Summary.xlsx"
CHARTERS_SUMMARY_TABLE_NAME = "Corporate Charters"

SAAS_CONTRACTS_DATA_FILE = TEST_DATA_DIR / "xlsx/SaaS Contracts Report.xlsx"
SAAS_CONTRACTS_TABLE_NAME = "SaaS Contracts"

FINANCIAL_SAMPLE_DATA_FILE = TEST_DATA_DIR / "xlsx/Financial Sample.xlsx"
FINANCIAL_SAMPLE_TABLE_NAME = "Financial Data"

DEMO_MSA_SERVICES_DATA_FILE = TEST_DATA_DIR / "xlsx/Report Services_preview.xlsx"
DEMO_MSA_SERVICES_TABLE_NAME = "Service Agreements Summary"


def is_core_tests_only_mode() -> bool:
    core_tests_env_var = os.environ.get("DOCUGAMI_ONLY_CORE_TESTS")
    if not core_tests_env_var:
        return False
    else:
        if isinstance(core_tests_env_var, bool):
            return core_tests_env_var
        else:
            return str(core_tests_env_var).lower() == "true"


def verify_chain_response(
    response: Optional[str],
    match_fragment_str_options: list[str] = [],
    empty_ok: bool = False,
) -> None:
    if empty_ok and not response:
        return

    assert response
    if match_fragment_str_options:
        output_match = False
        for fragment in match_fragment_str_options:
            output_match = output_match or fragment.lower() in response.lower()

        assert (
            output_match
        ), f"{response} does not contain one of the expected output substrings {match_fragment_str_options}"

    # Check guardrails and warn if any violations detected based on string checks
    for banned_word in ["sql"]:
        if banned_word.lower() in response.lower():
            warnings.warn(
                UserWarning(f"Output contains banned word {banned_word}: {response}")
            )
