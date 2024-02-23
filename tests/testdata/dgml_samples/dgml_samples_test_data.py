from dataclasses import dataclass

from tests.common import is_core_tests_only_mode


@dataclass
class DGSamplesTestData:
    test_data_dir: str
    is_core_test: bool = False


DG_SAMPLE_TEST_DATA: list[DGSamplesTestData] = [
    DGSamplesTestData(
        is_core_test=True,
        test_data_dir="Clinical Trial Protocols",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Commercial Leases",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Distributor Agreements",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Employee Benefit Insurance Proposals",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Issuer Notes",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Master Services Agreements",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Medical Notes (UNC)",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="NTSB Aviation Incident Reports",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Non-Disclosure Agreements",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Property and Casualty Insurance Proposals",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Safety Data Sheets",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Software License Agreements",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Staffing Service Statements of Work",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Supplier Agreements",
    ),
    DGSamplesTestData(
        is_core_test=False,
        test_data_dir="Transport Agreements",
    ),
]

if is_core_tests_only_mode():
    DG_SAMPLE_TEST_DATA = [t for t in DG_SAMPLE_TEST_DATA if t.is_core_test]
