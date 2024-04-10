from dataclasses import dataclass, field
from typing import Optional

from tests.common import TEST_DATA_DIR
from tests.conftest import is_core_tests_only_mode
from tests.testdata.xlsx.query_test_data import TestReportData


@dataclass
class TestQuestionData:
    question: str
    chat_history: list[tuple[str, str]] = field(default_factory=lambda: [])
    acceptable_answer_fragments: list[str] = field(default_factory=lambda: [])
    requires_report: bool = False


@dataclass
class DocsetTestData:
    name: str
    is_core_test: bool = False
    questions: list[TestQuestionData] = field(default_factory=lambda: [])
    report: Optional[TestReportData] = None


DOCSET_TEST_DATA: list[DocsetTestData] = [
    DocsetTestData(
        name="Clinical Trial Protocols",
    ),
    DocsetTestData(
        name="Commercial Leases",
    ),
    DocsetTestData(
        name="Distributor Agreements",
    ),
    DocsetTestData(
        name="Employee Benefit Insurance Proposals",
    ),
    DocsetTestData(
        name="Issuer Notes",
    ),
    DocsetTestData(
        name="Master Services Agreements",
    ),
    DocsetTestData(
        name="Medical Notes (UNC)",
    ),
    DocsetTestData(
        is_core_test=True,
        name="NTSB Aviation Incident Reports",
        report=TestReportData(
            data_file=TEST_DATA_DIR / "xlsx/Aviation Incidents Report.xlsx",
            name="Aviation Incidents Report",
        ),
        questions=[
            TestQuestionData(
                question="What is the accident number for the incident in madill, oklahoma?",
                acceptable_answer_fragments=["DFW08CA044"],
            ),
            TestQuestionData(
                question="How many accidents involved Cessna aircraft?",
                acceptable_answer_fragments=["27", "twenty", "seven"],
                requires_report=True,
            ),
            TestQuestionData(
                chat_history=[
                    (
                        "What is the largest city in Marshall county, OK?",
                        "Madill is the largest city in Marshall County, Oklahoma, with a population of 4,094 in 2023. It's also the county seat.",
                    ),
                    (
                        "Do you know who it was named after?",
                        "It was named in honor of George Alexander Madill, an attorney for the St. Louis-San Francisco Railway.",
                    ),
                ],
                question="List the accident numbers for any aviation incidents that happened there",
                acceptable_answer_fragments=["DFW08CA044", "N6135M", "Cessna"],
            ),
        ],
    ),
    DocsetTestData(
        name="Non-Disclosure Agreements",
    ),
    DocsetTestData(
        name="Property and Casualty Insurance Proposals",
    ),
    DocsetTestData(
        name="Safety Data Sheets",
    ),
    DocsetTestData(
        name="Software License Agreements",
    ),
    DocsetTestData(
        name="Staffing Service Statements of Work",
    ),
    DocsetTestData(
        name="Supplier Agreements",
    ),
    DocsetTestData(
        name="Transport Agreements",
    ),
]

if is_core_tests_only_mode():
    DOCSET_TEST_DATA = [t for t in DOCSET_TEST_DATA if t.is_core_test]
