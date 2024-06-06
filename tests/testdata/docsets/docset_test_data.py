from dataclasses import dataclass, field
from typing import Optional

from tests.conftest import is_core_tests_only_mode
from tests.testdata.xlsx.query_test_data import (
    AVIATION_INCIDENTS_DATA_FILE,
    AVIATION_INCIDENTS_TABLE_NAME,
    TestReportData,
)


@dataclass
class TestQuestionData:
    question: str
    chat_history: list[tuple[str, str]] = field(default_factory=lambda: [])
    acceptable_answer_fragments: list[str] = field(default_factory=lambda: [])
    requires_report: bool = False
    acceptable_citation_label_fragments: list[str] = field(default_factory=lambda: [])


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
            data_file=AVIATION_INCIDENTS_DATA_FILE,
            name=AVIATION_INCIDENTS_TABLE_NAME,
        ),
        questions=[
            TestQuestionData(
                question="What is the accident number for the incident in Madill, OK?",
                acceptable_answer_fragments=["DFW08CA044"],
                acceptable_citation_label_fragments=["madill", "20080111X00040.xml"],
            ),
            TestQuestionData(
                question="How many accidents involved Cessna planes?",
                acceptable_answer_fragments=["27", "twenty", "seven"],
                requires_report=True,
                acceptable_citation_label_fragments=["cessna"],
            ),
            TestQuestionData(
                chat_history=[
                    (
                        "What is the largest city in Marshall county, OK?",
                        "Madill, OK is the largest city in Marshall County with a population of 4,094 in 2023. It's also the county seat.",
                    ),
                    (
                        "Do you know who it was named after?",
                        "It was named in honor of George Alexander Madill, an attorney for the St. Louis-San Francisco Railway.",
                    ),
                ],
                question="List the accident numbers and dates for all aviation incidents that happened there",
                acceptable_answer_fragments=["DFW08CA044", "N6135M", "Cessna"],
                acceptable_citation_label_fragments=["madill", "20080111X00040.xml"],
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
