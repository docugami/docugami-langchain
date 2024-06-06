# flake8: noqa: E501

from dataclasses import dataclass
from pathlib import Path

from tests.common import TEST_DATA_DIR
from tests.conftest import is_core_tests_only_mode

CHARTERS_SUMMARY_DATA_FILE = TEST_DATA_DIR / "xlsx/Charters Summary.xlsx"
CHARTERS_SUMMARY_TABLE_NAME = "Corporate Charters"

SAAS_CONTRACTS_DATA_FILE = TEST_DATA_DIR / "xlsx/SaaS Contracts Report.xlsx"
SAAS_CONTRACTS_TABLE_NAME = "SaaS Contracts"

FINANCIAL_SAMPLE_DATA_FILE = TEST_DATA_DIR / "xlsx/Financial Sample.xlsx"
FINANCIAL_SAMPLE_TABLE_NAME = "Financial Data"

DEMO_MSA_SERVICES_DATA_FILE = TEST_DATA_DIR / "xlsx/Report Services_preview.xlsx"
DEMO_MSA_SERVICES_TABLE_NAME = "Service Agreements Summary"

AVIATION_INCIDENTS_DATA_FILE = TEST_DATA_DIR / "xlsx/Aviation Incidents Report.xlsx"
AVIATION_INCIDENTS_TABLE_NAME = "Aviation Incidents Report"

DATA_TYPE_TEST_DATA_FILE = TEST_DATA_DIR / "xlsx/Data Type Test.xlsx"
DATA_TYPE_TEST_TABLE_NAME = "Data Type Test"


@dataclass
class TestReportData:
    data_file: Path
    name: str


@dataclass
class QueryTestData:
    report: TestReportData
    question: str
    sql_query: str
    sql_result: str
    explained_result_answer_fragments: list[str]
    explained_sql_query_fragments: list[str]
    is_core_test: bool = False


QUERY_TEST_DATA: list[QueryTestData] = [
    QueryTestData(
        report=TestReportData(
            data_file=CHARTERS_SUMMARY_DATA_FILE,
            name=CHARTERS_SUMMARY_TABLE_NAME,
        ),
        question="When was the cardiva medical charter filed?",
        sql_query='SELECT "Corporation Name", "FILED Date" FROM "Corporate Charters" WHERE LOWER("Corporation Name") LIKE "%cardiva medical%"',
        sql_result="[('Cardiva Medical, Inc.', '12/19/2017')]",
        explained_result_answer_fragments=["cardiva", "12/19/2017", "december 19"],
        explained_sql_query_fragments=["cardiva", "filed"],
        is_core_test=True,
    ),
    QueryTestData(
        report=TestReportData(
            data_file=CHARTERS_SUMMARY_DATA_FILE,
            name=CHARTERS_SUMMARY_TABLE_NAME,
        ),
        question="What are the top 5 companies that issued the most common stock?",
        sql_query='SELECT "Corporation Name", "Shares of Common Stock" FROM "Corporate Charters" ORDER BY "Shares of Common Stock" DESC LIMIT 5',
        sql_result="[('Clearstory Data Inc.', 150000000), ('Cardiva Medical, Inc.', 112000000), ('Avi Networks, Inc.', 110000000), ('Bugcrowd Inc.', 108787009), ('Aisera, Inc.', 90000000)]",
        explained_result_answer_fragments=[
            "clearstory",
            "cardiva",
            "avi networks",
            "bugcrowd",
            "aisera",
        ],
        explained_sql_query_fragments=["top", "common stock"],
        is_core_test=True,
    ),
    QueryTestData(
        report=TestReportData(
            data_file=DEMO_MSA_SERVICES_DATA_FILE,
            name=DEMO_MSA_SERVICES_TABLE_NAME,
        ),
        question="Which client has the highest liability cap?",
        sql_query="""SELECT "Client", "Excess Liability Umbrella Coverage" FROM "Service Agreements Summary" ORDER BY "Excess Liability Umbrella Coverage ($)" DESC LIMIT 1""",
        sql_result="[('Inity, Inc.', 'Excess Liability/Umbrella coverage with a limit of no less than $9,000,000 per occurrence and in the aggregate (such limit may be achieved through increase of limits in underlying policies to reach the level of coverage shown here). This policy shall name Client as an additional insured with...')]",
        explained_result_answer_fragments=["$10,000,000", "inity"],
        explained_sql_query_fragments=[],
    ),
    QueryTestData(
        report=TestReportData(
            data_file=DEMO_MSA_SERVICES_DATA_FILE,
            name=DEMO_MSA_SERVICES_TABLE_NAME,
        ),
        question="Which client has the lowest initial payment?",
        sql_query="""SELECT "Client", "Initial Payment" FROM "Service Agreements Summary" ORDER BY "Initial Payment" ASC LIMIT 1""",
        sql_result="[('Inity, Inc.', 'Fifteen Thousand Dollars ($15,000)')]",
        explained_result_answer_fragments=["$15,000", "inity"],
        explained_sql_query_fragments=[],
    ),
    QueryTestData(
        report=TestReportData(
            data_file=DEMO_MSA_SERVICES_DATA_FILE,
            name=DEMO_MSA_SERVICES_TABLE_NAME,
        ),
        question="Which client's contract expires the earliest?",
        sql_query='SELECT "Client", "Completion of the Services by Company Date" FROM "Service Agreements Summary" ORDER BY "Completion of the Services by Company Date" LIMIT 1',
        sql_result="[('Bioplex, Inc.', 'February 15, 2022')]",
        explained_result_answer_fragments=[
            "february 15, 2022",
            "bioplex",
        ],
        explained_sql_query_fragments=[],
    ),
    QueryTestData(
        report=TestReportData(
            data_file=DEMO_MSA_SERVICES_DATA_FILE,
            name=DEMO_MSA_SERVICES_TABLE_NAME,
        ),
        question="What are the typical working hours?",
        sql_query='SELECT "A Daily Basis" FROM "Service Agreements Summary" LIMIT 5',
        sql_result="[('Onsite Services. 2.1 Onsite visits will be charged on a daily basis (minimum 8 hours). 2.2 Time and expenses will be charged based on actuals unless otherwise described in an Order Form or accompanying SOW. 2.3 All work will be executed during regular working hours Monday-Friday 0800-1900. For...',), ('Onsite Services. 2.1 Onsite visits will be charged on a daily basis (minimum 8 hours). 2.2 Time and expenses will be charged based on actuals unless otherwise described in an Order Form or accompanying SOW. 2.3 All work will be executed during regular working hours Monday-Friday 0800-1900. For...',), ('Onsite Services. 2.1 Onsite visits will be charged on a daily basis (minimum 8 hours). 2.2 Time and expenses will be charged based on actuals unless otherwise described in an Order Form or accompanying SOW. 2.3 All work will be executed during regular working hours Monday-Friday 0800-1900. For...',), ('Onsite Services. 2.1 Onsite visits will be charged on a daily basis (minimum 8 hours). 2.2 Time and expenses...')]",
        explained_result_answer_fragments=["hours"],
        explained_sql_query_fragments=[],
    ),
    QueryTestData(
        report=TestReportData(
            data_file=DEMO_MSA_SERVICES_DATA_FILE,
            name=DEMO_MSA_SERVICES_TABLE_NAME,
        ),
        question="What is the highest workers comp amount?",
        sql_query='SELECT MAX("Workers Compensation Insurance") FROM "Service Agreements Summary"',
        sql_result="[('Blackzim, Inc.', '$2,500,000',)]",
        explained_result_answer_fragments=["blackzim", "$2,500,000"],
        explained_sql_query_fragments=[],
    ),
    QueryTestData(
        report=TestReportData(
            data_file=DEMO_MSA_SERVICES_DATA_FILE,
            name=DEMO_MSA_SERVICES_TABLE_NAME,
        ),
        question="Describe our onsite services",
        sql_query="""SELECT * FROM "Service Agreements Summary" WHERE "A Daily Basis" LIKE \'%onsite%\' LIMIT 5""",
        sql_result="""[(\'Master Services Agreement - Vector Color.pdf\', \'Open in Docugami\', \'$1,000,000\', \'$1,000,000\', \'$2,000,000\', \'$2,500,000\', \'$5,000,000\', \'June 15, 2021\', \'June 15, 2022\', \'two hundred percent (200%)\', \'One Hundred Thousand Dollars ($100,000)\', \'November 30, 2025\', \'30%\', \'Magicsoft, Inc.\', \'Vector, Inc.\', \'8741 Hamlet Ave S, Seattle, WA 98118\', \'MS TER SERVICES EE MENT MagicSoft, Inc. 1648 NW Market St Suite 500 Seattle, WA 98107 555.555.0125 STANDARD SOFTWARE AND SERVICES AGREEMENT AGREEMENT This Services Agreement (the “Agreement”) sets forth terms under which Magicsoft, Inc. a Washington Corporation (“Company”) located at 600 4th...\', \'Warranty Disclaimer. EXCEPT FOR THE WARRANTIES SET FORTH IN THIS AGREEMENT AND ANY SOW, EACH PARTY EXPRESSLY DISCLAIMS ANY AND ALL OTHER WARRANTIES OF ANY KIND OR NATURE, WHETHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A...\', \'Independently Developed Intellectual Prope...""",
        explained_result_answer_fragments=[],
        explained_sql_query_fragments=[],
    ),
    QueryTestData(
        report=TestReportData(
            data_file=FINANCIAL_SAMPLE_DATA_FILE,
            name=FINANCIAL_SAMPLE_TABLE_NAME,
        ),
        question="What were the total midmarket gross sales for Mexico in 2014?",
        sql_query="""SELECT SUM("Gross Sales") FROM "Financial Data" WHERE Segment = "Midmarket" AND Country = "Mexico" AND Year = 2014""",
        sql_result='"[(451890.0,)]"',
        explained_result_answer_fragments=["gross sales", "mexico", "2014", "451,890"],
        explained_sql_query_fragments=["gross sales", "mexico", "2014"],
    ),
    QueryTestData(
        report=TestReportData(
            data_file=FINANCIAL_SAMPLE_DATA_FILE,
            name=FINANCIAL_SAMPLE_TABLE_NAME,
        ),
        question="What types of questions can I ask?",
        sql_query='SELECT * FROM "Financial Data" LIMIT 1',
        sql_result="[('Government', 'Canada', 'Carretera', None, 1618.5, 3, 20, 32370.0, 0.0, 32370.0, 16185.0, 16185.0, '2014-01-01 00:00:00', 1, 'January', 2014)]",
        explained_result_answer_fragments=[],
        explained_sql_query_fragments=["country", "report", "column"],
    ),
    QueryTestData(
        report=TestReportData(
            data_file=FINANCIAL_SAMPLE_DATA_FILE,
            name=FINANCIAL_SAMPLE_TABLE_NAME,
        ),
        question="Invalid question blah blah",
        sql_query='SELECT * FROM "Financial Data" LIMIT 1',
        sql_result="[('Government', 'Canada', 'Carretera', None, 1618.5, 3, 20, 32370.0, 0.0, 32370.0, 16185.0, 16185.0, '2014-01-01T00:00:00', 1, 'January', 2014)]",
        explained_result_answer_fragments=[],
        explained_sql_query_fragments=[],
    ),
    QueryTestData(
        report=TestReportData(
            data_file=SAAS_CONTRACTS_DATA_FILE,
            name=SAAS_CONTRACTS_TABLE_NAME,
        ),
        question="When did the BioLumin contract start?",
        sql_query='SELECT "Effective Date" FROM "SaaS Contracts" WHERE LOWER("Client Name") LIKE "%medcore%"',
        sql_result="[('2017-07-01',)]",
        explained_result_answer_fragments=["2017", "july"],
        explained_sql_query_fragments=["biolumin", "start", "effective"],
    ),
    QueryTestData(
        report=TestReportData(
            data_file=SAAS_CONTRACTS_DATA_FILE,
            name=SAAS_CONTRACTS_TABLE_NAME,
        ),
        question="When does the Fiction Corp contract expire?",
        sql_query='SELECT "Term Expiry" FROM "SaaS Contracts" WHERE LOWER("Client Name") LIKE "%fiction corp%"',
        sql_result="",
        explained_result_answer_fragments=[],  # should yield an empty answer
        explained_sql_query_fragments=[],
    ),
    QueryTestData(
        report=TestReportData(
            data_file=SAAS_CONTRACTS_DATA_FILE,
            name=SAAS_CONTRACTS_TABLE_NAME,
        ),
        question="How many contracts have payment terms defined?",
        sql_query='SELECT COUNT("Client Name") FROM "SaaS Contracts" WHERE "Payment Terms" IS NOT NULL AND TRIM("Payment Terms") <> ;',
        sql_result="[(36,)]",
        explained_result_answer_fragments=["36", "thirty", "six"],
        explained_sql_query_fragments=["count", "total", "expiry"],
    ),
]

if is_core_tests_only_mode():
    QUERY_TEST_DATA = [t for t in QUERY_TEST_DATA if t.is_core_test]
