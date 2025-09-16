from pathlib import Path

import pytest

from sortition_algorithms import core
from sortition_algorithms.adapters import CSVAdapter
from sortition_algorithms.committee_generation import GUROBI_AVAILABLE
from sortition_algorithms.core import run_stratification
from sortition_algorithms.settings import Settings

# only test leximin if gurobipy is available
ALGORITHMS = ("legacy", "maximin", "leximin", "nash") if GUROBI_AVAILABLE else ("legacy", "maximin", "nash")
PEOPLE_TO_SELECT = 22

test_path = Path(__file__).parent
features_csv_path = test_path / "fixtures/features.csv"
candidates_csv_path = test_path / "fixtures/candidates.csv"
features_content = features_csv_path.read_text("utf8")
candidates_content = candidates_csv_path.read_text("utf8")
candidates_lines = [line.strip() for line in candidates_content.split("\n") if line.strip()]

# do the string so we can split lines more nicely
selected_str = "p52,p140,p6,p71,p21,p1,p10,p103,p19,p84,p56,p88,p48,p112,p38,p119,p67,p45,p76,p79,p137,p100"
selected = frozenset(selected_str.split(","))

# The header line of candidates.csv is
csv_header = (
    "nationbuilder_id,first_name,last_name,mobile_number,email,primary_address1,"
    "primary_address2,primary_city,primary_zip,gender,age_bracket,geo_bucket,edu_level"
)


def get_settings(algorithm="legacy"):
    columns_to_keep = [
        "first_name",
        "last_name",
        "mobile_number",
        "email",
        "primary_address1",
        "primary_address2",
        "primary_city",
        "primary_zip",
        "gender",
        "age_bracket",
        "geo_bucket",
        "edu_level",
    ]
    return Settings(
        id_column="nationbuilder_id",
        columns_to_keep=columns_to_keep,
        check_same_address=True,
        check_same_address_columns=["primary_address1", "primary_zip"],
        selection_algorithm=algorithm,
    )


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_csv_selection_happy_path_defaults(algorithm):
    """
    Objective: Check the happy path completes.
    Context:
        This test is meant to do what the use will do via the GUI when using a CSV file.
    Expectations:
        Given default settings and an easy selection, we should get selected and remaining.
    """
    adapter = CSVAdapter()
    settings = get_settings(algorithm)
    features, _ = adapter.load_features_from_str(features_content)
    people, people_msgs = adapter.load_people_from_str(candidates_content, settings, features)
    print("load_people_message: ")
    print(people_msgs)

    success, people_selected, _ = run_stratification(features, people, PEOPLE_TO_SELECT, settings)

    # print(report.as_text())
    assert success
    assert len(people_selected) == 1
    assert len(people_selected[0]) == PEOPLE_TO_SELECT
    print(people_selected[0])


def test_csv_load_feature_from_file_or_str_give_same_output():
    adapter = CSVAdapter()
    feature_from_file, _ = adapter.load_features_from_file(features_csv_path)
    feature_from_str, _ = adapter.load_features_from_str(features_content)
    assert feature_from_file == feature_from_str


def test_csv_load_people_from_file_or_str_give_same_output():
    adapter = CSVAdapter()
    settings = get_settings()
    features, _ = adapter.load_features_from_str(features_content)
    people_from_file, _ = adapter.load_people_from_file(candidates_csv_path, settings, features)
    people_from_str, _ = adapter.load_people_from_str(candidates_content, settings, features)
    assert people_from_file == people_from_str


def test_csv_output_selected_remaining():
    adapter = CSVAdapter()
    settings = get_settings()
    features, _ = adapter.load_features_from_str(features_content)
    people, _ = adapter.load_people_from_str(candidates_content, settings, features)

    selected_rows, remaining_rows, _ = core.selected_remaining_tables(people, selected, features, settings)
    adapter.output_selected_remaining(selected_rows, remaining_rows)

    selected_content = adapter.selected_file.getvalue()
    selected_lines = selected_content.splitlines()
    assert selected_lines[0] == csv_header
    remaining_content = adapter.remaining_file.getvalue()
    remaining_lines = remaining_content.splitlines()
    assert remaining_lines[0] == csv_header

    # we have the -1 to remove the header
    num_selected = len(selected_lines) - 1
    num_remaining = len(remaining_lines) - 1
    num_candidates = len(candidates_lines) - 1

    assert num_selected + num_remaining == num_candidates
