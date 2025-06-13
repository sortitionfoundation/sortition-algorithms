from pathlib import Path

import pytest

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

dummy = """
The header line of candidates.csv is:
nationbuilder_id,first_name,last_name,email,mobile_number,primary_address1,primary_address2,primary_city,primary_zip,gender,age_bracket,geo_bucket,edu_level
"""


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
        max_attempts=100,
        selection_algorithm=algorithm,
        random_number_seed=0,
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
    settings = get_settings(algorithm)
    adapter = CSVAdapter()
    features, _ = adapter.load_features_from_str(features_content)
    people, people_msgs = adapter.load_people_from_str(candidates_content, settings, features)
    # people_cats.number_people_to_select = PEOPLE_TO_SELECT
    print("load_people_message: ")
    print(people_msgs)

    success, people_selected, output_lines = run_stratification(features, people, PEOPLE_TO_SELECT, settings)

    print(output_lines)
    assert success
    assert len(people_selected) == 1
    assert len(people_selected[0]) == PEOPLE_TO_SELECT


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


# TODO: test output_selected_remaining
# we have the -1 to remove the header
# assert len(selected_lines) + len(remaining_lines) == len(candidates_lines) - 1
