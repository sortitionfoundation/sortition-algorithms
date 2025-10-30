from pathlib import Path

import pytest

from sortition_algorithms import core
from sortition_algorithms.adapters import (
    CSVFileDataSource,
    CSVStringDataSource,
    GSheetTabNamer,
    SelectionData,
    generate_dupes,
)
from sortition_algorithms.committee_generation import GUROBI_AVAILABLE
from sortition_algorithms.core import run_stratification
from sortition_algorithms.errors import SelectionError
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
    data_source = CSVStringDataSource(features_content, candidates_content)
    select_data = SelectionData(data_source)
    settings = get_settings(algorithm)
    features, _ = select_data.load_features()
    people, people_report = select_data.load_people(settings, features)
    print("load_people_message: ")
    print(people_report.as_text())

    success, people_selected, _ = run_stratification(features, people, PEOPLE_TO_SELECT, settings)

    # print(report.as_text())
    assert success
    assert len(people_selected) == 1
    assert len(people_selected[0]) == PEOPLE_TO_SELECT
    print(people_selected[0])


def test_csv_load_feature_from_file_or_str_give_same_output():
    string_data_source = CSVStringDataSource(features_content, candidates_content)
    string_select_data = SelectionData(string_data_source)
    file_data_source = CSVFileDataSource(features_csv_path, candidates_csv_path, Path("/"), Path("/"))
    file_select_data = SelectionData(file_data_source)
    feature_from_str, _ = string_select_data.load_features()
    feature_from_file, _ = file_select_data.load_features()
    assert feature_from_file == feature_from_str


def test_csv_load_people_from_file_or_str_give_same_output():
    string_data_source = CSVStringDataSource(features_content, candidates_content)
    string_select_data = SelectionData(string_data_source)
    file_data_source = CSVFileDataSource(features_csv_path, candidates_csv_path, Path("/"), Path("/"))
    file_select_data = SelectionData(file_data_source)
    settings = get_settings()
    features, _ = string_select_data.load_features()
    people_from_str, _ = string_select_data.load_people(settings, features)
    people_from_file, _ = file_select_data.load_people(settings, features)
    assert people_from_file == people_from_str


def test_csv_output_selected_remaining():
    data_source = CSVStringDataSource(features_content, candidates_content)
    select_data = SelectionData(data_source)
    settings = get_settings()
    features, _ = select_data.load_features()
    people, _ = select_data.load_people(settings, features)

    selected_rows, remaining_rows, _ = core.selected_remaining_tables(people, selected, features, settings)
    select_data.output_selected_remaining(selected_rows, remaining_rows, settings=settings)

    selected_content = data_source.selected_file.getvalue()
    selected_lines = selected_content.splitlines()
    assert selected_lines[0] == csv_header
    remaining_content = data_source.remaining_file.getvalue()
    remaining_lines = remaining_content.splitlines()
    assert remaining_lines[0] == csv_header

    # we have the -1 to remove the header
    num_selected = len(selected_lines) - 1
    num_remaining = len(remaining_lines) - 1
    num_candidates = len(candidates_lines) - 1

    assert num_selected + num_remaining == num_candidates


def test_generate_dupes_with_duplicates():
    """
    Test that generate_dupes correctly identifies people sharing an address.
    """
    people_remaining_rows = [
        ["id", "name", "address_line_1", "postcode"],
        ["1", "Alice", "33 Acacia Avenue", "W1A 1AA"],
        ["2", "Bob", "31 Acacia Avenue", "W1A 1AA"],
        ["3", "Charlotte", "33 Acacia Avenue", "W1A 1AA"],
        ["4", "David", "33 Acacia Avenue", "W1B 1BB"],
    ]
    settings = Settings(
        id_column="id",
        columns_to_keep=["name"],
        check_same_address=True,
        check_same_address_columns=["address_line_1", "postcode"],
    )
    dupes = generate_dupes(people_remaining_rows, settings)
    assert dupes == [1, 3]


def test_generate_dupes_no_duplicates():
    """
    Test that generate_dupes returns empty list when no one shares an address.
    """
    people_remaining_rows = [
        ["id", "name", "address_line_1", "postcode"],
        ["1", "Alice", "33 Acacia Avenue", "W1A 1AA"],
        ["2", "Bob", "31 Acacia Avenue", "W1A 1BB"],
        ["3", "Charlotte", "35 Acacia Avenue", "W1A 1CC"],
    ]
    settings = Settings(
        id_column="id",
        columns_to_keep=["name"],
        check_same_address=True,
        check_same_address_columns=["address_line_1", "postcode"],
    )
    dupes = generate_dupes(people_remaining_rows, settings)
    assert dupes == []


def test_generate_dupes_check_disabled():
    """
    Test that generate_dupes returns empty list when check_same_address is disabled.
    """
    people_remaining_rows = [
        ["id", "name", "address_line_1", "postcode"],
        ["1", "Alice", "33 Acacia Avenue", "W1A 1AA"],
        ["2", "Bob", "33 Acacia Avenue", "W1A 1AA"],
    ]
    settings = Settings(
        id_column="id",
        columns_to_keep=["name"],
        check_same_address=False,
        check_same_address_columns=["address_line_1", "postcode"],
    )
    dupes = generate_dupes(people_remaining_rows, settings)
    assert dupes == []


def test_generate_dupes_multiple_groups():
    """
    Test that generate_dupes correctly identifies multiple groups of duplicates.
    """
    people_remaining_rows = [
        ["id", "name", "address_line_1", "postcode"],
        ["1", "Alice", "33 Acacia Avenue", "W1A 1AA"],
        ["2", "Bob", "33 Acacia Avenue", "W1A 1AA"],
        ["3", "Charlotte", "15 Oak Street", "W2B 2BB"],
        ["4", "David", "15 Oak Street", "W2B 2BB"],
        ["5", "Eve", "99 Pine Road", "W3C 3CC"],
    ]
    settings = Settings(
        id_column="id",
        columns_to_keep=["name"],
        check_same_address=True,
        check_same_address_columns=["address_line_1", "postcode"],
    )
    dupes = generate_dupes(people_remaining_rows, settings)
    assert dupes == [1, 2, 3, 4]


def test_generate_dupes_three_at_same_address():
    """
    Test that generate_dupes correctly identifies when three people share an address.
    """
    people_remaining_rows = [
        ["id", "name", "address_line_1", "postcode"],
        ["1", "Alice", "33 Acacia Avenue", "W1A 1AA"],
        ["2", "Bob", "33 Acacia Avenue", "W1A 1AA"],
        ["3", "Charlotte", "33 Acacia Avenue", "W1A 1AA"],
    ]
    settings = Settings(
        id_column="id",
        columns_to_keep=["name"],
        check_same_address=True,
        check_same_address_columns=["address_line_1", "postcode"],
    )
    dupes = generate_dupes(people_remaining_rows, settings)
    assert dupes == [1, 2, 3]


def test_generate_dupes_single_address_column():
    """
    Test that generate_dupes works correctly with a single address column.
    """
    people_remaining_rows = [
        ["id", "name", "postcode"],
        ["1", "Alice", "W1A 1AA"],
        ["2", "Bob", "W1A 1AA"],
        ["3", "Charlotte", "W1B 1BB"],
    ]
    settings = Settings(
        id_column="id",
        columns_to_keep=["name"],
        check_same_address=True,
        check_same_address_columns=["postcode"],
    )
    dupes = generate_dupes(people_remaining_rows, settings)
    assert dupes == [1, 2]


def test_generate_dupes_ignores_non_address_columns():
    """
    Test that generate_dupes only checks the specified columns, ignoring others.
    """
    people_remaining_rows = [
        ["id", "name", "address_line_1", "postcode", "phone"],
        ["1", "Alice", "33 Acacia Avenue", "W1A 1AA", "123"],
        ["2", "Bob", "33 Acacia Avenue", "W1A 1AA", "456"],
        ["3", "Charlotte", "35 Acacia Avenue", "W1B 1BB", "789"],
    ]
    settings = Settings(
        id_column="id",
        columns_to_keep=["name"],
        check_same_address=True,
        check_same_address_columns=["address_line_1", "postcode"],
    )
    dupes = generate_dupes(people_remaining_rows, settings)
    # Alice and Bob have same address_line_1 and postcode, despite different phone
    assert dupes == [1, 2]


def test_generate_dupes_only_header_row():
    """
    Test that generate_dupes returns empty list when only header row is present.
    """
    people_remaining_rows = [["id", "name", "address_line_1", "postcode"]]
    settings = Settings(
        id_column="id",
        columns_to_keep=["name"],
        check_same_address=True,
        check_same_address_columns=["address_line_1", "postcode"],
    )
    dupes = generate_dupes(people_remaining_rows, settings)
    assert dupes == []


def test_generate_dupes_partial_address_match():
    """
    Test that partial address matches (only one column matches) don't count as duplicates.
    """
    people_remaining_rows = [
        ["id", "name", "address_line_1", "postcode"],
        ["1", "Alice", "33 Acacia Avenue", "W1A 1AA"],
        ["2", "Bob", "33 Acacia Avenue", "W1B 1BB"],
        ["3", "Charlotte", "31 Acacia Avenue", "W1A 1AA"],
    ]
    settings = Settings(
        id_column="id",
        columns_to_keep=["name"],
        check_same_address=True,
        check_same_address_columns=["address_line_1", "postcode"],
    )
    dupes = generate_dupes(people_remaining_rows, settings)
    assert dupes == []


def test_tab_namer_refuses_to_give_names_before_scanning():
    """
    Test the tab namer refuses to give names before running find_unused_tab_suffix()
    """
    namer = GSheetTabNamer()
    with pytest.raises(SelectionError):
        namer.selected_tab_name()
    with pytest.raises(SelectionError):
        namer.remaining_tab_name()


def test_tab_namer_gives_names_after_scanning():
    """
    Test the tab namer give names after running find_unused_tab_suffix()
    """
    namer = GSheetTabNamer()
    namer.find_unused_tab_suffix(["Categories", "Remaining"])
    assert namer.selected_tab_name() == "Original Selected - output - 0"
    assert namer.remaining_tab_name() == "Remaining - output - 0"


def test_tab_namer_refuses_to_give_names_after_reset():
    """
    Test the tab namer refuses to give names before running find_unused_tab_suffix()
    """
    namer = GSheetTabNamer()
    namer.find_unused_tab_suffix(["Categories", "Remaining"])
    namer.reset()
    with pytest.raises(SelectionError):
        namer.selected_tab_name()
    with pytest.raises(SelectionError):
        namer.remaining_tab_name()


@pytest.mark.parametrize(
    "tab_names,selected_tab_name",
    [
        (["Cat"], "select 0"),
        (["Cat", "select 1"], "select 0"),
        (["Cat", "remain 1"], "select 0"),
        (["Cat", "select 1", "remain 1"], "select 0"),
        (["Cat", "select 0"], "select 1"),
        (["Cat", "remain 0"], "select 1"),
        (["Cat", "select 0", "remain 0"], "select 1"),
        (["Cat", "select 0", "select 1"], "select 2"),
        (["Cat", "select 0", "remain 1"], "select 2"),
    ],
)
def test_tab_namer_finds_correct_suffix(tab_names, selected_tab_name):
    namer = GSheetTabNamer()
    namer.selected_tab_name_stub = "select "
    namer.remaining_tab_name_stub = "remain "
    namer.find_unused_tab_suffix(tab_names)
    assert namer.selected_tab_name() == selected_tab_name
