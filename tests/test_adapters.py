from pathlib import Path

import pytest

from sortition_algorithms import core
from sortition_algorithms.adapters import (
    CSVFileDataSource,
    CSVStringDataSource,
    GSheetDataSource,
    GSheetTabNamer,
    SelectionData,
    generate_dupes,
)
from sortition_algorithms.committee_generation import GUROBI_AVAILABLE
from sortition_algorithms.core import run_stratification
from sortition_algorithms.errors import (
    ParseTableErrorMsg,
    ParseTableMultiError,
    ParseTableMultiValueErrorMsg,
    SelectionError,
    SelectionMultilineError,
)
from sortition_algorithms.settings import Settings
from tests.helpers import candidates_csv_path, features_csv_path, get_settings_for_fixtures

# only test leximin if gurobipy is available
ALGORITHMS = ("legacy", "maximin", "leximin", "nash") if GUROBI_AVAILABLE else ("legacy", "maximin", "nash")
PEOPLE_TO_SELECT = 22

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
    settings = get_settings_for_fixtures(algorithm)
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
    settings = get_settings_for_fixtures()
    features, _ = string_select_data.load_features()
    people_from_str, _ = string_select_data.load_people(settings, features)
    people_from_file, _ = file_select_data.load_people(settings, features)
    assert people_from_file == people_from_str


def test_csv_output_selected_remaining():
    data_source = CSVStringDataSource(features_content, candidates_content)
    select_data = SelectionData(data_source)
    settings = get_settings_for_fixtures()
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


@pytest.mark.parametrize(
    "tab_name,match",
    [
        ("select - 0", True),
        ("select ", True),
        ("select", False),
        ("remain - 0", True),
        ("remain ", True),
        ("remain", False),
        ("other - 0", False),
    ],
)
def test_tab_namer_matches_stubs(tab_name, match):
    namer = GSheetTabNamer()
    namer.selected_tab_name_stub = "select "
    namer.remaining_tab_name_stub = "remain "
    assert namer.matches_stubs(tab_name) == match


def test_csv_files_customise_features_parse_error():
    file_data_source = CSVFileDataSource(features_csv_path, candidates_csv_path, Path("/"), Path("/"))
    parse_features_error = ParseTableMultiError([
        ParseTableErrorMsg(4, row_name="gender", key="name", value="", msg="Some error")
    ])
    new_error = file_data_source.customise_features_parse_error(parse_features_error, headers=("category", "name"))
    assert "Some error" in str(new_error)
    assert str(features_csv_path) in str(new_error)


def test_csv_files_customise_people_parse_error():
    file_data_source = CSVFileDataSource(features_csv_path, candidates_csv_path, Path("/"), Path("/"))
    parse_features_error = ParseTableMultiError([
        ParseTableErrorMsg(4, row_name="nb_1234", key="gender", value="asdf", msg="Another error")
    ])
    new_error = file_data_source.customise_people_parse_error(
        parse_features_error, headers=("nationbuilder_id", "gender")
    )
    assert "Another error" in str(new_error)
    assert str(candidates_csv_path) in str(new_error)


def test_gsheet_customise_features_parse_error():
    gsheet_data_source = GSheetDataSource(
        feature_tab_name="Categories", people_tab_name="Respondents", auth_json_path=Path("/")
    )
    parse_features_error = ParseTableMultiError([
        ParseTableErrorMsg(4, row_name="gender", key="name", value="", msg="Some error"),
        ParseTableMultiValueErrorMsg(
            6, row_name="gender", keys=["min", "max"], values=["6", "4"], msg="Min greater than max"
        ),
    ])
    new_error = gsheet_data_source.customise_features_parse_error(
        parse_features_error, headers=("category", "name", "min", "max")
    )
    assert "Some error - see cell B4" in str(new_error)
    assert "Min greater than max - see cells C6 D6" in str(new_error)
    assert "Categories worksheet" in str(new_error)


def test_gsheet_customise_people_parse_error():
    gsheet_data_source = GSheetDataSource(
        feature_tab_name="Categories", people_tab_name="Respondents", auth_json_path=Path("/")
    )
    parse_features_error = ParseTableMultiError([
        ParseTableErrorMsg(4, row_name="nb_1234", key="gender", value="asdf", msg="Another error")
    ])
    new_error = gsheet_data_source.customise_people_parse_error(
        parse_features_error, headers=("nationbuilder_id", "name", "gender")
    )
    assert "Another error - see cell C4" in str(new_error)
    assert "Respondents worksheet" in str(new_error)


def test_csv_load_feature_from_file_failure(tmp_path: Path):
    new_features_csv_path = tmp_path / "new_features.csv"
    with open(new_features_csv_path, "wb") as new_file:
        new_file.write(features_csv_path.read_bytes())
        # this line has min > max
        new_file.write(b"\ngeo_bucket,PictsieLand,5,3,0,5\n")
    file_data_source = CSVFileDataSource(new_features_csv_path, candidates_csv_path, Path("/"), Path("/"))
    file_select_data = SelectionData(file_data_source)
    with pytest.raises(SelectionMultilineError) as excinfo:
        file_select_data.load_features()
    assert "new_features.csv" in str(excinfo.value)
    assert "Minimum (5) should not be greater than maximum (3)" in str(excinfo.value)


def test_csv_load_people_from_file_failure(tmp_path: Path):
    new_people_csv_path = tmp_path / "new_people.csv"
    with open(new_people_csv_path, "wb") as new_file:
        new_file.write(candidates_csv_path.read_bytes())
        # this line has PictsieLand but the features do not have that
        new_file.write(
            b"p9991,first_name1,last_name1,email1,mobile_number1,"
            b"primary_address11,primary_address21,primary_city1,primary_zip1,Female,16-29,"
            b"PictsieLand,Level 1"
        )
    file_data_source = CSVFileDataSource(features_csv_path, new_people_csv_path, Path("/"), Path("/"))
    file_select_data = SelectionData(file_data_source)
    settings = get_settings_for_fixtures("maximin")
    features, _ = file_select_data.load_features()
    with pytest.raises(SelectionMultilineError) as excinfo:
        file_select_data.load_people(settings, features)
    assert "new_people.csv" in str(excinfo.value)
    assert "'PictsieLand' not in feature geo_bucket" in str(excinfo.value)
