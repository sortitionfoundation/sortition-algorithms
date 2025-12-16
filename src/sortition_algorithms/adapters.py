"""
Adapters for loading and saving data.

Initially we have CSV files locally, and Google Docs Spreadsheets.
"""

import abc
import csv
import logging
from collections import Counter, defaultdict
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar, TextIO

import gspread
from gspread.utils import ValueRenderOption
from oauth2client.service_account import ServiceAccountCredentials

from sortition_algorithms.errors import (
    ParseTableErrorMsg,
    ParseTableMultiError,
    SelectionError,
    SelectionMultilineError,
)
from sortition_algorithms.features import FeatureCollection, read_in_features
from sortition_algorithms.people import People, read_in_people
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import RunReport, get_cell_name, user_logger


def _stringify_records(
    records: Iterable[dict[str, str | int | float] | dict[str, str]],
) -> list[dict[str, str]]:
    new_records: list[dict[str, str]] = []
    for record in records:
        new_records.append({k: str(v) for k, v in record.items()})
    return new_records


def _non_empty_cells(records: Iterable[str | float] | Iterable[str]) -> list[str]:
    """
    Take a list of strings and numbers and return the cells that are non-empty.
    A string with only whitespace counts as empty.
    Convert all numbers to strings.
    """
    return [str(cell).strip() for cell in records if cell and str(cell).strip()]


def generate_dupes(  # noqa: C901
    people_remaining_rows: list[list[str]],
    people_selected_rows: list[list[str]],
    settings: Settings,
    already_selected: People | None = None,
) -> list[int]:
    """
    Generate a list of indexes of people who share an address with someone else in this set of rows.

    Note that the first row of people_remaining_rows is the column headers.  The indexes generated
    are for the rows in this table, so the index takes account of the first row being the header.

    So if we had people_remaining_rows:

    id,name,address_line_1,postcode
    1,Alice,33 Acacia Avenue,W1A 1AA
    1,Bob,31 Acacia Avenue,W1A 1AA
    1,Charlotte,33 Acacia Avenue,W1A 1AA
    1,David,33 Acacia Avenue,W1B 1BB

    And settings with `check_same_address_columns = ["address_line_1", "postcode"]`

    Then we should return [1, 3]
    """
    if not settings.check_same_address:
        return []

    table_col_names = people_remaining_rows[0]
    address_col_indexes: list[int] = [
        index for index, col in enumerate(table_col_names) if col in settings.check_same_address_columns
    ]

    def _address_from_row(person: list[str]) -> tuple[str, ...]:
        return tuple(col.lower() for col_index, col in enumerate(person) if col_index in address_col_indexes)

    # first, we assemble a dict with the key being the address, the value being the list of
    # indexes of people at that address
    address_remaining_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for person_index, person in enumerate(people_remaining_rows[1:], start=1):  # skip the header row
        address_remaining_index[_address_from_row(person)].append(person_index)

    # now extract all those people where the number of people at their address is more than one
    dupes: set[int] = set()
    for persons_at_address in address_remaining_index.values():
        if len(persons_at_address) > 1:
            dupes.update(persons_at_address)

    # Now we assemble the list of all selected addresses.
    already_selected_addresses: set[tuple[str, ...]] = set()
    # First those selected in this round of selection
    for person in people_selected_rows[1:]:  # skip the header row
        already_selected_addresses.add(_address_from_row(person))
    # Then those previously selected (if supplied)
    if already_selected:
        for selected_key in already_selected:
            already_selected_addresses.add(
                already_selected.get_address(selected_key, settings.check_same_address_columns)
            )

    # and check if any of the remaining people has an address matching
    # those selected in this round or previous rounds
    for person_index, person in enumerate(people_remaining_rows[1:], start=1):  # skip the header row
        if _address_from_row(person) in already_selected_addresses:
            dupes.add(person_index)

    return sorted(dupes)


class AbstractDataSource(abc.ABC):
    @property
    @abc.abstractmethod
    def people_data_container(self) -> str: ...

    @property
    @abc.abstractmethod
    def already_selected_data_container(self) -> str: ...

    @abc.abstractmethod
    @contextmanager
    def read_feature_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]: ...

    @abc.abstractmethod
    @contextmanager
    def read_people_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]: ...

    @abc.abstractmethod
    @contextmanager
    def read_already_selected_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]: ...

    @abc.abstractmethod
    def write_selected(self, selected: list[list[str]], report: RunReport) -> None: ...

    @abc.abstractmethod
    def write_remaining(self, remaining: list[list[str]], report: RunReport) -> None: ...

    @abc.abstractmethod
    def highlight_dupes(self, dupes: list[int]) -> None: ...

    @abc.abstractmethod
    def customise_features_parse_error(
        self, error: ParseTableMultiError, headers: Sequence[str]
    ) -> SelectionMultilineError: ...

    @abc.abstractmethod
    def customise_people_parse_error(
        self, error: ParseTableMultiError, headers: Sequence[str]
    ) -> SelectionMultilineError: ...

    @abc.abstractmethod
    def customise_already_selected_parse_error(
        self, error: ParseTableMultiError, headers: Sequence[str]
    ) -> SelectionMultilineError: ...


class SelectionData:
    def __init__(self, data_source: AbstractDataSource, gen_rem_tab: bool = True) -> None:
        self.data_source = data_source
        # short for "generate remaining tab"
        self.gen_rem_tab = gen_rem_tab  # Added for checkbox in strat select app
        # record the column headers for feature/category and for value
        self.feature_column_name = "feature"
        self.feature_value_column_name = "value"

    def load_features(self, number_to_select: int = 0) -> tuple[FeatureCollection, RunReport]:
        report = RunReport()
        with self.data_source.read_feature_data(report) as headers_body:
            headers_iter, body = headers_body
            headers = list(headers_iter)
            try:
                features, self.feature_column_name, self.feature_value_column_name = read_in_features(
                    headers, body, number_to_select=number_to_select
                )
            except ParseTableMultiError as error:
                new_error = self.data_source.customise_features_parse_error(error, headers)
                raise new_error from error
        report.add_message("features_found", count=len(features))
        return features, report

    def load_people(self, settings: Settings, features: FeatureCollection) -> tuple[People, RunReport]:
        report = RunReport()
        with self.data_source.read_people_data(report) as headers_body:
            headers_iter, body = headers_body
            headers = list(headers_iter)
            try:
                people, report = read_in_people(
                    people_head=headers,
                    people_body=body,
                    features=features,
                    settings=settings,
                    feature_column_name=self.feature_column_name,
                    data_container=self.data_source.people_data_container,
                )
            except ParseTableMultiError as error:
                new_error = self.data_source.customise_people_parse_error(error, headers)
                raise new_error from error
        return people, report

    def load_already_selected(self, settings: Settings, features: FeatureCollection) -> tuple[People, RunReport]:
        report = RunReport()
        with self.data_source.read_already_selected_data(report) as headers_body:
            headers_iter, body = headers_body
            headers = list(headers_iter)
            # we can return empty if there is nothing to load, so return empty People
            if not headers:
                return People(columns_to_keep=settings.columns_to_keep), report

            try:
                people, report = read_in_people(
                    people_head=headers,
                    people_body=body,
                    features=features,
                    settings=settings,
                    feature_column_name=self.feature_column_name,
                    data_container=self.data_source.already_selected_data_container,
                )
            except ParseTableMultiError as error:
                new_error = self.data_source.customise_already_selected_parse_error(error, headers)
                raise new_error from error
        return people, report

    # TODO: decide if we want this - it would just call the function in core.py
    # Is that worth it?
    # def run_stratification(self):

    def output_selected_remaining(
        self,
        people_selected_rows: list[list[str]],
        people_remaining_rows: list[list[str]],
        settings: Settings,
        already_selected: People | None = None,
    ) -> tuple[list[int], RunReport]:
        report = RunReport()
        self.data_source.write_selected(people_selected_rows, report)
        if not self.gen_rem_tab:
            report.add_message_and_log("finished_writing_selected_only", logging.INFO)
            return [], report
        self.data_source.write_remaining(people_remaining_rows, report)
        # TODO: also highlight dupes of address in selected tab/set
        dupes = generate_dupes(people_remaining_rows, people_selected_rows, settings, already_selected=already_selected)
        self.data_source.highlight_dupes(dupes)
        report.add_message_and_log("finished_writing_selected_and_remaining", logging.INFO)
        return dupes, report

    def output_multi_selections(self, multi_selections: list[list[str]]) -> RunReport:
        assert not self.gen_rem_tab
        report = RunReport()
        self.data_source.write_selected(multi_selections, report)
        return report


def _write_csv_rows(out_file: TextIO, rows: list[list[str]]) -> None:
    writer = csv.writer(
        out_file,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
    )
    for row in rows:
        writer.writerow(row)


class CSVStringDataSource(AbstractDataSource):
    def __init__(self, features_data: str, people_data: str, already_selected_data: str = "") -> None:
        self.features_data = features_data
        self.people_data = people_data
        self.already_selected_data = already_selected_data
        self.selected_file = StringIO()
        self.remaining_file = StringIO()
        self.selected_file_written = False
        self.remaining_file_written = False

    @property
    def people_data_container(self) -> str:
        return "people CSV data"

    @property
    def already_selected_data_container(self) -> str:
        return "already selected CSV data"

    @contextmanager
    def read_feature_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        report.add_message("loading_features_from_string")
        feature_reader = csv.DictReader(StringIO(self.features_data), strict=True)
        assert feature_reader.fieldnames is not None
        yield list(feature_reader.fieldnames), feature_reader

    @contextmanager
    def read_people_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        report.add_message("loading_people_from_string")
        people_reader = csv.DictReader(StringIO(self.people_data), strict=True)
        assert people_reader.fieldnames is not None
        yield list(people_reader.fieldnames), people_reader

    @contextmanager
    def read_already_selected_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        if not self.already_selected_data or not self.already_selected_data.strip():
            report.add_message("no_already_selected_data")
            yield [], []
            return
        report.add_message("loading_already_selected_from_string")
        already_selected_reader = csv.DictReader(StringIO(self.already_selected_data), strict=True)
        assert already_selected_reader.fieldnames is not None
        yield list(already_selected_reader.fieldnames), already_selected_reader

    def write_selected(self, selected: list[list[str]], report: RunReport) -> None:
        _write_csv_rows(self.selected_file, selected)
        self.selected_file_written = True

    def write_remaining(self, remaining: list[list[str]], report: RunReport) -> None:
        _write_csv_rows(self.remaining_file, remaining)
        self.remaining_file_written = True

    def highlight_dupes(self, dupes: list[int]) -> None:
        """Cannot highlight a CSV file"""

    def customise_features_parse_error(
        self, error: ParseTableMultiError, headers: Sequence[str]
    ) -> SelectionMultilineError:
        # given the info is in strings, we can't usefully add anything
        return error

    def customise_people_parse_error(
        self, error: ParseTableMultiError, headers: Sequence[str]
    ) -> SelectionMultilineError:
        # given the info is in strings, we can't usefully add anything
        return error

    def customise_already_selected_parse_error(
        self, error: ParseTableMultiError, headers: Sequence[str]
    ) -> SelectionMultilineError:
        # given the info is in strings, we can't usefully add anything
        return error


class CSVFileDataSource(AbstractDataSource):
    def __init__(
        self,
        *,
        features_file: Path,
        people_file: Path,
        already_selected_file: Path | None,
        selected_file: Path,
        remaining_file: Path,
    ) -> None:
        self.features_file = features_file
        self.people_file = people_file
        self.already_selected_file = already_selected_file
        self.selected_file = selected_file
        self.remaining_file = remaining_file

    @property
    def people_data_container(self) -> str:
        return f" CSV file '{self.people_file}'"

    @property
    def already_selected_data_container(self) -> str:
        return f" CSV file '{self.already_selected_file}'"

    @contextmanager
    def read_feature_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        report.add_message("loading_features_from_file", file_path=str(self.features_file))
        with open(self.features_file, newline="") as csv_file:
            feature_reader = csv.DictReader(csv_file, strict=True)
            assert feature_reader.fieldnames is not None
            yield list(feature_reader.fieldnames), feature_reader

    @contextmanager
    def read_people_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        report.add_message("loading_people_from_file", file_path=str(self.people_file))
        with open(self.people_file, newline="") as csv_file:
            people_reader = csv.DictReader(csv_file, strict=True)
            assert people_reader.fieldnames is not None
            yield list(people_reader.fieldnames), people_reader

    @contextmanager
    def read_already_selected_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        if self.already_selected_file is None or not self.already_selected_file.exists():
            report.add_message("no_already_selected_file")
            yield [], []
            return
        report.add_message("loading_already_selected_from_file", file_path=str(self.already_selected_file))
        with open(self.already_selected_file, newline="") as csv_file:
            already_selected_reader = csv.DictReader(csv_file, strict=True)
            assert already_selected_reader.fieldnames is not None
            yield list(already_selected_reader.fieldnames), already_selected_reader

    def write_selected(self, selected: list[list[str]], report: RunReport) -> None:
        report.add_message_and_log("writing_selected_csv", logging.INFO, file_path=self.selected_file)
        with open(self.selected_file, "w", newline="") as csv_file:
            _write_csv_rows(csv_file, selected)

    def write_remaining(self, remaining: list[list[str]], report: RunReport) -> None:
        report.add_message_and_log("writing_remaining_csv", logging.INFO, file_path=self.remaining_file)
        with open(self.remaining_file, "w", newline="") as csv_file:
            _write_csv_rows(csv_file, remaining)

    def highlight_dupes(self, dupes: list[int]) -> None:
        """Cannot highlight a CSV file"""

    def customise_features_parse_error(
        self, error: ParseTableMultiError, headers: Sequence[str]
    ) -> SelectionMultilineError:
        return SelectionMultilineError([
            f"Parser error(s) while reading features from {self.features_file}",
            *[str(e) for e in error.all_errors],
        ])

    def customise_people_parse_error(
        self, error: ParseTableMultiError, headers: Sequence[str]
    ) -> SelectionMultilineError:
        return SelectionMultilineError([
            f"Parser error(s) while reading people from {self.people_file}",
            *[str(e) for e in error.all_errors],
        ])

    def customise_already_selected_parse_error(
        self, error: ParseTableMultiError, headers: Sequence[str]
    ) -> SelectionMultilineError:
        return SelectionMultilineError([
            f"Parser error(s) while reading already selected people from {self.already_selected_file}",
            *[str(e) for e in error.all_errors],
        ])


class GSheetTabNamer:
    def __init__(self) -> None:
        self.selected_tab_name_stub = "Original Selected - output - "
        self.remaining_tab_name_stub = "Remaining - output - "
        self._write_tab_suffix = ""

    def reset(self) -> None:
        self._write_tab_suffix = ""

    def find_unused_tab_suffix(self, current_tab_titles: list[str]) -> None:
        # have an upper limit to avoid infinite loops
        for number in range(1000):
            selected_tab_name_candidate = f"{self.selected_tab_name_stub}{number}"
            remaining_tab_name_candidate = f"{self.remaining_tab_name_stub}{number}"
            if (
                selected_tab_name_candidate not in current_tab_titles
                and remaining_tab_name_candidate not in current_tab_titles
            ):
                self._write_tab_suffix = f"{number}"
                break

    def selected_tab_name(self) -> str:
        if not self._write_tab_suffix:
            raise SelectionError(
                message="Logic error - trying to create new tab before choosing suffix",
                error_code="logic_error_tab_suffix",
                error_params={},
            )
        return f"{self.selected_tab_name_stub}{self._write_tab_suffix}"

    def remaining_tab_name(self) -> str:
        if not self._write_tab_suffix:
            raise SelectionError(
                message="Logic error - trying to create new tab before choosing suffix",
                error_code="logic_error_tab_suffix",
                error_params={},
            )
        return f"{self.remaining_tab_name_stub}{self._write_tab_suffix}"

    def matches_stubs(self, tab_name: str) -> bool:
        return tab_name.startswith(self.selected_tab_name_stub) or tab_name.startswith(self.remaining_tab_name_stub)


class GSheetDataSource(AbstractDataSource):
    scope: ClassVar = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    hl_light_blue: ClassVar = {
        "backgroundColor": {
            "red": 153 / 255,
            "green": 204 / 255,
            "blue": 255 / 255,
        }
    }
    hl_orange: ClassVar = {"backgroundColor": {"red": 5, "green": 2.5, "blue": 0}}

    def __init__(
        self,
        *,
        feature_tab_name: str,
        people_tab_name: str,
        already_selected_tab_name: str = "",
        id_column: str = "not_set",
        auth_json_path: Path,
    ) -> None:
        """
        Args:
        - feature_tab_name - the name of the tab/worksheet containing the features (aka categories)
        - people_tab_name - the name of the tab/worksheet containing the people to select from
        - already_selected_tab_name (optional) - the name of the tab/worksheet containing people who have already been selected
        - id_column (optional) - the name of the column containing the ID. Only required if already_selected_tab_name is set
        - auth_json_path - path to the file containing the google service account details.
        """
        self.feature_tab_name = feature_tab_name
        self.people_tab_name = people_tab_name
        self.already_selected_tab_name = already_selected_tab_name
        self.id_column = id_column
        self.auth_json_path = auth_json_path
        self._client: gspread.client.Client | None = None
        self._spreadsheet: gspread.Spreadsheet | None = None
        self.new_tab_default_size_rows = 2
        self.new_tab_default_size_cols = 40
        self._g_sheet_name = ""
        self._open_g_sheet_name = ""
        self.selected_tab_name = ""
        self.remaining_tab_name = ""
        self.tab_namer = GSheetTabNamer()
        self._report = RunReport()

    @property
    def people_data_container(self) -> str:
        return f"'{self.people_tab_name}' tab"

    @property
    def already_selected_data_container(self) -> str:
        return f"'{self.already_selected_tab_name}' tab"

    @property
    def client(self) -> gspread.client.Client:
        if self._client is None:
            creds = ServiceAccountCredentials.from_json_keyfile_name(str(self.auth_json_path), self.scope)
            # if we're getting rate limited, go slower!
            # by using the BackOffHTTPClient, that will sleep and retry
            # if it gets an error related to API usage rate limits.
            self._client = gspread.authorize(creds, http_client=gspread.BackOffHTTPClient)
        return self._client

    @property
    def spreadsheet(self) -> gspread.Spreadsheet:
        if self._open_g_sheet_name != self._g_sheet_name:
            # reset the spreadsheet if the name changed
            self._spreadsheet = None
            self.tab_namer.reset()
        if self._spreadsheet is None:
            if self._g_sheet_name.startswith("https://"):
                self._spreadsheet = self.client.open_by_url(self._g_sheet_name)
            else:
                self._spreadsheet = self.client.open(self._g_sheet_name)
            self._open_g_sheet_name = self._g_sheet_name
            self._report.add_message_and_log("opened_gsheet", logging.INFO, title=self._spreadsheet.title)
        return self._spreadsheet

    def _get_tab(self, tab_name: str) -> gspread.Worksheet | None:
        if not self._g_sheet_name:
            return None
        tab_list = self.spreadsheet.worksheets()
        try:
            return next(tab for tab in tab_list if tab.title == tab_name)
        except StopIteration:
            return None

    def _tab_exists(self, tab_name: str) -> bool:
        return bool(self._get_tab(tab_name))

    def _get_tab_titles(self) -> list[str]:
        if not self._g_sheet_name:
            return []
        return [tab.title for tab in self.spreadsheet.worksheets()]

    def _create_tab(self, tab_name: str) -> gspread.Worksheet:
        return self.spreadsheet.add_worksheet(
            title=tab_name,
            rows=self.new_tab_default_size_rows,
            cols=self.new_tab_default_size_cols,
        )

    def set_g_sheet_name(self, g_sheet_name: str) -> None:
        # if we're changing spreadsheet, reset the spreadsheet object
        if self._g_sheet_name != g_sheet_name:
            self._spreadsheet = None
            self._g_sheet_name = g_sheet_name
            self.tab_namer.reset()

    def get_title(self) -> str:
        try:
            return self.spreadsheet.title
        except gspread.SpreadsheetNotFound as err:
            msg = f"Google spreadsheet not found: {self._g_sheet_name}."
            raise SelectionError(
                message=msg, error_code="spreadsheet_not_found", error_params={"spreadsheet_name": self._g_sheet_name}
            ) from err

    @contextmanager
    def read_feature_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        self._report = report
        try:
            if not self._tab_exists(self.feature_tab_name):
                msg = (
                    f"Error in Google sheet: no tab called '{self.feature_tab_name}' "
                    f"found in spreadsheet '{self.spreadsheet.title}'."
                )
                raise SelectionError(
                    message=msg,
                    error_code="tab_not_found",
                    error_params={"tab_name": self.feature_tab_name, "spreadsheet_title": self.spreadsheet.title},
                )
        except gspread.SpreadsheetNotFound as err:
            msg = f"Google spreadsheet not found: {self._g_sheet_name}."
            raise SelectionError(
                message=msg, error_code="spreadsheet_not_found", error_params={"spreadsheet_name": self._g_sheet_name}
            ) from err
        tab_features = self.spreadsheet.worksheet(self.feature_tab_name)
        feature_head = tab_features.row_values(1)
        feature_body = _stringify_records(tab_features.get_all_records(expected_headers=[]))
        yield feature_head, feature_body

    @contextmanager
    def read_people_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        self._report = report
        try:
            if not self._tab_exists(self.people_tab_name):
                msg = (
                    f"Error in Google sheet: no tab called '{self.people_tab_name}' "
                    f"found in spreadsheet '{self.spreadsheet.title}'."
                )
                raise SelectionError(
                    message=msg,
                    error_code="tab_not_found",
                    error_params={"tab_name": self.people_tab_name, "spreadsheet_title": self.spreadsheet.title},
                )
        except gspread.SpreadsheetNotFound as err:
            msg = f"Google spreadsheet not found: {self._g_sheet_name}. "
            raise SelectionError(
                message=msg, error_code="spreadsheet_not_found", error_params={"spreadsheet_name": self._g_sheet_name}
            ) from err

        tab_people = self.spreadsheet.worksheet(self.people_tab_name)
        # if we don't read this in here we can't check if there are 2 columns with the same name
        people_head = tab_people.row_values(1)
        # the numericise_ignore doesn't convert the phone numbers to ints...
        # 1 Oct 2024: the final argument with expected_headers is to deal with the fact that
        # updated versions of gspread can't cope with duplicate headers
        people_body = _stringify_records(
            tab_people.get_all_records(
                numericise_ignore=["all"],
                expected_headers=[],
            )
        )
        self._report.add_message("reading_gsheet_tab", tab_name=self.people_tab_name)
        yield people_head, people_body

    @staticmethod
    def find_header_row(
        all_values: gspread.ValueRange | list[list[Any]], id_column: str, min_headers: int = 5
    ) -> tuple[int, list[str]]:
        """
        Find the first row that looks like a header row in the worksheet.

        A row is considered a header row if it has at least min_headers non-empty cells.
        This helps skip over title rows, empty rows, etc.

        Args:
            worksheet: The worksheet to search
            id_column: The text for the id_column - from settings
            min_headers: Minimum number of non-empty cells to consider a row as headers (default: 3)

        Returns:
            Tuple of (row_number, header_values) where row_number is 1-indexed
        """
        # Find the first row with enough non-empty cells to be a header row
        for row_idx, row in enumerate(all_values, start=1):
            # Count non-empty cells in this row
            non_empty_cells = _non_empty_cells(row)
            # also check if the id_column is in this row
            if len(non_empty_cells) >= min_headers and id_column in non_empty_cells:
                return row_idx, list(row)

        # If no suitable header row found, return empty
        return 1, []

    @staticmethod
    def get_valid_people_rows(
        entire_sheet: list[list[str]], header_row_num: int, min_values: int = 5
    ) -> Generator[list[str]]:
        """
        Find all rows under the header row that contain at least 3 values.

        Sometimes in the spreadsheet there will be a comment in a cell underneath all the valid rows.
        We should ignore such rows rather than attempt to convert them into an element of People.
        """
        all_rows_under_header = entire_sheet[header_row_num:]
        for row in all_rows_under_header:
            # Count non-empty cells in this row
            non_empty_cells = _non_empty_cells(row)
            if len(non_empty_cells) > min_values:
                yield row

    @staticmethod
    def check_header_duplicates(already_selected_head: list[str], already_selected_tab_name: str) -> None:
        counts = Counter(already_selected_head)
        # note that we ignore empty strings when checking for duplicates
        header_dupes = [item for item in counts if item and counts[item] > 1]
        if header_dupes:
            msg = f"the header row in the {already_selected_tab_name} tab contains duplicates: {header_dupes}"
            raise SelectionError(
                message=msg,
                error_code="already_selected_duplicate_headers",
                error_params={"tab_name": already_selected_tab_name, "duplicates": ", ".join(header_dupes)},
            )

    @contextmanager
    def read_already_selected_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        self._report = report
        if not self.already_selected_tab_name:
            # If no tab name provided, return empty data
            self._report.add_message("no_already_selected_tab")
            yield [], []
            return

        try:
            if not self._tab_exists(self.already_selected_tab_name):
                msg = (
                    f"Error in Google sheet: no tab called '{self.already_selected_tab_name}' "
                    f"found in spreadsheet '{self.spreadsheet.title}'."
                )
                raise SelectionError(
                    message=msg,
                    error_code="tab_not_found",
                    error_params={
                        "tab_name": self.already_selected_tab_name,
                        "spreadsheet_title": self.spreadsheet.title,
                    },
                )
        except gspread.SpreadsheetNotFound as err:
            msg = f"Google spreadsheet not found: {self._g_sheet_name}. "
            raise SelectionError(
                message=msg, error_code="spreadsheet_not_found", error_params={"spreadsheet_name": self._g_sheet_name}
            ) from err

        tab_already_selected = self.spreadsheet.worksheet(self.already_selected_tab_name)

        # Get all values from the sheet
        all_values = tab_already_selected.get(value_render_option=ValueRenderOption.unformatted, pad_values=True)

        # Find which row contains the headers
        header_row_num, already_selected_head = self.find_header_row(all_values, self.id_column)

        # If no header row found or it's empty, return empty data
        if not already_selected_head or not any(cell.strip() for cell in already_selected_head):
            self._report.add_line(
                f"Tab '{self.already_selected_tab_name}' is empty or has no valid header row, using empty data."
            )
            yield [], []
            return

        self.check_header_duplicates(already_selected_head, self.already_selected_tab_name)

        already_selected_body = _stringify_records([
            dict(zip(already_selected_head, row, strict=False))
            for row in self.get_valid_people_rows(all_values, header_row_num)
        ])
        self._report.add_message(
            "reading_already_selected_tab", tab_name=self.already_selected_tab_name, header_row=header_row_num
        )
        yield already_selected_head, already_selected_body

    def write_selected(self, selected: list[list[str]], report: RunReport) -> None:
        self.tab_namer.find_unused_tab_suffix(self._get_tab_titles())
        tab_selected = self._create_tab(self.tab_namer.selected_tab_name())
        report.add_message_and_log("writing_selected_tab", logging.INFO, tab_name=tab_selected.title)
        self.selected_tab_name = tab_selected.title
        tab_selected.update(selected)
        tab_selected.format("A1:U1", self.hl_light_blue)
        user_logger.info(f"Selected people written to {tab_selected.title} tab")

    def write_remaining(self, remaining: list[list[str]], report: RunReport) -> None:
        # the number is selected during write_selected(), so we reuse it here
        tab_remaining = self._create_tab(self.tab_namer.remaining_tab_name())
        report.add_message_and_log("writing_remaining_tab", logging.INFO, tab_name=tab_remaining.title)
        self.remaining_tab_name = tab_remaining.title
        tab_remaining.update(remaining)
        tab_remaining.format("A1:U1", self.hl_light_blue)

    def highlight_dupes(self, dupes: list[int]) -> None:
        if not dupes:
            return
        tab_remaining = self._get_tab(self.tab_namer.remaining_tab_name())
        assert tab_remaining is not None, "highlight_dupes() has been called without first calling write_remaining()"
        # note that the indexes we have produced start at 0, but the row indexes start at 1
        # so we need to add 1 to the indexes.
        row_strings = [f"A{index + 1}:U{index + 1}" for index in dupes]
        tab_remaining.format(row_strings, self.hl_orange)

    def delete_old_output_tabs(self, dry_run: bool = False) -> list[str]:
        """
        Find and delete all tabs with names starting with the tab stubs for selected or remaining

        Args:
            dry_run: If True, report what would be deleted without actually deleting.

        Returns:
            List of tab names that were deleted (or would be deleted in dry_run mode).
        """
        if not self._g_sheet_name:
            return []

        all_tabs = self.spreadsheet.worksheets()
        tabs_to_delete: list[gspread.Worksheet] = []

        for tab in all_tabs:
            if self.tab_namer.matches_stubs(tab.title):
                tabs_to_delete.append(tab)

        deleted_names: list[str] = []
        for tab in tabs_to_delete:
            deleted_names.append(tab.title)
            if not dry_run:
                self.spreadsheet.del_worksheet(tab)

        return deleted_names

    def _annotate_parse_errors_with_cell_names(self, error: ParseTableMultiError, headers: Sequence[str]) -> list[str]:
        msgs: list[str] = []
        for sub_error in error.all_errors:
            if isinstance(sub_error, ParseTableErrorMsg):
                cell_name = get_cell_name(sub_error.row, sub_error.key, headers)
                msgs.append(f"{sub_error.msg} - see cell {cell_name}")
            else:
                cell_names = [get_cell_name(sub_error.row, key, headers) for key in sub_error.keys]
                msgs.append(f"{sub_error.msg} - see cells {' '.join(cell_names)}")
        return msgs

    def customise_features_parse_error(
        self, error: ParseTableMultiError, headers: Sequence[str]
    ) -> SelectionMultilineError:
        return SelectionMultilineError([
            f"Parser error(s) while reading features from '{self.feature_tab_name}' worksheet",
            *self._annotate_parse_errors_with_cell_names(error, headers),
        ])

    def customise_people_parse_error(
        self, error: ParseTableMultiError, headers: Sequence[str]
    ) -> SelectionMultilineError:
        return SelectionMultilineError([
            f"Parser error(s) while reading people from '{self.people_tab_name}' worksheet",
            *self._annotate_parse_errors_with_cell_names(error, headers),
        ])

    def customise_already_selected_parse_error(
        self, error: ParseTableMultiError, headers: Sequence[str]
    ) -> SelectionMultilineError:
        return SelectionMultilineError([
            f"Parser error(s) while reading people from '{self.already_selected_tab_name}' worksheet",
            *self._annotate_parse_errors_with_cell_names(error, headers),
        ])
