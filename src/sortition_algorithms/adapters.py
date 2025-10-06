"""
Adapters for loading and saving data.

Initially we have CSV files locally, and Google Docs Spreadsheets.
"""

import abc
import csv
import logging
from collections import defaultdict
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import ClassVar, TextIO

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from sortition_algorithms.errors import SelectionError
from sortition_algorithms.features import FeatureCollection, read_in_features
from sortition_algorithms.people import People, read_in_people
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import RunReport, user_logger


def _stringify_records(
    records: Iterable[dict[str, str | int | float] | dict[str, str]],
) -> list[dict[str, str]]:
    new_records: list[dict[str, str]] = []
    for record in records:
        new_records.append({k: str(v) for k, v in record.items()})
    return new_records


def generate_dupes(people_remaining_rows: list[list[str]], settings: Settings) -> list[int]:
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
    address_remaining_index: dict[tuple[str, ...], list[int]] = defaultdict(list)

    # first, we assemble a dict with the key being the address, the value being the list of
    # indexes of people at that address
    for person_index, person in enumerate(people_remaining_rows):
        if person_index == 0:
            continue  # skip the header row
        address_tuple = tuple(col for col_index, col in enumerate(person) if col_index in address_col_indexes)
        address_remaining_index[address_tuple].append(person_index)

    # now extract all those people where the number of people at their address is more than one
    dupes: list[int] = []
    for persons_at_address in address_remaining_index.values():
        if len(persons_at_address) > 1:
            dupes += persons_at_address

    return sorted(dupes)


class AbstractDataSource(abc.ABC):
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
    def write_selected(self, selected: list[list[str]], report: RunReport) -> None: ...

    @abc.abstractmethod
    def write_remaining(self, remaining: list[list[str]], report: RunReport) -> None: ...

    @abc.abstractmethod
    def highlight_dupes(self, dupes: list[int]) -> None: ...


class SelectionData:
    def __init__(self, data_source: AbstractDataSource, gen_rem_tab: bool = True) -> None:
        self.data_source = data_source
        # short for "generate remaining tab"
        self.gen_rem_tab = gen_rem_tab  # Added for checkbox in strat select app

    def load_features(self) -> tuple[FeatureCollection, RunReport]:
        report = RunReport()
        with self.data_source.read_feature_data(report) as headers_body:
            headers, body = headers_body
            features = read_in_features(list(headers), body)
        report.add_line(f"Number of features found: {len(features)}")
        return features, report

    def load_people(self, settings: Settings, features: FeatureCollection) -> tuple[People, RunReport]:
        report = RunReport()
        with self.data_source.read_people_data(report) as header_body:
            header, body = header_body
            people, report = read_in_people(list(header), body, features, settings)
        return people, report

    # TODO: decide if we want this - it would just call the function in core.py
    # Is that worth it?
    # def run_stratification(self):

    def output_selected_remaining(
        self,
        people_selected_rows: list[list[str]],
        people_remaining_rows: list[list[str]],
        settings: Settings,
    ) -> tuple[list[int], RunReport]:
        report = RunReport()
        self.data_source.write_selected(people_selected_rows, report)
        if not self.gen_rem_tab:
            report.add_line_and_log("Finished writing selected (only)", logging.INFO)
            return [], report
        self.data_source.write_remaining(people_remaining_rows, report)
        dupes = generate_dupes(people_remaining_rows, settings)
        self.data_source.highlight_dupes(dupes)
        report.add_line_and_log("Finished writing both selected and remaining", logging.INFO)
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
    def __init__(self, features_data: str, people_data: str) -> None:
        self.features_data = features_data
        self.people_data = people_data
        self.selected_file = StringIO()
        self.remaining_file = StringIO()
        self.selected_file_written = False
        self.remaining_file_written = False

    @contextmanager
    def read_feature_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        report.add_line("Loading features from string.")
        feature_reader = csv.DictReader(StringIO(self.features_data), strict=True)
        assert feature_reader.fieldnames is not None
        yield list(feature_reader.fieldnames), feature_reader

    @contextmanager
    def read_people_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        report.add_line("Loading people from string.")
        people_reader = csv.DictReader(StringIO(self.people_data), strict=True)
        assert people_reader.fieldnames is not None
        yield list(people_reader.fieldnames), people_reader

    def write_selected(self, selected: list[list[str]], report: RunReport) -> None:
        _write_csv_rows(self.selected_file, selected)
        self.selected_file_written = True

    def write_remaining(self, remaining: list[list[str]], report: RunReport) -> None:
        _write_csv_rows(self.remaining_file, remaining)
        self.remaining_file_written = True

    def highlight_dupes(self, dupes: list[int]) -> None:
        """Cannot highlight a CSV file"""


class CSVFileDataSource(AbstractDataSource):
    def __init__(self, features_file: Path, people_file: Path, selected_file: Path, remaining_file: Path) -> None:
        self.features_file = features_file
        self.people_file = people_file
        self.selected_file = selected_file
        self.remaining_file = remaining_file

    @contextmanager
    def read_feature_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        report.add_line(f"Loading features from file {self.features_file}.")
        with open(self.features_file, newline="") as csv_file:
            feature_reader = csv.DictReader(csv_file, strict=True)
            assert feature_reader.fieldnames is not None
            yield list(feature_reader.fieldnames), feature_reader

    @contextmanager
    def read_people_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        report.add_line(f"Loading people from file {self.people_file}.")
        with open(self.people_file, newline="") as csv_file:
            people_reader = csv.DictReader(csv_file, strict=True)
            assert people_reader.fieldnames is not None
            yield list(people_reader.fieldnames), people_reader

    def write_selected(self, selected: list[list[str]], report: RunReport) -> None:
        report.add_line_and_log(f"Writing selected rows to {self.selected_file}", logging.INFO)
        with open(self.selected_file, "w", newline="") as csv_file:
            _write_csv_rows(csv_file, selected)

    def write_remaining(self, remaining: list[list[str]], report: RunReport) -> None:
        report.add_line_and_log(f"Writing remaining rows to {self.selected_file}", logging.INFO)
        with open(self.remaining_file, "w", newline="") as csv_file:
            _write_csv_rows(csv_file, remaining)

    def highlight_dupes(self, dupes: list[int]) -> None:
        """Cannot highlight a CSV file"""


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

    def __init__(self, feature_tab_name: str, people_tab_name: str, auth_json_path: Path) -> None:
        self.feature_tab_name = feature_tab_name
        self.people_tab_name = people_tab_name
        self.auth_json_path = auth_json_path
        self._client: gspread.client.Client | None = None
        self._spreadsheet: gspread.Spreadsheet | None = None
        self.selected_tab_name_stub = "Original Selected - output - "
        self.remaining_tab_name_stub = "Remaining - output - "
        self.new_tab_default_size_rows = 2
        self.new_tab_default_size_cols = 40
        self._g_sheet_name = ""
        self._open_g_sheet_name = ""
        self.selected_tab_name = ""
        self.remaining_tab_name = ""
        self._report = RunReport()

    @property
    def client(self) -> gspread.client.Client:
        if self._client is None:
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                str(self.auth_json_path),
                self.scope,
            )
            self._client = gspread.authorize(creds)
        return self._client

    @property
    def spreadsheet(self) -> gspread.Spreadsheet:
        if self._open_g_sheet_name != self._g_sheet_name:
            # reset the spreadsheet if the name changed
            self._spreadsheet = None
        if self._spreadsheet is None:
            if self._g_sheet_name.startswith("https://"):
                self._spreadsheet = self.client.open_by_url(self._g_sheet_name)
            else:
                self._spreadsheet = self.client.open(self._g_sheet_name)
            self._open_g_sheet_name = self._g_sheet_name
            self._report.add_line_and_log(f"Opened Google Sheet: '{self._spreadsheet.title}'. ", log_level=logging.INFO)
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

    def _clear_or_create_tab(self, tab_name: str, other_tab_name: str, inc: int) -> gspread.Worksheet:
        # this now does not clear data but increments the sheet number...
        num = 0
        tab_ready: gspread.Worksheet | None = None
        tab_name_new = f"{tab_name}{num}"
        other_tab_name_new = f"{other_tab_name}{num}"
        while tab_ready is None:
            if self._tab_exists(tab_name_new) or self._tab_exists(other_tab_name_new):
                num += 1
                tab_name_new = f"{tab_name}{num}"
                other_tab_name_new = f"{other_tab_name}{num}"
            else:
                if inc == -1:
                    tab_name_new = f"{tab_name}{num - 1}"
                tab_ready = self.spreadsheet.add_worksheet(
                    title=tab_name_new,
                    rows=self.new_tab_default_size_rows,
                    cols=self.new_tab_default_size_cols,
                )
        return tab_ready

    def set_g_sheet_name(self, g_sheet_name: str) -> None:
        # if we're changing spreadsheet, reset the spreadsheet object
        if self._g_sheet_name != g_sheet_name:
            self._spreadsheet = None
            self._g_sheet_name = g_sheet_name

    @contextmanager
    def read_feature_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        self._report = report
        try:
            if not self._tab_exists(self.feature_tab_name):
                msg = f"Error in Google sheet: no tab called '{self.feature_tab_name}' found."
                self._report.add_line_and_log(msg, log_level=logging.ERROR)
                raise SelectionError(msg, self._report)
        except gspread.SpreadsheetNotFound as err:
            msg = f"Google spreadsheet not found: {self._g_sheet_name}."
            self._report.add_line_and_log(msg, log_level=logging.ERROR)
            raise SelectionError(msg, self._report) from err
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
                msg = f"Error in Google sheet: no tab called '{self.people_tab_name}' found. "
                self._report.add_line(msg)
                raise SelectionError(msg, self._report)
        except gspread.SpreadsheetNotFound as err:
            msg = f"Google spreadsheet not found: {self._g_sheet_name}. "
            self._report.add_line(msg)
            raise SelectionError(msg, self._report) from err

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
        self._report.add_line(f"Reading in '{self.people_tab_name}' tab in above Google sheet.")
        yield people_head, people_body

    def write_selected(self, selected: list[list[str]], report: RunReport) -> None:
        tab_selected = self._clear_or_create_tab(
            self.selected_tab_name_stub,
            "ignoreme",
            0,
        )
        report.add_line_and_log(f"Writing selected people to tab: {tab_selected.title}", logging.INFO)
        self.selected_tab_name = tab_selected.title
        tab_selected.update(selected)
        tab_selected.format("A1:U1", self.hl_light_blue)
        user_logger.info(f"Selected people written to {tab_selected.title} tab")

    def write_remaining(self, remaining: list[list[str]], report: RunReport) -> None:
        tab_remaining = self._clear_or_create_tab(
            self.remaining_tab_name_stub,
            self.selected_tab_name_stub,
            -1,
        )
        report.add_line_and_log(f"Writing remaining people to tab: {tab_remaining.title}", logging.INFO)
        self.remaining_tab_name = tab_remaining.title
        tab_remaining.update(remaining)
        tab_remaining.format("A1:U1", self.hl_light_blue)

    def highlight_dupes(self, dupes: list[int]) -> None:
        tab_remaining = self._get_tab(self.remaining_tab_name)
        assert tab_remaining is not None, "highlight_dupes() has been called without first calling write_remaining()"
        # note that the indexes we have produced start at 0, but the row indexes start at 1
        # so we need to add 1 to the indexes.
        row_strings = [f"A{index + 1}:U{index + 1}" for index in dupes]
        tab_remaining.format(row_strings, self.hl_orange)
