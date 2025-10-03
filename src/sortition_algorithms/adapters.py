"""
Adapters for loading and saving data.

Initially we have CSV files locally, and Google Docs Spreadsheets.
"""

import abc
import csv
import logging
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import ClassVar, TextIO

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from sortition_algorithms.features import FeatureCollection, read_in_features
from sortition_algorithms.people import People, read_in_people
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import RunReport, user_logger


def _stringify_records(
    records: Iterable[dict[str, str | int | float]],
) -> list[dict[str, str]]:
    new_records: list[dict[str, str]] = []
    for record in records:
        new_records.append({k: str(v) for k, v in record.items()})
    return new_records


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
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str | int | float]]], None, None]: ...

    @abc.abstractmethod
    def write_selected(self, selected: list[list[str]]) -> None: ...

    @abc.abstractmethod
    def write_remaining(self, remaining: list[list[str]]) -> None: ...


class SelectionData:
    def __init__(self, data_source: AbstractDataSource) -> None:
        self.data_source = data_source

    def load_features(self) -> tuple[FeatureCollection, RunReport]:
        report = RunReport()
        with self.data_source.read_feature_data(report) as headers_body:
            headers, body = headers_body
            features = read_in_features(list(headers), body)
        report.add_line(f"Number of features found: {len(features)}")
        return features, report

    def load_people(self, settings: Settings, features: FeatureCollection) -> tuple[People, RunReport]:
        report = RunReport()
        with self.data_source.read_people_data(report) as header_raw_body:
            header, raw_body = header_raw_body
            body = _stringify_records(raw_body)
            people, report = read_in_people(list(header), body, features, settings)
        return people, report

    # TODO: decide if we want this - it would just call the function in core.py
    # Is that worth it?
    # def run_stratification(self):

    def output_selected_remaining(
        self,
        people_selected_rows: list[list[str]],
        people_remaining_rows: list[list[str]],
    ) -> None:
        self.data_source.write_selected(people_selected_rows)
        self.data_source.write_remaining(people_remaining_rows)

    def output_multi_selections(self, multi_selections: list[list[str]]) -> None:
        self.data_source.write_selected(multi_selections)


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
        feature_reader = csv.DictReader(StringIO(self.features_data))
        assert feature_reader.fieldnames is not None
        yield list(feature_reader.fieldnames), feature_reader

    @contextmanager
    def read_people_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str | int | float]]], None, None]:
        report.add_line("Loading people from string.")
        people_reader = csv.DictReader(StringIO(self.people_data))
        assert people_reader.fieldnames is not None
        yield list(people_reader.fieldnames), people_reader

    def write_selected(self, selected: list[list[str]]) -> None:
        _write_csv_rows(self.selected_file, selected)
        self.selected_file_written = True

    def write_remaining(self, remaining: list[list[str]]) -> None:
        _write_csv_rows(self.remaining_file, remaining)
        self.remaining_file_written = True


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
            feature_reader = csv.DictReader(csv_file)
            assert feature_reader.fieldnames is not None
            yield list(feature_reader.fieldnames), feature_reader

    @contextmanager
    def read_people_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str | int | float]]], None, None]:
        report.add_line(f"Loading people from file {self.people_file}.")
        with open(self.people_file, newline="") as csv_file:
            people_reader = csv.DictReader(csv_file)
            assert people_reader.fieldnames is not None
            yield list(people_reader.fieldnames), people_reader

    def write_selected(self, selected: list[list[str]]) -> None:
        with open(self.selected_file, "w", newline="") as csv_file:
            _write_csv_rows(csv_file, selected)

    def write_remaining(self, remaining: list[list[str]]) -> None:
        with open(self.remaining_file, "w", newline="") as csv_file:
            _write_csv_rows(csv_file, remaining)


class GSheetAdapter:
    # TODO: refactor: split out GSheetWrapper class that can be passed
    # into the __init__ for this class. The wrapper class should only
    # depend on gspread, not on anything else in here.
    # Could even have an AbstractDataWrapper class that we can have both
    # csv and gspread implementations of and then we have a generic adapter.
    # But that might lose too much control - highlighting rows done by gspread?
    # I guess the CSV wrapper could just have no-op methods for that.
    # Then tests can use a FakeDataWrapper
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

    def __init__(self, auth_json_path: Path, gen_rem_tab: bool = True) -> None:
        self.auth_json_path = auth_json_path
        self._client: gspread.client.Client | None = None
        self._spreadsheet: gspread.Spreadsheet | None = None
        self.original_selected_tab_name = "Original Selected - output - "
        self.selected_tab_name = "Selected"
        self.columns_selected_first = "C"
        self.column_selected_blank_num = 6
        self.remaining_tab_name = "Remaining - output - "
        self.new_tab_default_size_rows = 2
        self.new_tab_default_size_cols = 40
        self._g_sheet_name = ""
        self._open_g_sheet_name = ""
        self._report = RunReport()
        # short for "generate remaining tab"
        self.gen_rem_tab = gen_rem_tab  # Added for checkbox.

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
            self._report.add_line_and_log(f"Opened Google Sheet: '{self._g_sheet_name}'. ", log_level=logging.INFO)
        return self._spreadsheet

    def _tab_exists(self, tab_name: str) -> bool:
        if not self._g_sheet_name:
            return False
        tab_list = self.spreadsheet.worksheets()
        return any(tab.title == tab_name for tab in tab_list)

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

    def load_features(self, feature_tab_name: str) -> tuple[FeatureCollection | None, RunReport]:
        self._report = RunReport()  # reset report for new task
        try:
            if not self._tab_exists(feature_tab_name):
                self._report.add_line_and_log(
                    f"Error in Google sheet: no tab called '{feature_tab_name}' found.", log_level=logging.ERROR
                )
                return None, self._report
        except gspread.SpreadsheetNotFound:
            self._report.add_line_and_log(
                f"Google spreadsheet not found: {self._g_sheet_name}.", log_level=logging.ERROR
            )
            return None, self._report
        tab_features = self.spreadsheet.worksheet(feature_tab_name)
        feature_head = tab_features.row_values(1)
        feature_body = _stringify_records(tab_features.get_all_records(expected_headers=[]))
        features = read_in_features(feature_head, feature_body)
        self._report.add_line(f"Number of features found: {len(features)}")
        return features, self._report

    def load_people(
        self,
        respondents_tab_name: str,
        settings: Settings,
        features: FeatureCollection,
    ) -> tuple[People | None, RunReport]:
        self._report = RunReport()  # reset report for new task
        people: People | None = None
        try:
            if not self._tab_exists(respondents_tab_name):
                self._report.add_line(
                    f"Error in Google sheet: no tab called '{respondents_tab_name}' found. ",
                )
                return None, self._report
        except gspread.SpreadsheetNotFound:
            self._report.add_line(f"Google spreadsheet not found: {self._g_sheet_name}. ")
            return None, self._report

        tab_people = self.spreadsheet.worksheet(respondents_tab_name)
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
        self._report.add_line(f"Reading in '{respondents_tab_name}' tab in above Google sheet.")
        people, read_report = read_in_people(people_head, people_body, features, settings)
        self._report.add_report(read_report)
        return people, self._report

    def output_selected_remaining(
        self,
        people_selected_rows: list[list[str]],
        people_remaining_rows: list[list[str]],
        settings: Settings,
    ) -> list[int]:
        tab_original_selected = self._clear_or_create_tab(
            self.original_selected_tab_name,
            self.remaining_tab_name,
            0,
        )
        tab_original_selected.update(people_selected_rows)
        tab_original_selected.format("A1:U1", self.hl_light_blue)
        dupes: list[int] = []
        user_logger.info("Selected people written to {tab_original_selected.title} tab")
        if self.gen_rem_tab:
            tab_remaining = self._clear_or_create_tab(
                self.remaining_tab_name,
                self.original_selected_tab_name,
                -1,
            )
            tab_remaining.update(people_remaining_rows)
            tab_remaining.format("A1:U1", self.hl_light_blue)
            user_logger.info("Remaining people written to {tab_remaining.title} tab")
            # highlight any people in remaining tab at the same address
            if settings.check_same_address:
                address_cols: list[int] = [tab_remaining.find(csa).col for csa in settings.check_same_address_columns]  # type: ignore[union-attr]
                # TODO: spin out to separate function
                # TODO: rather than a set and O(N^2) nested loop
                # just go through once, adding row index to dict with key as tuple of the values of the columns
                # and value as the list of row indexes for that key.
                # then at the end go through the dict and any key with > 1 row index has duplicates
                # and we can add all the row indexes to the duplicate set.
                dupes_set: set[int] = set()
                n = len(people_remaining_rows)
                for i in range(n):
                    rowrem1 = people_remaining_rows[i]
                    for j in range(i + 1, n):
                        rowrem2 = people_remaining_rows[j]
                        if rowrem1 != rowrem2 and all(rowrem1[col] == rowrem2[col] for col in address_cols):
                            dupes_set.add(i + 1)
                            dupes_set.add(j + 1)
                dupes = sorted(dupes_set)
                for i in range(min(30, len(dupes))):
                    tab_remaining.format(str(dupes[i]), self.hl_orange)
        return dupes

    def output_multi_selections(
        self,
        multi_selections: list[list[str]],
    ) -> None:
        assert not self.gen_rem_tab
        tab_original_selected = self._clear_or_create_tab(
            self.original_selected_tab_name,
            "ignoreme",
            0,
        )
        tab_original_selected.update(multi_selections)
        tab_original_selected.format("A1:U1", self.hl_light_blue)
