from collections import Counter, defaultdict
from collections.abc import Generator, ItemsView, Iterable, Iterator, MutableMapping
from copy import deepcopy
from typing import Any

from requests.structures import CaseInsensitiveDict

from sortition_algorithms.errors import (
    BadDataError,
    ParseErrorsCollector,
    ParseTableMultiError,
    SelectionError,
    SelectionMultilineError,
)
from sortition_algorithms.features import FeatureCollection, iterate_feature_collection
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import RunReport, normalise_dict


class People:
    def __init__(self, columns_to_keep: list[str]) -> None:
        self._columns_to_keep = columns_to_keep
        self._full_data: dict[str, MutableMapping[str, str]] = {}

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._full_data == other._full_data and self._columns_to_keep == self._columns_to_keep

    @property
    def count(self) -> int:
        return len(self._full_data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._full_data)

    def items(self) -> ItemsView[str, MutableMapping[str, str]]:
        return self._full_data.items()

    def add(
        self,
        person_key: str,
        data: MutableMapping[str, str],
        features: FeatureCollection,
        row_number: int,
        feature_column_name: str = "feature",
    ) -> None:
        person_full_data: MutableMapping[str, str] = CaseInsensitiveDict()
        errors = ParseErrorsCollector()
        # get the feature values: these are the most important and we must check them
        for feature_name, feature_values in features.items():
            # check for input errors here - if it's not in the list of feature values...
            # allow for some unclean data - at least strip empty space, but only if a str!
            # (some values will can be numbers)
            p_value = data[feature_name]
            if p_value in feature_values:
                person_full_data[feature_name] = p_value
            else:
                if p_value:
                    msg = f"Value '{p_value}' not in {feature_column_name} {feature_name}"
                    error_code = "value_not_in_feature"
                    error_params = {
                        "value": p_value,
                        "feature_column_name": feature_column_name,
                        "feature_name": feature_name,
                    }
                else:
                    msg = f"Empty value in {feature_column_name} {feature_name}"
                    error_code = "empty_value_in_feature"
                    error_params = {"feature_column_name": feature_column_name, "feature_name": feature_name}
                errors.add(
                    msg=msg,
                    key=feature_name,
                    value=p_value,
                    row=row_number,
                    row_name=person_key,
                    error_code=error_code,
                    error_params=error_params,
                )
        if errors:
            raise errors.to_error()
        # then get the other column values we need
        # this is address, name etc that we need to keep for output file
        # we don't check anything here - it's just for user convenience
        for col in self._columns_to_keep:
            person_full_data[col] = data[col]

        # add all the data to our people object
        self._full_data[person_key] = person_full_data

    def remove(self, person_key: str) -> None:
        del self._full_data[person_key]

    def remove_many(self, person_keys: Iterable[str]) -> None:
        for key in person_keys:
            self.remove(key)

    def get_person_dict(self, person_key: str) -> MutableMapping[str, str]:
        return self._full_data[person_key]

    @staticmethod
    def _address_tuple(person_dict: MutableMapping[str, str], address_columns: Iterable[str]) -> tuple[str, ...]:
        return tuple(person_dict[col].lower() for col in address_columns)

    def get_address(self, person_key: str, address_columns: Iterable[str]) -> tuple[str, ...]:
        return self._address_tuple(self._full_data[person_key], address_columns)

    def households(self, address_columns: list[str]) -> dict[tuple[str, ...], list[str]]:
        """
        Generates a dict with:
        - keys: a tuple containing the address strings
        - values: a list of person_key for each person at that address
        """
        households = defaultdict(list)
        for person_key, person in self._full_data.items():
            households[self._address_tuple(person, address_columns)].append(person_key)
        return households

    def matching_address(self, person_key: str, address_columns: list[str]) -> Iterable[str]:
        """
        Returns a list of person keys for all people who have an address matching
        the address of the person passed in.
        """
        person = self._full_data[person_key]
        person_address = self._address_tuple(person, address_columns)
        for loop_key, loop_person in self._full_data.items():
            if loop_key == person_key:
                continue  # skip the person we've been given
            if person_address == self._address_tuple(loop_person, address_columns):
                yield loop_key

    def _iter_matching(self, feature_name: str, feature_value: str) -> Generator[str]:
        for person_key, person_dict in self._full_data.items():
            if person_dict[feature_name].lower() == feature_value.lower():
                yield person_key

    def count_feature_value(self, feature_name: str, feature_value: str) -> int:
        return len(list(self._iter_matching(feature_name, feature_value)))

    def find_person_by_position_in_category(self, feature_name: str, feature_value: str, position: int) -> str:
        """
        Find the nth person (1-indexed) in a specific feature category.

        Args:
            feature_name: Name of the feature (e.g., "gender")
            feature_value: Value of the feature (e.g., "male")
            position: 1-indexed position within the category

        Returns:
            Person key of the person at the specified position

        Raises:
            SelectionError: If no person is found at the specified position
        """
        people_in_category = list(self._iter_matching(feature_name, feature_value))
        try:
            return people_in_category[position - 1]
        except IndexError:
            # Should always find someone if position is valid
            # If we hit this line it is a bug
            msg = f"Failed to find person at position {position} in {feature_name}/{feature_value}"
            raise SelectionError(
                message=msg,
                error_code="person_not_found",
                error_params={"position": position, "feature_name": feature_name, "feature_value": feature_value},
            ) from None


# simple helper function to tidy the code below
def _check_columns_exist_or_multiple(
    people_head: list[str], column_list: Iterable[str], error_label: str, data_container: str = "people data"
) -> None:
    people_head_lower = [h.lower() for h in people_head]
    for column in column_list:
        column = column.lower()
        column_count = people_head_lower.count(column)
        if column_count == 0:
            msg = f"No '{column}' column {error_label} found in {data_container}!"
            raise BadDataError(
                message=msg,
                error_code="missing_column",
                error_params={"column": column, "error_label": error_label, "data_container": data_container},
            )
        elif column_count > 1:
            msg = f"MORE THAN 1 '{column}' column {error_label} found in {data_container}!"
            raise BadDataError(
                message=msg,
                error_code="duplicate_column",
                error_params={"column": column, "error_label": error_label, "data_container": data_container},
            )


def _check_people_head(
    people_head: list[str], features: FeatureCollection, settings: Settings, data_container: str = "people data"
) -> None:
    # check that id_column and all the features, columns_to_keep and
    # check_same_address_columns are in the people data fields...
    # check both for existence and duplicate column names
    _check_columns_exist_or_multiple(people_head, [settings.id_column], "(unique id)", data_container)
    _check_columns_exist_or_multiple(people_head, list(features.keys()), "(a feature)", data_container)
    _check_columns_exist_or_multiple(people_head, settings.columns_to_keep, "(to keep)", data_container)
    _check_columns_exist_or_multiple(
        people_head,
        settings.check_same_address_columns,
        "(to check same address)",
        data_container,
    )


def _all_in_list_equal(list_to_check: list[Any]) -> bool:
    return all(item == list_to_check[0] for item in list_to_check)


def check_for_duplicate_people(people_body: Iterable[MutableMapping[str, str]], settings: Settings) -> RunReport:
    """
    If we have rows with duplicate IDs things are going to go bad.
    First check for any duplicate IDs. If we find any, check if the duplicates are identical.

    Returns:
        RunReport containing warnings about duplicate people

    Raises:
        SelectionMultilineError: If duplicate IDs have different data
    """
    report = RunReport()

    # first check for any duplicate_ids
    id_counter = Counter(row[settings.id_column] for row in people_body)
    duplicate_ids = {k for k, v in id_counter.items() if v > 1}
    if not duplicate_ids:
        return report

    # find the duplicate rows
    duplicate_rows: dict[str, list[MutableMapping[str, str]]] = defaultdict(list)
    for row in people_body:
        pkey = row[settings.id_column]
        if pkey in duplicate_ids:
            duplicate_rows[pkey].append(row)

    report.add_line(f"Found {len(duplicate_rows)} IDs that have more than one row")
    report.add_line(f"Duplicated IDs are: {' '.join(duplicate_rows)}")

    # find rows where everything is not equal
    duplicate_differing_rows: dict[str, list[MutableMapping[str, str]]] = {}
    for key, value in duplicate_rows.items():
        if not _all_in_list_equal(value):
            duplicate_differing_rows[key] = value
    if not duplicate_differing_rows:
        report.add_line("All duplicate rows have identical data - processing continuing.")
        return report

    # Build error message with full context
    output: list[str] = []
    output.append(f"Found {len(duplicate_rows)} IDs that have more than one row")
    output.append(f"Duplicated IDs are: {' '.join(duplicate_rows)}")
    output.append(f"Found {len(duplicate_differing_rows)} IDs that have more than one row with different data")
    for key, value in duplicate_differing_rows.items():
        for row in value:
            output.append(f"For ID '{key}' one row of data is: {row}")
    raise SelectionMultilineError(output)


def check_enough_people_for_every_feature_value(features: FeatureCollection, people: People) -> None:
    """For each feature/value, if the min>0, check there are enough people who have that feature/value"""
    error_list: list[str] = []
    for fname, fvalue, fv_minmax in iterate_feature_collection(features):
        matching_count = people.count_feature_value(fname, fvalue)
        if matching_count < fv_minmax.min:
            error_list.append(
                f"Not enough people with the value '{fvalue}' in category '{fname}' - "
                f"the minimum is {fv_minmax.min} but we only have {matching_count}"
            )
    if error_list:
        raise SelectionMultilineError(error_list)


def exclude_matching_selected_addresses(people: People, already_selected: People | None, settings: Settings) -> People:
    """
    If we are checking the same addresses, then we should start by excluding people
    who have the same address as someone who is already selected.
    """
    if already_selected is None or not already_selected.count or not settings.check_same_address:
        return people
    selected_addresses = {
        already_selected.get_address(pkey, settings.check_same_address_columns) for pkey in already_selected
    }
    new_people = deepcopy(people)
    for person_key in people:
        if people.get_address(person_key, settings.check_same_address_columns) in selected_addresses:
            new_people.remove(person_key)
    return new_people


def read_in_people(
    people_head: list[str],
    people_body: Iterable[dict[str, str] | dict[str, str | int]],
    features: FeatureCollection,
    settings: Settings,
    feature_column_name: str = "feature",
    data_container: str = "people data",
) -> tuple[People, RunReport]:
    report = RunReport()
    _check_people_head(people_head, features, settings, data_container)
    # we need to iterate through more than once, so save as list here
    stripped_people_body = [normalise_dict(row) for row in people_body]
    report.add_report(check_for_duplicate_people(stripped_people_body, settings))
    people = People(settings.full_columns_to_keep)
    combined_error = ParseTableMultiError()
    # row 1 is the header, so the body starts on row 2
    for row_number, stripped_row in enumerate(stripped_people_body, start=2):
        pkey = stripped_row[settings.id_column]
        # skip over any blank lines... but warn the user
        if pkey == "":
            report.add_message("blank_id_skipped", row=row_number)
            continue
        try:
            people.add(
                person_key=pkey,
                data=stripped_row,
                features=features,
                row_number=row_number,
                feature_column_name=feature_column_name,
            )
        except ParseTableMultiError as error:
            # gather all the errors so we can tell the user as many problems as possible in one pass
            combined_error.combine(error)

    # if we got any errors in the above loop, raise the combined error.
    if combined_error:
        raise combined_error

    # Note this function just reads in people but doesn't update the features
    # to generate the remaining and prune those with max 0.
    # That is done in committee_generation.legacy.find_random_sample_legacy()
    return people, report
