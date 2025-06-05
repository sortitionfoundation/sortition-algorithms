from collections import defaultdict
from collections.abc import ItemsView, Iterable, Iterator

from sortition_algorithms import errors
from sortition_algorithms.features import FeatureCollection
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import StrippedDict


class People:
    def __init__(self, columns_to_keep: list[str]) -> None:
        self._columns_to_keep = columns_to_keep
        self._full_data: dict[str, dict[str, str]] = {}
        # TODO: review if we need to keep this separate - it is `self.columns_data` in the old code
        # maybe it can just be a property:
        # return {k, v for k, v in self.full_data.items() if k in self.columns_to_keep}
        # actually that dict comprehension would need to be done for every
        # sub dict in the original - bit more work
        self._col_to_keep_data: dict[str, dict[str, str]] = {}

    @property
    def count(self) -> int:
        return len(self._full_data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._full_data)

    def items(self) -> ItemsView[str, dict[str, str]]:
        return self._full_data.items()

    def add(self, person_key: str, data: StrippedDict, features: FeatureCollection) -> None:
        person_full_data: dict[str, str] = {}
        # get the feature values: these are the most important and we must check them
        for feature_name, feature_values in features.feature_values():
            # check for input errors here - if it's not in the list of feature values...
            # allow for some unclean data - at least strip empty space, but only if a str!
            # (some values will can be numbers)
            p_value = data[feature_name]
            if p_value not in feature_values:
                exc_msg = (
                    f"ERROR reading in people (read_in_people): "
                    f"Person (id = {person_key}) has value '{p_value}' not in feature {feature_name}"
                )
                raise errors.BadDataError(exc_msg)
            person_full_data[feature_name] = p_value
        # then get the other column values we need
        # this is address, name etc that we need to keep for output file
        # we don't check anything here - it's just for user convenience
        person_col_to_keep_data: dict[str, str] = {}
        for col in self._columns_to_keep:
            person_col_to_keep_data[col] = person_full_data[col] = data[col]

        # add all the data to our people object
        self._full_data[person_key] = person_full_data
        self._col_to_keep_data[person_key] = person_col_to_keep_data

    def remove(self, person_key: str) -> None:
        del self._full_data[person_key]
        del self._col_to_keep_data[person_key]

    def remove_many(self, person_keys: Iterable[str]) -> None:
        for key in person_keys:
            self.remove(key)

    def get_person_dict(self, person_key: str) -> dict[str, str]:
        return self._full_data[person_key]

    def households(self, address_columns: list[str]) -> dict[tuple[str, ...], list[str]]:
        """
        Generates a dict with:
        - keys: a tuple containing the address strings
        - values: a list of person_key for each person at that address
        """
        households = defaultdict(list)
        for person_key, person in self._full_data.items():
            address = tuple(person[col] for col in address_columns)
            households[address].append(person_key)
        return households

    def matching_address(self, person_key: str, address_columns: list[str]) -> Iterable[str]:
        """
        Returns a list of person keys for all people who have an address matching
        the address of the person passed in.
        """
        person = self._full_data[person_key]
        person_address = tuple(person[col] for col in address_columns)
        for loop_key, loop_person in self._full_data.items():
            if loop_key == person_key:
                continue  # skip the person we've been given
            if person_address == tuple(loop_person[col] for col in address_columns):
                yield loop_key

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
        current_position = 0

        for person_key, person_dict in self._full_data.items():
            if person_dict[feature_name] == feature_value:
                current_position += 1
                if current_position == position:
                    return person_key

        # Should always find someone if position is valid
        msg = f"Failed to find person at position {position} in {feature_name}/{feature_value}"
        raise errors.SelectionError(msg)


# simple helper function to tidy the code below
def _check_columns_exist_or_multiple(people_head: list[str], column_list: Iterable[str], error_text: str) -> None:
    for column in column_list:
        column_count = people_head.count(column)
        if column_count == 0:
            msg = f"No '{column}' column {error_text} found in people data!"
            raise errors.BadDataError(msg)
        elif column_count > 1:
            msg = f"MORE THAN 1 '{column}' column {error_text} found in people data!"
            raise errors.BadDataError(msg)


def _check_people_head(people_head: list[str], features: FeatureCollection, settings: Settings) -> None:
    # check that id_column and all the features, columns_to_keep and
    # check_same_address_columns are in the people data fields...
    # check both for existence and duplicate column names
    _check_columns_exist_or_multiple(people_head, [settings.id_column], "(unique id)")
    _check_columns_exist_or_multiple(people_head, features.feature_names, "(a feature)")
    _check_columns_exist_or_multiple(people_head, settings.columns_to_keep, "(to keep)")
    _check_columns_exist_or_multiple(
        people_head,
        settings.check_same_address_columns,
        "(to check same address)",
    )


def _ensure_settings_keep_address_columns(settings: Settings) -> None:
    # let's just merge the check_same_address_columns into columns_to_keep in case they aren't in both
    # TODO: review this - should we do this in settings rather than here?
    for col in settings.check_same_address_columns:
        if col not in settings.columns_to_keep:
            settings.columns_to_keep.append(col)


def read_in_people(
    people_head: list[str],
    people_body: list[dict[str, str | int]],
    features: FeatureCollection,
    settings: Settings,
) -> tuple[People, list[str]]:
    all_msg: list[str] = []
    _check_people_head(people_head, features, settings)
    _ensure_settings_keep_address_columns(settings)
    people = People(settings.columns_to_keep)
    for index, row in enumerate(people_body):
        stripped_row = StrippedDict(row)
        pkey = stripped_row[settings.id_column]
        # skip over any blank lines... but warn the user
        if pkey == "":
            all_msg.append(f"<b>WARNING</b>: blank cell found in ID column in row {index} - skipped that line!")
            continue
        people.add(pkey, stripped_row, features)
    # TODO: should this be done outside this function?
    # so this function just reads in people but doesn't update the features stuff
    # people_features = PeopleFeatures(people, features)
    # people_features.update_all_features_remaining()
    # people_features.prune_for_feature_max_0()
    return people, all_msg
