from collections.abc import Iterable, Iterator
from copy import deepcopy

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

    def __iter__(self) -> Iterator[str]:
        return iter(self._full_data)

    def get_person_dict(self, person_key: str) -> dict[str, str]:
        return self._full_data[person_key]


class PeopleFeatures:
    """
    This class manipulates people and features together, making a deepcopy on init.
    """

    # TODO: consider naming: maybe SelectionState

    def __init__(self, people: People, features: FeatureCollection) -> None:
        self.people = deepcopy(people)
        self.features = deepcopy(features)

    def update_features_remaining(self, person_key: str) -> None:
        # this will blow up if the person does not exist
        person = self.people.get_person_dict(person_key)
        for feature_name in self.features.feature_names:
            self.features.add_remaining(feature_name, person[feature_name])

    def update_all_features_remaining(self) -> None:
        for person_key in self.people:
            self.update_features_remaining(person_key)

    def delete_all_with_feature_value(
        self,
        feature_name: str,
        feature_value: str,
    ) -> tuple[int, int]:
        """
        When a feature is full we want to delete everyone in it.
        Returns count of those deleted, and count of those left
        """
        people_to_delete: list[str] = []
        for pkey in self.people:
            person = self.people.get_person_dict(pkey)
            if person[feature_name] == feature_value:
                people_to_delete.append(pkey)
                # adjust the features "remaining" values for each feature in turn
                for feature in self.features.feature_names:
                    self.features.remove_remaining(feature, person[feature])
        for p in people_to_delete:
            self.people.remove(p)
        # return the number of people deleted and the number of people left
        return len(people_to_delete), self.people.count

    def prune_for_feature_max_0(self) -> list[str]:
        """
        Check if any feature_value.max is set to zero. if so delete everyone with that feature value
        NOT DONE: could then check if anyone is left.
        """
        # TODO: when do we want to do this?
        msg: list[str] = []
        msg.append(f"Number of people: {self.people.count}.")
        total_num_deleted = 0
        for (
            feature_name,
            feature_value,
            fv_counts,
        ) in self.features.feature_values_counts():
            if fv_counts.max == 0:  # we don't want any of these people
                # pass the message in as deleting them might throw an exception
                msg.append(f"Feature/value {feature_name}/{feature_value} full - deleting people...")
                num_deleted, num_left = self.delete_all_with_feature_value(feature_name, feature_value)
                # if no exception was thrown above add this bit to the end of the previous message
                msg[-1] += f" Deleted {num_deleted}, {num_left} left."
                total_num_deleted += num_deleted
        # if the total number of people deleted is lots then we're probably doing a replacement selection, which means
        # the 'remaining' file will be useless - remind the user of this!
        if total_num_deleted >= self.people.count / 2:
            msg.append(
                ">>> WARNING <<< That deleted MANY PEOPLE - are you doing a "
                "replacement? If so your REMAINING FILE WILL BE USELESS!!!"
            )
        return msg


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
