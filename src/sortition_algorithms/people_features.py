import typing
from collections.abc import Generator, Iterable, MutableMapping

from attrs import define
from requests.structures import CaseInsensitiveDict

from sortition_algorithms import errors
from sortition_algorithms.features import FeatureCollection, FeatureValueMinMax, iterate_feature_collection
from sortition_algorithms.people import People


@define(kw_only=True, slots=True)
class SelectCounts:
    """
    Note that remaining and most of the methods are only used by the legacy algorithm.
    But to avoid duplication we also use this class, and the associated methods for both
    reporting and the legacy algorithm. Maybe at some point we will duplicate the code
    and have separate versions for the two purposes.
    """

    min_max: FeatureValueMinMax
    selected: int = 0
    remaining: int = 0

    def add_remaining(self) -> None:
        self.remaining += 1

    def add_selected(self) -> None:
        self.selected += 1

    def remove_remaining(self) -> None:
        self.remaining -= 1
        if self.remaining == 0 and self.selected < self.min_max.min:
            msg = "SELECTION IMPOSSIBLE: FAIL - no one/not enough left after deletion."
            raise errors.RetryableSelectionError(msg)

    @property
    def hit_target(self) -> bool:
        """Return true if selected is between min and max (inclusive)"""
        return self.selected >= self.min_max.min and self.selected <= self.min_max.max

    def percent_selected(self, number_people_wanted: int) -> float:
        return self.selected * 100 / float(number_people_wanted)

    @property
    def people_still_needed(self) -> int:
        """The number of extra people to select to get to the minimum - it should never be negative"""
        return max(self.min_max.min - self.selected, 0)

    def sufficient_people(self) -> bool:
        """
        Return true if we can still make the minimum. So either:
        - we have already selected at least the minimum, or
        - the remaining number is at least as big as the number still required
        """
        return self.selected >= self.min_max.min or self.remaining >= self.people_still_needed


SelectCollection: typing.TypeAlias = MutableMapping[str, MutableMapping[str, SelectCounts]]  # noqa: UP040
# TODO: when python3.11 is dropped, change to:
# type SelectCollection = dict[str, dict[str, SelectCounts]]


def select_from_feature_collection(fc: FeatureCollection) -> SelectCollection:
    select_collection: SelectCollection = CaseInsensitiveDict()
    for fname, fvalue_name, fv_minmax in iterate_feature_collection(fc):
        if fname not in select_collection:
            select_collection[fname] = CaseInsensitiveDict()
        select_collection[fname][fvalue_name] = SelectCounts(min_max=fv_minmax)
    return select_collection


def iterate_select_collection(select_collection: SelectCollection) -> Generator[tuple[str, str, SelectCounts]]:
    """Helper function to iterate over select_collection."""
    for feature_name, feature_values in select_collection.items():
        for value_name, select_counts in feature_values.items():
            yield feature_name, value_name, select_counts


def simple_add_selected(person_keys: Iterable[str], people: People, features: SelectCollection) -> None:
    """
    Just add the person to the selected counts for the feature values for that person.
    Don't do the more complex handling of the full PeopleFeatures.add_selected()
    """
    for person_key in person_keys:
        person = people.get_person_dict(person_key)
        for feature_name in features:
            feature_value = person[feature_name]
            features[feature_name][feature_value].add_selected()
