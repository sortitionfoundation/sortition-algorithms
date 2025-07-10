from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

from attrs import define

from sortition_algorithms import errors, utils

"""
Note on terminology.  The word "categories" can mean various things, so for this
code we use the terms "Feature" and "Value".

- Feature example: gender
- Value examples: male, female, non-binary

In the UI we may use the terms

- "category" to mean "feature" and
- "bucket" to mean "value".

More in the docs:
https://sortitionfoundation.github.io/sortition-algorithms/concepts/#features-and-feature-values
"""

FEATURE_FILE_FIELD_NAMES = ("feature", "value", "min", "max")
FEATURE_FILE_FIELD_NAMES_FLEX = (
    *FEATURE_FILE_FIELD_NAMES,
    "min_flex",
    "max_flex",
)
FEATURE_FILE_FIELD_NAMES_OLD = ("category", "name", "min", "max")
FEATURE_FILE_FIELD_NAMES_FLEX_OLD = (
    *FEATURE_FILE_FIELD_NAMES_OLD,
    "min_flex",
    "max_flex",
)
ALL_FEATURE_FIELD_NAMES = frozenset([*FEATURE_FILE_FIELD_NAMES_FLEX, *FEATURE_FILE_FIELD_NAMES_OLD])

MAX_FLEX_UNSET = -1


@define(kw_only=True, slots=True)
class FeatureValueCounts:
    min: int
    max: int
    selected: int = 0
    remaining: int = 0
    min_flex: int = 0
    max_flex: int = MAX_FLEX_UNSET

    def set_default_max_flex(self, max_flex: int) -> None:
        # this must be bigger than the largest max - and could even be more than number of people
        if self.max_flex == MAX_FLEX_UNSET:
            self.max_flex = max_flex

    def add_remaining(self) -> None:
        self.remaining += 1

    def add_selected(self) -> None:
        self.selected += 1

    def remove_remaining(self) -> None:
        self.remaining -= 1
        if self.remaining == 0 and self.selected < self.min:
            msg = "SELECTION IMPOSSIBLE: FAIL - no one/not enough left after deletion."
            raise errors.SelectionError(msg)

    def percent_selected(self, number_people_wanted: int) -> float:
        return self.selected * 100 / float(number_people_wanted)


class FeatureValues:
    """
    A full set of values for a single feature.

    If the feature is gender, the values could be: male, female, non_binary_other

    The values are FeatureValueCounts objects - the min, max and current counts of the
    selected people in that feature value.
    """

    def __init__(self) -> None:
        self.feature_values: dict[str, FeatureValueCounts] = {}

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.feature_values == other.feature_values

    def add_value_counts(self, value_name: str, fv_counts: FeatureValueCounts) -> None:
        self.feature_values[value_name] = fv_counts

    def set_default_max_flex(self, max_flex: int) -> None:
        """Note this only sets it if left at the default value"""
        for fv_counts in self.feature_values.values():
            fv_counts.set_default_max_flex(max_flex)

    @property
    def values(self) -> list[str]:
        return list(self.feature_values.keys())

    def values_counts(self) -> Iterator[tuple[str, FeatureValueCounts]]:
        yield from self.feature_values.items()

    def add_remaining(self, value_name: str) -> None:
        self.feature_values[value_name].add_remaining()

    def add_selected(self, value_name: str) -> None:
        self.feature_values[value_name].add_selected()

    def remove_remaining(self, value_name: str) -> None:
        self.feature_values[value_name].remove_remaining()

    def minimum_selection(self) -> int:
        """
        For this feature, we have to select at least the sum of the minimum of each value
        """
        return sum(c.min for c in self.feature_values.values())

    def maximum_selection(self) -> int:
        """
        For this feature, we have to select at most the sum of the maximum of each value
        """
        return sum(c.max for c in self.feature_values.values())

    def get_counts(self, value_name: str) -> FeatureValueCounts:
        return self.feature_values[value_name]


class FeatureCollection:
    """
    A full set of features for a stratification.

    The keys here are the names of the features. They could be: gender, age_bracket, education_level etc

    The values are FeatureValues objects - the breakdown of the values for a feature.
    """

    # TODO: consider splitting the updates/remaining into a parallel set of classes
    # then this can just have targets, and the running totals can be in classes we can
    # regenerate now and then

    def __init__(self) -> None:
        self.collection: dict[str, FeatureValues] = defaultdict(FeatureValues)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.collection == other.collection

    def add_feature(self, feature_name: str, value_name: str, fv_counts: FeatureValueCounts) -> None:
        self.collection[feature_name].add_value_counts(value_name, fv_counts)

    @property
    def feature_names(self) -> list[str]:
        return list(self.collection.keys())

    def feature_values(self) -> Iterator[tuple[str, list[str]]]:
        for feature_name, feature_value in self.collection.items():
            yield feature_name, feature_value.values

    def feature_value_pairs(self) -> Iterator[tuple[str, str]]:
        for feature_name, feature_value in self.collection.items():
            for value_name in feature_value.values:
                yield feature_name, value_name

    def feature_values_counts(self) -> Iterator[tuple[str, str, FeatureValueCounts]]:
        for feature_name, feature_values in self.collection.items():
            for value, value_counts in feature_values.values_counts():
                yield feature_name, value, value_counts

    def get_counts(self, feature_name: str, value_name: str) -> FeatureValueCounts:
        return self.collection[feature_name].get_counts(value_name)

    def _safe_max_flex_val(self) -> int:
        if not self.collection:
            return 0
        # to avoid errors, if max_flex is not set we must set it at least as high as the highest
        return max(v.maximum_selection() for v in self.collection.values())

    def set_default_max_flex(self) -> None:
        """Note this only sets it if left at the default value"""
        max_flex = self._safe_max_flex_val()
        for feature_values in self.collection.values():
            feature_values.set_default_max_flex(max_flex)

    def add_remaining(self, feature: str, value_name: str) -> None:
        self.collection[feature].add_remaining(value_name)

    def add_selected(self, feature: str, value_name: str) -> None:
        self.collection[feature].add_selected(value_name)

    def remove_remaining(self, feature: str, value_name: str) -> None:
        try:
            self.collection[feature].remove_remaining(value_name)
        except errors.SelectionError as e:
            msg = f"Failed removing from {feature}/{value_name}: {e}"
            raise errors.SelectionError(msg) from None

    def minimum_selection(self) -> int:
        """
        The minimum selection for this set of features is the largest minimum selection
        of any individual feature.
        """
        if not self.collection:
            return 0
        return max(v.minimum_selection() for v in self.collection.values())

    def maximum_selection(self) -> int:
        """
        The maximum selection for this set of features is the smallest maximum selection
        of any individual feature.
        """
        if not self.collection:
            return 0
        return min(v.maximum_selection() for v in self.collection.values())

    def check_min_max(self) -> None:
        """
        If the min is bigger than the max we're in trouble i.e. there's an input error
        """
        if self.minimum_selection() > self.maximum_selection():
            msg = (
                "Inconsistent numbers in min and max in the features input: the sum "
                "of the minimum values of a features is larger than the sum of the "
                "maximum values of a(nother) feature. "
            )
            raise ValueError(msg)

    def check_desired(self, desired_number: int) -> None:
        """
        Check if the desired number of people is within the min/max of every feature.
        """
        for feature_name, feature_values in self.collection.items():
            if (
                desired_number < feature_values.minimum_selection()
                or desired_number > feature_values.maximum_selection()
            ):
                msg = (
                    f"The number of people to select ({desired_number}) is out of the range of "
                    f"the numbers of people in the {feature_name} feature. It should be within "
                    f"[{feature_values.minimum_selection()}, {feature_values.maximum_selection()}]."
                )
                raise Exception(msg)


def _normalise_col_names(row: dict[str, str]) -> dict[str, str]:
    """
    if the dict has "category" as the key, change that to "feature"
    if the dict has "name" as the key, change that to "value"
    """
    if "category" in row:
        row["feature"] = row.pop("category")
    if "name" in row:
        row["value"] = row.pop("name")
    return row


def _feature_headers_flex(headers: list[str]) -> tuple[bool, list[str]]:
    """
    Determine if the headers match either the required ones, or required plus flex fields.
    Return True if the flex headers are present, False if not.

    It is fine to have extra headers - they will just be ignored.

    If an invalid set of headers are present, report details.
    """
    filtered_headers = [h for h in headers if h in ALL_FEATURE_FIELD_NAMES]
    # check that the fieldnames are (at least) what we expect, and only once.
    # BUT (for reverse compatibility) let min_flex and max_flex be optional.
    if sorted(filtered_headers) in (
        sorted(FEATURE_FILE_FIELD_NAMES),
        sorted(FEATURE_FILE_FIELD_NAMES_OLD),
    ):
        return False, filtered_headers
    if sorted(filtered_headers) in (
        sorted(FEATURE_FILE_FIELD_NAMES_FLEX),
        sorted(FEATURE_FILE_FIELD_NAMES_FLEX_OLD),
    ):
        return True, filtered_headers
    # below here we are reporting errors with the headers
    messages: list[str] = []
    required_fields = FEATURE_FILE_FIELD_NAMES if "feature" in filtered_headers else FEATURE_FILE_FIELD_NAMES_OLD
    for field_name in required_fields:
        feature_head_field_name_count = filtered_headers.count(field_name)
        if feature_head_field_name_count == 0 and (field_name != "min_flex" and field_name != "max_flex"):
            messages.append(f"Did not find required column name '{field_name}' in the input")
        elif feature_head_field_name_count > 1:
            messages.append(
                f"Found MORE THAN 1 column named '{field_name}' in the input (found {feature_head_field_name_count})"
            )
    msg = "\n".join(messages) if messages else f"Unexpected error in set of column names: {headers}"
    raise ValueError(msg)


def _clean_row(row: utils.StrippedDict, feature_flex: bool) -> tuple[str, str, FeatureValueCounts]:
    """
    allow for some dirty data - at least strip white space from feature name and value
    but only if they are strings! (sometimes people use ints as feature names or values
    and then strip produces an exception...)
    """
    feature_name = row["feature"]
    # check for blank entries and report a meaningful error
    feature_value = row["value"]
    if feature_value == "" or row["min"] == "" or row["max"] == "":
        msg = f"ERROR reading in feature file: found a blank cell in a row of the feature: {feature_name}."
        raise ValueError(msg)
    # must convert min/max to ints
    value_min = int(row["min"])
    value_max = int(row["max"])
    if feature_flex:
        if row["min_flex"] == "" or row["max_flex"] == "":
            msg = (
                f"ERROR reading in feature file: found a blank min_flex or "
                f"max_flex cell in a feature value: {feature_value}."
            )
            raise ValueError(msg)
        value_min_flex = int(row["min_flex"])
        value_max_flex = int(row["max_flex"])
        # if these values exist they must be at least this...
        if value_min_flex > value_min or value_max_flex < value_max:
            msg = (
                f"Inconsistent numbers in min_flex and max_flex in the features input for {feature_value}: "
                f"the flex values must be equal or outside the max and min values."
            )
            raise ValueError(msg)
    else:
        value_min_flex = 0
        # since we don't know self.number_people_to_select yet! We correct this below
        value_max_flex = MAX_FLEX_UNSET
    fv_counts = FeatureValueCounts(
        min=value_min,
        max=value_max,
        min_flex=value_min_flex,
        max_flex=value_max_flex,
    )
    return feature_name, feature_value, fv_counts


def read_in_features(
    features_head: Iterable[str], features_body: Iterable[dict[str, str]]
) -> tuple[FeatureCollection, list[str]]:
    """
    Read in stratified selection features and values

    Note we do want features_head to ensure we don't have multiple columns with the same name
    """
    features = FeatureCollection()
    msg: list[str] = []
    features_flex, filtered_headers = _feature_headers_flex(list(features_head))
    for row in features_body:
        # check the set of keys in the row are the same as the headers
        assert set(filtered_headers) <= set(row.keys())
        stripped_row = utils.StrippedDict(_normalise_col_names(row))
        if not stripped_row["feature"]:
            continue
        features.add_feature(*_clean_row(stripped_row, features_flex))

    msg.append(f"Number of features: {len(features.feature_names)}")
    features.check_min_max()
    # check feature_flex to see if we need to set the max here
    # this only changes the max_flex value if these (optional) flex values are NOT set already
    features.set_default_max_flex()
    return features, msg
