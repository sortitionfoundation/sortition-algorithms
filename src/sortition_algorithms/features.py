from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator
from typing import TypeAlias

from attrs import define

from sortition_algorithms import utils

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
class FeatureValueMinMax:
    min: int
    max: int
    min_flex: int = 0
    max_flex: int = MAX_FLEX_UNSET

    def set_default_max_flex(self, max_flex: int) -> None:
        # this must be bigger than the largest max - and could even be more than number of people
        if self.max_flex == MAX_FLEX_UNSET:
            self.max_flex = max_flex


FeatureCollection: TypeAlias = dict[str, dict[str, FeatureValueMinMax]]  # noqa: UP040
# TODO: when python3.11 is dropped, change to:
# type FeatureCollection = dict[str, dict[str, FeatureValueMinMax]]


def iterate_feature_collection(features: FeatureCollection) -> Generator[tuple[str, str, FeatureValueMinMax]]:
    """Helper function to iterate over feature collection."""
    for feature_name, feature_values in features.items():
        for value_name, fv_minmax in feature_values.items():
            yield feature_name, value_name, fv_minmax


def feature_value_pairs(fc: FeatureCollection) -> Iterator[tuple[str, str]]:
    for feature_name, feature_value in fc.items():
        for value_name in feature_value:
            yield feature_name, value_name


def _fv_minimum_selection(fv: dict[str, FeatureValueMinMax]) -> int:
    return sum(c.min for c in fv.values())


def _fv_maximum_selection(fv: dict[str, FeatureValueMinMax]) -> int:
    return sum(c.max for c in fv.values())


def minimum_selection(fc: FeatureCollection) -> int:
    """
    The minimum selection for this set of features is the largest minimum selection
    of any individual feature.
    """
    if not fc:
        return 0

    return max(_fv_minimum_selection(fv) for fv in fc.values())


def maximum_selection(fc: FeatureCollection) -> int:
    """
    The maximum selection for this set of features is the smallest maximum selection
    of any individual feature.
    """
    if not fc:
        return 0

    return min(_fv_maximum_selection(fv) for fv in fc.values())


def check_min_max(fc: FeatureCollection) -> None:
    """
    If the min is bigger than the max we're in trouble i.e. there's an input error
    """
    if minimum_selection(fc) > maximum_selection(fc):
        msg = (
            "Inconsistent numbers in min and max in the features input: the sum "
            "of the minimum values of a features is larger than the sum of the "
            "maximum values of a(nother) feature. "
        )
        raise ValueError(msg)


def check_desired(fc: FeatureCollection, desired_number: int) -> None:
    """
    Check if the desired number of people is within the min/max of every feature.
    """
    for feature_name, fvalues in fc.items():
        if desired_number < _fv_minimum_selection(fvalues) or desired_number > _fv_maximum_selection(fvalues):
            msg = (
                f"The number of people to select ({desired_number}) is out of the range of "
                f"the numbers of people in the {feature_name} feature. It should be within "
                f"[{_fv_minimum_selection(fvalues)}, {_fv_maximum_selection(fvalues)}]."
            )
            raise Exception(msg)


def _safe_max_flex_val(fc: FeatureCollection) -> int:
    if not fc:
        return 0
    # to avoid errors, if max_flex is not set we must set it at least as high as the highest
    return max(_fv_maximum_selection(fv) for fv in fc.values())


def set_default_max_flex(fc: FeatureCollection) -> None:
    """Note this only sets it if left at the default value"""
    max_flex = _safe_max_flex_val(fc)
    for feature_values in fc.values():
        for fv_minmax in feature_values.values():
            fv_minmax.set_default_max_flex(max_flex)


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


def _clean_row(row: utils.StrippedDict, feature_flex: bool) -> tuple[str, str, FeatureValueMinMax]:
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
    fv_minmax = FeatureValueMinMax(
        min=value_min,
        max=value_max,
        min_flex=value_min_flex,
        max_flex=value_max_flex,
    )
    return feature_name, feature_value, fv_minmax


def read_in_features(
    features_head: Iterable[str], features_body: Iterable[dict[str, str]]
) -> tuple[FeatureCollection, list[str]]:
    """
    Read in stratified selection features and values

    Note we do want features_head to ensure we don't have multiple columns with the same name
    """
    features: FeatureCollection = defaultdict(dict)
    msg: list[str] = []
    features_flex, filtered_headers = _feature_headers_flex(list(features_head))
    for row in features_body:
        # check the set of keys in the row are the same as the headers
        assert set(filtered_headers) <= set(row.keys())
        stripped_row = utils.StrippedDict(_normalise_col_names(row))
        if not stripped_row["feature"]:
            continue
        fname, fvalue, fv_minmax = _clean_row(stripped_row, features_flex)
        features[fname][fvalue] = fv_minmax

    msg.append(f"Number of features: {len(features)}")
    check_min_max(features)
    # check feature_flex to see if we need to set the max here
    # this only changes the max_flex value if these (optional) flex values are NOT set already
    set_default_max_flex(features)
    return features, msg
