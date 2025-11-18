from collections.abc import Generator, Iterable, Iterator, Mapping, MutableMapping
from typing import TypeAlias

from attrs import define
from requests.structures import CaseInsensitiveDict

from sortition_algorithms import utils
from sortition_algorithms.errors import (
    ParseErrorsCollector,
    ParseTableMultiError,
    SelectionMultilineError,
)

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


FeatureCollection: TypeAlias = MutableMapping[str, MutableMapping[str, FeatureValueMinMax]]  # noqa: UP040
# TODO: when python3.11 is dropped, change to:
# type FeatureCollection = MutableMapping[str, MutableMapping[str, FeatureValueMinMax]]


def iterate_feature_collection(features: FeatureCollection) -> Generator[tuple[str, str, FeatureValueMinMax]]:
    """Helper function to iterate over feature collection."""
    for feature_name, feature_values in features.items():
        for value_name, fv_minmax in feature_values.items():
            yield feature_name, value_name, fv_minmax


def feature_value_pairs(fc: FeatureCollection) -> Iterator[tuple[str, str]]:
    for feature_name, feature_value in fc.items():
        for value_name in feature_value:
            yield feature_name, value_name


def _fv_minimum_selection(fv: MutableMapping[str, FeatureValueMinMax]) -> int:
    return sum(c.min for c in fv.values())


def _fv_maximum_selection(fv: MutableMapping[str, FeatureValueMinMax]) -> int:
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


def report_min_max_error_details(fc: FeatureCollection, feature_column_name: str = "feature") -> list[str]:
    """
    Return a list of problems in detail, so that the user can debug the errors in detail
    """
    if not fc:
        return []

    max_feature, max_val = min(((key, _fv_maximum_selection(fv)) for key, fv in fc.items()), key=lambda x: x[1])
    min_feature, min_val = max(((key, _fv_minimum_selection(fv)) for key, fv in fc.items()), key=lambda x: x[1])
    return [
        f"Inconsistent numbers in min and max in the {feature_column_name} input:",
        f"The smallest maximum is {max_val} for {feature_column_name} '{max_feature}'",
        f"The largest minimum is {min_val} for {feature_column_name} '{min_feature}'",
        f"You need to reduce the minimums for {min_feature} or increase the maximums for {max_feature}.",
    ]


def check_min_max(fc: FeatureCollection, feature_column_name: str = "feature") -> None:
    """
    If the min is bigger than the max we're in trouble i.e. there's an input error
    """
    if minimum_selection(fc) > maximum_selection(fc):
        raise SelectionMultilineError(report_min_max_error_details(fc, feature_column_name))


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


def _get_feature_from_row(row: Mapping[str, str]) -> tuple[str, str]:
    feature_column_names = ("feature", "category")
    for key in feature_column_names:
        if key in row:
            return row[key], key
    raise ValueError(f"Could not find feature column, looked for column headers: {' '.join(feature_column_names)}")


def _get_feature_value_from_row(row: Mapping[str, str]) -> tuple[str, str]:
    feature_value_column_names = ("value", "name", "category value", "category_value", "category-value")
    for key in feature_value_column_names:
        if key in row:
            return row[key], key
    raise ValueError(
        f"Could not find feature value column, looked for column headers: {' '.join(feature_value_column_names)}"
    )


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


def _row_val_to_int(row: utils.StrippedDict, key: str) -> tuple[int, str]:
    """
    Convert row value to integer, with detailed error messages if it fails
    """
    if row[key] == "":
        return 0, f"There is no {key} value set"
    try:
        int_value = int(row[key])
    except ValueError:
        return 0, f"'{row[key]}' is not a number"
    return int_value, ""


def _clean_row(row: utils.StrippedDict, feature_flex: bool, row_number: int) -> tuple[str, str, FeatureValueMinMax]:
    """
    allow for some dirty data - at least strip white space from feature name and value
    but only if they are strings! (sometimes people use ints as feature names or values
    and then strip produces an exception...)
    """
    errors = ParseErrorsCollector()
    # this column has been checked to ensure it is not empty before this function is called
    feature_name, feature_column_name = _get_feature_from_row(row)
    # check for blank entries and report a meaningful error
    feature_value, value_column_name = _get_feature_value_from_row(row)
    if feature_value == "":
        errors.add(
            row=row_number,
            row_name=feature_name,
            key=value_column_name,
            value="",
            msg=f"Empty {value_column_name} in {feature_column_name} {feature_name}",
        )
        raise errors.to_error()

    row_name = f"{feature_name}/{feature_value}"
    value_min, error_min = _row_val_to_int(row, "min")
    errors.add(row=row_number, row_name=row_name, key="min", value=row["min"], msg=error_min)
    value_max, error_max = _row_val_to_int(row, "max")
    errors.add(row=row_number, row_name=row_name, key="max", value=row["max"], msg=error_max)
    if errors:
        # if we don't have valid min/max values, exit here
        raise errors.to_error()

    if value_min > value_max:
        errors.add_multi_value(
            row=row_number,
            row_name=row_name,
            keys=["min", "max"],
            values=[row["min"], row["max"]],
            msg=f"Minimum ({value_min}) should not be greater than maximum ({value_max})",
        )

    if feature_flex:
        value_min_flex, error_min_flex = _row_val_to_int(row, "min_flex")
        errors.add(row=row_number, row_name=row_name, key="min_flex", value=row["min_flex"], msg=error_min_flex)
        value_max_flex, error_max_flex = _row_val_to_int(row, "max_flex")
        errors.add(row=row_number, row_name=row_name, key="max_flex", value=row["max_flex"], msg=error_max_flex)
        # if these values exist they must be at least this...
        if not errors:
            if value_min_flex > value_min:
                errors.add_multi_value(
                    row=row_number,
                    row_name=row_name,
                    keys=["min", "min_flex"],
                    values=[row["min"], row["min_flex"]],
                    msg=f"min_flex ({value_min_flex}) should not be greater than min ({value_min})",
                )
            if value_max_flex < value_max:
                errors.add_multi_value(
                    row=row_number,
                    row_name=row_name,
                    keys=["max", "max_flex"],
                    values=[row["max"], row["max_flex"]],
                    msg=f"max_flex ({value_max_flex}) should not be less than max ({value_max})",
                )
    else:
        value_min_flex = 0
        # since we don't know self.number_people_to_select yet! We correct this below
        value_max_flex = MAX_FLEX_UNSET

    if errors:
        raise errors.to_error()

    fv_minmax = FeatureValueMinMax(
        min=value_min,
        max=value_max,
        min_flex=value_min_flex,
        max_flex=value_max_flex,
    )
    return feature_name, feature_value, fv_minmax


def read_in_features(
    features_head: Iterable[str], features_body: Iterable[dict[str, str]]
) -> tuple[FeatureCollection, str, str]:
    """
    Read in stratified selection features and values

    Note we do want features_head to ensure we don't have multiple columns with the same name
    """
    features: FeatureCollection = CaseInsensitiveDict()
    features_flex, filtered_headers = _feature_headers_flex(list(features_head))
    combined_error = ParseTableMultiError()
    feature_column_name = "feature"
    feature_value_column_name = "value"
    # row 1 is the header, so the body starts on row 2
    for row_number, row in enumerate(features_body, start=2):
        if row_number == 2:
            _, feature_column_name = _get_feature_from_row(row)
            _, feature_value_column_name = _get_feature_value_from_row(row)
        # check the set of keys in the row are the same as the headers
        assert set(filtered_headers) <= set(row.keys())
        stripped_row = utils.StrippedDict(row)
        fname, _ = _get_feature_from_row(row)
        if not fname:
            continue
        try:
            fname, fvalue, fv_minmax = _clean_row(stripped_row, features_flex, row_number)
        except ParseTableMultiError as error:
            # add all the lines into one large error, so we report all the errors in one go
            combined_error.combine(error)
        else:
            if fname not in features:
                features[fname] = CaseInsensitiveDict()
            features[fname][fvalue] = fv_minmax

    # if we got any errors in the above loop, raise the combined error.
    if combined_error:
        raise combined_error

    check_min_max(features, feature_column_name)
    # check feature_flex to see if we need to set the max here
    # this only changes the max_flex value if these (optional) flex values are NOT set already
    set_default_max_flex(features)
    return CaseInsensitiveDict(features), feature_column_name, feature_value_column_name
