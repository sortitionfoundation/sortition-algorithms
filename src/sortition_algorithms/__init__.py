"""Sortition algorithms for democratic lotteries."""

from sortition_algorithms.adapters import (
    AbstractDataSource,
    CSVFileDataSource,
    CSVStringDataSource,
    GSheetDataSource,
    SelectionData,
)
from sortition_algorithms.core import (
    find_random_sample,
    run_stratification,
    selected_remaining_tables,
)
from sortition_algorithms.errors import (
    BadDataError,
    ConfigurationError,
    InfeasibleQuotasCantRelaxError,
    InfeasibleQuotasError,
    SelectionError,
    SortitionBaseError,
)
from sortition_algorithms.features import (
    MinMaxCrossFeatureIssue,
    read_in_features,
    report_min_max_against_number_to_select_structured,
    report_min_max_error_details_structured,
    write_features,
)
from sortition_algorithms.people import (
    FeatureValueCountCheck,
    check_people_per_feature_value,
    count_people_per_feature_value,
    read_in_people,
)
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import RunReport

__all__ = [
    "AbstractDataSource",
    "BadDataError",
    "CSVFileDataSource",
    "CSVStringDataSource",
    "ConfigurationError",
    "FeatureValueCountCheck",
    "GSheetDataSource",
    "InfeasibleQuotasCantRelaxError",
    "InfeasibleQuotasError",
    "MinMaxCrossFeatureIssue",
    "RunReport",
    "SelectionData",
    "SelectionError",
    "Settings",
    "SortitionBaseError",
    "check_people_per_feature_value",
    "count_people_per_feature_value",
    "find_random_sample",
    "read_in_features",
    "read_in_people",
    "report_min_max_against_number_to_select_structured",
    "report_min_max_error_details_structured",
    "run_stratification",
    "selected_remaining_tables",
    "write_features",
]
