import tomllib
from pathlib import Path
from typing import Any

from attrs import define, field, validators
from cattrs import structure, unstructure

from sortition_algorithms.utils import ReportLevel, RunReport

SELECTION_ALGORITHMS = ("legacy", "maximin", "nash", "leximin")

DEFAULT_SETTINGS = """
# #####################################################################
#
# IF YOU EDIT THIS FILE YOU NEED TO RESTART THE APPLICATION
#
# #####################################################################

# this is written in TOML - https://github.com/toml-lang/toml

# this is the name of the (unique) field for each person
id_column = "nationbuilder_id"

# if check_same_address is true, then no 2 people from the same address will be selected
# the comparison checks if the TWO fields listed here are the same for any person
check_same_address = true
check_same_address_columns = [
    "primary_address1",
    "zip_royal_mail"
]

max_attempts = 100
columns_to_keep = [
    "first_name",
    "last_name",
    "mobile_number",
    "email",
    "primary_address1",
    "primary_address2",
    "primary_city",
    "zip_royal_mail",
    "tag_list",
    "age",
    "gender"
]

# selection_algorithm can either be "legacy", "maximin", "leximin", or "nash"
# see https://sortitionfoundation.github.io/sortition-algorithms/concepts/#selection-algorithms
selection_algorithm = "maximin"

# random number seed - if this is NOT zero then it is used to set the random number generator seed
random_number_seed = 0
"""


def check_columns_for_same_address(instance: "Settings", attribute: Any, value: Any) -> None:
    if not isinstance(value, list):
        raise TypeError("check_same_address_columns must be a LIST of strings")
    if not all(isinstance(element, str) for element in value):
        raise TypeError("check_same_address_columns must be a list of STRINGS")
    if len(value) == 0 and instance.check_same_address:
        raise ValueError(
            "check_same_address is TRUE but there are no columns listed to check! FIX THIS and RESTART this program!"
        )


@define
class Settings:
    """
    Settings to use when selecting committees. Note that only the first two are required.
    A minimal example would be:

    Settings(id_column="my_id", columns_to_keep=["name", "email"])
    """

    # required
    id_column: str = field(validator=validators.instance_of(str))
    columns_to_keep: list[str] = field()

    # fields with defaults
    check_same_address: bool = field(validator=validators.instance_of(bool), default=False)
    check_same_address_columns: list[str] = field(validator=check_columns_for_same_address, factory=list)
    max_attempts: int = field(validator=validators.instance_of(int), default=100)
    selection_algorithm: str = field(default="maximin")
    random_number_seed: int = field(validator=validators.instance_of(int), default=0)

    @columns_to_keep.validator
    def check_columns_to_keep(self, attribute: Any, value: Any) -> None:
        if not isinstance(value, list):
            raise TypeError("columns_to_keep must be a LIST of strings")
        if not all(isinstance(element, str) for element in value):
            raise TypeError("columns_to_keep must be a list of STRINGS")

    @selection_algorithm.validator
    def check_selection_algorithm(self, attribute: Any, value: str) -> None:
        if value not in SELECTION_ALGORITHMS:
            raise ValueError(f"selection_algorithm {value} is not one of: {', '.join(SELECTION_ALGORITHMS)}")

    @property
    def normalised_address_columns(self) -> list[str]:
        """
        Returns an empty list if address columns should not be checked (or if the columns
        specified was an empty list). Otherwise return the columns. That way other code can
        just check if the columns are empty rather than checking the bool separately.
        """
        return self.check_same_address_columns if self.check_same_address else []

    @property
    def full_columns_to_keep(self) -> list[str]:
        """
        We always need to keep the address columns, so in case they are not listed in
        self.columns_to_keep we have this property to have the combined list.
        """
        extra_address_columns = [col for col in self.check_same_address_columns if col not in self.columns_to_keep]
        return [*self.columns_to_keep, *extra_address_columns]

    def as_dict(self) -> dict[str, Any]:
        dict_repr = unstructure(self)
        assert isinstance(dict_repr, dict)
        assert all(isinstance(key, str) for key in dict_repr)
        return dict_repr

    @classmethod
    def load_from_file(
        cls,
        settings_file_path: Path,
    ) -> tuple["Settings", RunReport]:
        report = RunReport()
        if not settings_file_path.is_file():
            with open(settings_file_path, "w", encoding="utf-8") as settings_file:
                settings_file.write(DEFAULT_SETTINGS)
            report.add_line(
                f"Wrote default settings to '{settings_file_path.absolute()}' "
                "- if editing is required, restart this app."
            )
        with open(settings_file_path, "rb") as settings_file:
            settings = tomllib.load(settings_file)
        # you can't check an address if there is no info about which columns to check...
        if settings["check_same_address"] is False:
            report.add_line(
                "WARNING: Settings file is such that we do NOT check if respondents have same address.",
                ReportLevel.IMPORTANT,
            )
            settings["check_same_address_columns"] = []
        return structure(settings, cls), report
