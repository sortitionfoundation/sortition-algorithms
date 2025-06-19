import tomllib
from pathlib import Path
from typing import Any

from attrs import define, field, validators
from cattrs import structure

SELECTION_ALGORITHMS = ("legacy", "maximin", "nash")

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
    if len(value) not in (0, 2):
        raise ValueError("check_same_address_columns must be a list of ZERO OR TWO strings")
    if not all(isinstance(element, str) for element in value):
        raise TypeError("check_same_address_columns must be a list of STRINGS")
    if len(value) == 0 and instance.check_same_address:
        raise ValueError(
            "check_same_address is TRUE but there are no columns listed to check! FIX THIS and RESTART this program!"
        )


@define
class Settings:
    id_column: str = field(validator=validators.instance_of(str))
    columns_to_keep: list[str] = field()
    check_same_address: bool = field(validator=validators.instance_of(bool))
    check_same_address_columns: list[str] = field(validator=check_columns_for_same_address)
    max_attempts: int = field(validator=validators.instance_of(int))
    selection_algorithm: str = field()
    random_number_seed: int = field(validator=validators.instance_of(int))

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

    @classmethod
    def load_from_file(
        cls,
        settings_file_path: Path,
    ) -> tuple["Settings", str]:
        messages: list[str] = []
        if not settings_file_path.is_file():
            with open(settings_file_path, "w", encoding="utf-8") as settings_file:
                settings_file.write(DEFAULT_SETTINGS)
            messages.append(
                f"Wrote default settings to '{settings_file_path.absolute()}' "
                "- if editing is required, restart this app."
            )
        with open(settings_file_path, "rb") as settings_file:
            settings = tomllib.load(settings_file)
        # you can't check an address if there is no info about which columns to check...
        if settings["check_same_address"] is False:
            messages.append(
                "<b>WARNING</b>: Settings file is such that we do NOT check if respondents have same address."
            )
            settings["check_same_address_columns"] = []
        return structure(settings, cls), "\n".join(messages)
