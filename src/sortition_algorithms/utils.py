import secrets
from collections.abc import Mapping


def print_ret(message: str) -> str:
    """Print and return a message for output collection."""
    # TODO: should we replace this with logging or similar?
    print(message)
    return message


def strip_str_int(value: str | int) -> str:
    return str(value).strip()


class StrippedDict:
    """
    Wraps a dict, and whenever we get a value from it, we convert to str and
    strip() whitespace
    """

    def __init__(self, raw_dict: Mapping[str, str] | Mapping[str, str | int]) -> None:
        self.raw_dict = raw_dict

    def __getitem__(self, key: str) -> str:
        return strip_str_int(self.raw_dict[key])


def secrets_uniform(lower: float, upper: float) -> float:
    assert upper > lower
    diff = upper - lower
    rand_int = secrets.randbelow(1_000_000)
    return lower + (rand_int * diff / 1_000_000)
