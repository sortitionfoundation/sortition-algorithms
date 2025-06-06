import secrets
from collections.abc import Mapping


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
