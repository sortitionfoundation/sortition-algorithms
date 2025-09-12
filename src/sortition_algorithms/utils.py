import enum
import html
import random
import secrets
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsLenAndGetItem


class ReportLevel(enum.Enum):
    NORMAL = 0
    IMPORTANT = 1
    CRITICAL = 2


class RunReport:
    """A class to hold a report to show to the user at the end"""

    def __init__(self) -> None:
        self._lines: list[tuple[str, ReportLevel]] = []

    def add_line(self, line: str, level: ReportLevel = ReportLevel.NORMAL) -> None:
        """Add a line of text, and a level - we use the logging levels"""
        self._lines.append((line, level))

    def add_table(self, table) -> None:
        # TODO: implement this
        raise NotImplementedError

    def as_text(self) -> str:
        # TODO: tables
        return "\n".join(line for line, _ in self._lines)

    def _line_to_html(self, line_level: tuple[str, ReportLevel]) -> str:
        tags = {
            ReportLevel.NORMAL: ("", ""),
            ReportLevel.IMPORTANT: ("<b>", "</b>"),
            ReportLevel.CRITICAL: ('<b style="color: red">', "</b>"),
        }
        line, level = line_level
        start_tag, end_tag = tags[level]
        escaped_line = html.escape(line)
        return f"{start_tag}{escaped_line}{end_tag}"

    def as_html(self) -> str:
        # TODO: tables
        return "<br />\n".join(self._line_to_html(line_level) for line_level in self._lines)


def print_ret(message: str) -> str:
    """Print and return a message for output collection."""
    # TODO: should we replace this with logging or similar?
    print(message)
    return message


def strip_str_int(value: str | int | float) -> str:
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


class RandomProvider(ABC):
    """
    This is something of a hack. Mostly we want to use the `secrets` module.
    But for repeatable testing we might want to set the random.seed sometimes.

    So we have a global `_random_provider` which can be switched between an
    instance of this class that uses the `secrets` module and an instance that
    uses `random` with a seed. The switch is done by the `set_random_provider()`
    function.

    Then every time we want some randomness, we call `random_provider()` to get
    the current version of the global.
    """

    @classmethod
    @abstractmethod
    def uniform(cls, lower: float, upper: float) -> float: ...

    @classmethod
    @abstractmethod
    def randbelow(cls, upper: int) -> int: ...

    @classmethod
    @abstractmethod
    def choice(cls, seq: "SupportsLenAndGetItem[str]") -> str: ...


class GenRandom(RandomProvider):
    def __init__(self, seed: int) -> None:
        random.seed(seed)

    @classmethod
    def uniform(cls, lower: float, upper: float) -> float:
        return random.uniform(lower, upper)  # noqa: S311

    @classmethod
    def randbelow(cls, upper: int) -> int:
        return random.randrange(upper)  # noqa: S311

    @classmethod
    def choice(cls, seq: "SupportsLenAndGetItem[str]") -> str:
        return random.choice(seq)  # noqa: S311


class GenSecrets(RandomProvider):
    @classmethod
    def uniform(cls, lower: float, upper: float) -> float:
        assert upper > lower
        diff = upper - lower
        rand_int = secrets.randbelow(1_000_000)
        return lower + (rand_int * diff / 1_000_000)

    @classmethod
    def randbelow(cls, upper: int) -> int:
        return secrets.randbelow(upper)

    @classmethod
    def choice(cls, seq: "SupportsLenAndGetItem[str]") -> str:
        return secrets.choice(seq)


_random_provider: RandomProvider = GenSecrets()


def set_random_provider(seed: int | None = None) -> None:
    global _random_provider
    if seed:
        _random_provider = GenRandom(seed)
    _random_provider = GenSecrets()


def random_provider() -> RandomProvider:
    return _random_provider
