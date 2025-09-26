import enum
import html
import logging
import random
import secrets
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tabulate import tabulate

if TYPE_CHECKING:
    from _typeshed import SupportsLenAndGetItem


def default_logging_setup() -> tuple[logging.Logger, logging.Logger]:
    """Set both logger and user_logger to send output to stdout"""
    # we have two loggers
    # - user_logger is used for messages that any user should see
    # - logger is used for messages that only a developer or admin should need to see
    user_logger = logging.getLogger("sortition_algorithms.user")
    user_logger.setLevel(logging.INFO)
    if not user_logger.handlers:
        # no set up has been done yet - so we do it here
        pass
    logger = logging.getLogger("sortition_algorithms")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # no set up has been done yet - so we do it here
        # this logger just goes straight to stdout - no timestamps or anything
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return user_logger, logger


def override_logging_handlers(
    user_logger_handlers: list[logging.Handler], logger_handlers: list[logging.Handler]
) -> None:
    """Replace the default handlers with other ones"""
    user_logger = logging.getLogger("sortition_algorithms.user")
    logger = logging.getLogger("sortition_algorithms")
    # first get rid of the old handlers
    for handler in user_logger.handlers[:]:
        user_logger.removeHandler(handler)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # now add the new handlers
    for handler in user_logger_handlers:
        user_logger.addHandler(handler)
    for handler in logger_handlers:
        logger.addHandler(handler)


def set_log_level(log_level: int) -> None:
    user_logger.setLevel(log_level)
    logger.setLevel(log_level)


user_logger, logger = default_logging_setup()


class ReportLevel(enum.Enum):
    NORMAL = 0
    IMPORTANT = 1
    CRITICAL = 2


@dataclass
class RunLineLevel:
    line: str
    level: ReportLevel
    log_level: int = logging.NOTSET


@dataclass
class RunTable:
    headers: list[str]
    data: list[list[str | int | float]]


class RunReport:
    """A class to hold a report to show to the user at the end"""

    def __init__(self) -> None:
        self._data: list[RunLineLevel | RunTable] = []

    def __bool__(self) -> bool:
        """
        Basically, False is the report is empty, or True if there is some content. So you can do
        things like

        ```
        if run_report:
            print(f"Run Report\n\n{run_report.as_text()}")
        ```
        """
        return self.has_content()

    def has_content(self) -> bool:
        """
        False is the report is empty, or True if there is some content. So you can do
        things like

        ```
        if run_report.has_content():
            print(f"Run Report\n\n{run_report.as_text()}")
        ```
        """
        return bool(self._data)

    def add_line(self, line: str, level: ReportLevel = ReportLevel.NORMAL) -> None:
        """
        Add a line of text, and a level - so important/critical messages can be highlighted in the HTML report.
        """
        self._data.append(RunLineLevel(line, level))

    def add_line_and_log(self, line: str, log_level: int) -> None:
        """
        Add a line of text, and a level - so important/critical messages can be highlighted in the HTML report.

        This method will also log the message to the `user_logger`. This message can be shown to the user as
        the run is happening, so the user has feedback on what is going on while the run is in progress.

        When generating the report we can skip those messages, to avoid duplication. But if the user_logger
        has not been set up to be shown to the user during the run, we do want those messages to be in the
        final report.
        """
        self._data.append(RunLineLevel(line, ReportLevel.NORMAL, log_level))
        user_logger.log(level=log_level, msg=line)

    def add_lines(self, lines: Iterable[str], level: ReportLevel = ReportLevel.NORMAL) -> None:
        """Add a line of text, and a level - we use the logging levels"""
        for line in lines:
            self._data.append(RunLineLevel(line, level))

    def add_table(self, table_headings: list[str], table_data: list[list[str | int | float]]) -> None:
        self._data.append(RunTable(table_headings, table_data))

    def add_report(self, other: "RunReport") -> None:
        self._data += other._data

    def _element_to_text(self, element: RunLineLevel | RunTable, include_logged: bool) -> str | None:
        if isinstance(element, RunLineLevel):
            # we might want to skip lines that were already logged
            if include_logged or element.log_level == logging.NOTSET:
                return element.line
            else:
                # sometimes we want empty strings for blank lines, so here we return None
                # instead so the logged lines can be filtered out
                return None
        else:
            table_text = tabulate(element.data, headers=element.headers, tablefmt="simple")
            # we want a blank line before and after the table.
            return f"\n{table_text}\n"

    def as_text(self, include_logged: bool = True) -> str:
        parts = [self._element_to_text(element, include_logged) for element in self._data]
        return "\n".join(p for p in parts if p is not None)

    def _line_to_html(self, line_level: RunLineLevel) -> str:
        tags = {
            ReportLevel.NORMAL: ("", ""),
            ReportLevel.IMPORTANT: ("<b>", "</b>"),
            ReportLevel.CRITICAL: ('<b style="color: red">', "</b>"),
        }
        start_tag, end_tag = tags[line_level.level]
        escaped_line = html.escape(line_level.line)
        return f"{start_tag}{escaped_line}{end_tag}"

    def _element_to_html(self, element: RunLineLevel | RunTable, include_logged: bool) -> str | None:
        if isinstance(element, RunLineLevel):
            if include_logged or element.log_level == logging.NOTSET:
                return self._line_to_html(element)
            else:
                return None
        else:
            # TODO: add attributes to the `<table>` tag - the original code had:
            # <table border='1' cellpadding='5'> - though do we really want that?
            # Probably better to use CSS so others can style as they see fit.
            return tabulate(element.data, headers=element.headers, tablefmt="html")

    def as_html(self, include_logged: bool = True) -> str:
        parts = [self._element_to_html(element, include_logged) for element in self._data]
        return "<br />\n".join(p for p in parts if p is not None)


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
