import html
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # this is done to avoid circular imports
    from sortition_algorithms.features import FeatureCollection


class SortitionBaseError(Exception):
    """A base class that allows all errors to be caught easily."""

    def to_html(self) -> str:
        return html.escape(str(self))


class BadDataError(SortitionBaseError):
    """Error for when bad data is found while reading things in"""


class SelectionError(SortitionBaseError):
    """Generic error for things that happen in selection"""


class SelectionMultilineError(SelectionError):
    """Generic error for things that happen in selection - multiline"""

    def __init__(self, lines: list[str]) -> None:
        self.all_lines = lines

    def __str__(self) -> str:
        return "\n".join(self.lines())

    def to_html(self) -> str:
        return "<br />".join(html.escape(line) for line in self.lines())

    def lines(self) -> list[str]:
        return self.all_lines

    def combine(self, other: "SelectionMultilineError") -> None:
        """Add all the lines from the other error to this one."""
        self.all_lines += other.lines()


@dataclass
class ParseTableErrorMsg:
    row: int
    row_name: str  # this could be "feature_name/feature_value" or "person_id"
    key: str
    value: str
    msg: str

    def __str__(self) -> str:
        return f"{self.msg}: for row {self.row}, column header {self.key}"


@dataclass
class ParseTableMultiValueErrorMsg:
    row: int
    row_name: str  # this could be "feature_name/feature_value"
    keys: list[str]
    values: list[str]
    msg: str

    def __str__(self) -> str:
        return f"{self.msg}: for row {self.row}, column headers {', '.join(self.keys)}"


class ParseTableMultiError(SelectionMultilineError):
    """
    Specifically for collecting errors from parsing a table

    This has information that can be collected at a low level. Then higher level code can read
    the errors and make a SelectionMultilineError instance with strings with more context,
    relating to a CSV file, Spreadsheet etc.
    """

    def __init__(self, errors: list[ParseTableErrorMsg | ParseTableMultiValueErrorMsg] | None = None) -> None:
        self.all_errors: list[ParseTableErrorMsg | ParseTableMultiValueErrorMsg] = errors or []

    def __len__(self) -> int:
        """This means that we will be falsy if len is 0, so is effectively a __bool__ as well"""
        return len(self.all_errors)

    def lines(self) -> list[str]:
        return [str(e) for e in self.all_errors]

    def combine(self, other: SelectionMultilineError) -> None:
        """Add all the lines from the other error to this one."""
        assert isinstance(other, ParseTableMultiError)
        self.all_errors += other.all_errors


class ParseErrorsCollector:
    """Class that we can add errors to, but errors with empty messages will be dropped"""

    def __init__(self) -> None:
        self.errors: list[ParseTableErrorMsg | ParseTableMultiValueErrorMsg] = []

    def __len__(self) -> int:
        """This means that we will be falsy if len is 0, so is effectively a __bool__ as well"""
        return len(self.errors)

    def add(self, msg: str, key: str, value: str, row: int, row_name: str) -> None:
        if msg:
            self.errors.append(ParseTableErrorMsg(row=row, row_name=row_name, key=key, value=value, msg=msg))

    def add_multi_value(self, msg: str, keys: list[str], values: list[str], row: int, row_name: str) -> None:
        if msg:
            self.errors.append(
                ParseTableMultiValueErrorMsg(row=row, row_name=row_name, keys=keys, values=values, msg=msg)
            )

    def to_error(self) -> ParseTableMultiError:
        return ParseTableMultiError(self.errors)


class InfeasibleQuotasError(SelectionMultilineError):
    """
    The quotas can't be met, and a feasible relaxation was found.

    The details of what relaxations are recommended are included in the error.
    """

    def __init__(self, features: "FeatureCollection", output: list[str]) -> None:
        self.features = features
        super().__init__(lines=["The quotas are infeasible:", *output])


class InfeasibleQuotasCantRelaxError(SortitionBaseError):
    """The quotas can't be met, and no feasible relaxation was found"""
