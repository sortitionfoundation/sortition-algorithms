from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # this is done to avoid circular imports
    from sortition_algorithms.features import FeatureCollection


class SortitionBaseError(Exception):
    """A base class that allows all errors to be caught easily."""


class BadDataError(SortitionBaseError):
    """Error for when bad data is found while reading things in"""


class SelectionError(SortitionBaseError):
    """Generic error for things that happen in selection"""


class SelectionMultilineError(SelectionError):
    """Generic error for things that happen in selection - multiline"""

    def __init__(self, lines: list[str]) -> None:
        self.all_lines = lines

    def __str__(self) -> str:
        return "\n".join(self.all_lines)

    def lines(self) -> list[str]:
        return self.all_lines


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
