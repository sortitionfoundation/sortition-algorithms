from typing import TYPE_CHECKING

from sortition_algorithms.utils import RunReport

if TYPE_CHECKING:
    # this is done to avoid circular imports
    from sortition_algorithms.features import FeatureCollection


class BadDataError(Exception):
    pass


class SelectionError(Exception):
    def __init__(self, *args: object) -> None:
        """
        If one of the args is a RunReport, extract it and save it
        """
        report_args = [a for a in args if isinstance(a, RunReport)]
        non_report_args = [a for a in args if not isinstance(a, RunReport)]
        super().__init__(*non_report_args)
        self.report = report_args[0] if report_args else RunReport()


class InfeasibleQuotasError(Exception):
    def __init__(self, features: "FeatureCollection", output: list[str]) -> None:
        self.features = features
        self.output = ["The quotas are infeasible:", *output]

    def __str__(self) -> str:
        return "\n".join(self.output)


class InfeasibleQuotasCantRelaxError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
