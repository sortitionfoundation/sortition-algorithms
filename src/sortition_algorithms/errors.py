from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sortition_algorithms.features import FeatureCollection


class BadDataError(Exception):
    pass


class SelectionError(Exception):
    pass


class InfeasibleQuotasError(Exception):
    def __init__(self, features: "FeatureCollection", output: list[str]) -> None:
        self.features = features
        self.output = ["The quotas are infeasible:", *output]

    def __str__(self) -> str:
        return "\n".join(self.output)


class InfeasibleQuotasCantRelaxError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
