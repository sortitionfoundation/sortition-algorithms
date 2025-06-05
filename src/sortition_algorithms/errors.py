class BadDataError(Exception):
    pass


class SelectionError(Exception):
    pass


class InfeasibleQuotasError(Exception):
    def __init__(self, quotas: dict[tuple[str, str], tuple[int, int]], output: list[str]) -> None:
        self.quotas = quotas
        self.output = ["The quotas are infeasible:", *output]

    def __str__(self) -> str:
        return "\n".join(self.output)


class InfeasibleQuotasCantRelaxError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
