class BadDataError(Exception):
    pass


class SelectionError(Exception):
    pass


class InfeasibleQuotasError(Exception):
    def __init__(self, quotas: dict[tuple[str, str], tuple[int, int]], output: list[str]):
        self.quotas = quotas
        self.output = ["The quotas are infeasible:", *output]

    def __str__(self):
        return "\n".join(self.output)


class InfeasibleQuotasCantRelaxError(Exception):
    def __init__(self, message: str):
        self.message = message
