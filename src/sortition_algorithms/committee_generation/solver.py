# ABOUTME: Abstraction layer for LP/MIP solvers (HiGHS, python-mip).
# ABOUTME: Provides a unified interface for committee generation algorithms.

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

from sortition_algorithms.utils import random_provider


class SolverStatus(Enum):
    """Status returned by solver after optimization."""

    OPTIMAL = auto()
    FEASIBLE = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    ERROR = auto()


class SolverSense(Enum):
    """Optimization direction."""

    MINIMIZE = auto()
    MAXIMIZE = auto()


class Solver(ABC):
    """Abstract base class for LP/MIP solvers.

    Provides a unified interface for committee generation algorithms to use
    different solver backends (HiGHS, python-mip, Gurobi) interchangeably.
    """

    @abstractmethod
    def add_binary_var(self, name: str = "") -> Any:
        """Add a binary (0/1) variable to the model.

        Args:
            name: Optional name for the variable

        Returns:
            A variable object that can be used in constraints and objectives
        """
        ...

    @abstractmethod
    def add_integer_var(self, lb: float = 0.0, ub: float | None = None, name: str = "") -> Any:
        """Add an integer variable to the model.

        Args:
            lb: Lower bound (default 0.0)
            ub: Upper bound (default None means infinity)
            name: Optional name for the variable

        Returns:
            A variable object that can be used in constraints and objectives
        """
        ...

    @abstractmethod
    def add_continuous_var(self, lb: float = 0.0, ub: float = 1.0, name: str = "") -> Any:
        """Add a continuous variable to the model.

        Args:
            lb: Lower bound (default 0.0)
            ub: Upper bound (default 1.0)
            name: Optional name for the variable

        Returns:
            A variable object that can be used in constraints and objectives
        """
        ...

    @abstractmethod
    def add_constr(self, constraint: Any) -> None:
        """Add a constraint to the model.

        Args:
            constraint: A constraint expression (e.g., var1 + var2 <= 5)
        """
        ...

    @abstractmethod
    def set_objective(self, expr: Any, sense: SolverSense) -> None:
        """Set the objective function.

        Args:
            expr: The expression to optimize
            sense: MINIMIZE or MAXIMIZE
        """
        ...

    @abstractmethod
    def optimize(self) -> SolverStatus:
        """Solve the optimization problem.

        Returns:
            The optimization status
        """
        ...

    @abstractmethod
    def get_value(self, var: Any) -> float:
        """Get the value of a variable after optimization.

        Args:
            var: The variable to query

        Returns:
            The optimal value of the variable
        """
        ...

    @abstractmethod
    def get_objective_value(self) -> float:
        """Get the objective value after optimization.

        Returns:
            The optimal objective value
        """
        ...

    @abstractmethod
    def get_var_by_name(self, name: str) -> Any:
        """Get a variable by its name.

        Args:
            name: The name of the variable

        Returns:
            The variable object, or None if not found
        """
        ...

    @abstractmethod
    def get_gap(self) -> float:
        """Get the MIP gap after optimization.

        Returns:
            The optimality gap (0.0 for optimal, higher for feasible solutions)
        """
        ...


class HighsSolver(Solver):
    """HiGHS solver implementation using highspy."""

    def __init__(
        self,
        verbose: bool = False,
        seed: int | None = None,
        time_limit: float | None = None,
        mip_gap: float | None = None,
    ):
        """Create a new HiGHS solver instance.

        Args:
            verbose: If True, enable solver output
            seed: Random seed for reproducibility
            time_limit: Maximum solve time in seconds
            mip_gap: Acceptable MIP gap (e.g., 0.1 for 10%)
        """
        import highspy

        self._highspy = highspy
        self._h = highspy.Highs()
        self._h.setOptionValue("output_flag", verbose)

        if seed is not None:
            self._h.setOptionValue("random_seed", seed)
        else:
            self._h.setOptionValue("random_seed", random_provider().randint())

        if time_limit is not None:
            self._h.setOptionValue("time_limit", time_limit)

        if mip_gap is not None:
            self._h.setOptionValue("mip_rel_gap", mip_gap)

        self._var_names: dict[str, Any] = {}
        self._objective_sense: SolverSense | None = None

    def add_binary_var(self, name: str = "") -> Any:
        var = self._h.addVariable(lb=0.0, ub=1.0, type=self._highspy.HighsVarType.kInteger)
        if name:
            self._var_names[name] = var
        return var

    def add_integer_var(self, lb: float = 0.0, ub: float | None = None, name: str = "") -> Any:
        if ub is None:
            ub = self._highspy.kHighsInf
        var = self._h.addVariable(lb=lb, ub=ub, type=self._highspy.HighsVarType.kInteger)
        if name:
            self._var_names[name] = var
        return var

    def add_continuous_var(self, lb: float = 0.0, ub: float = 1.0, name: str = "") -> Any:
        var = self._h.addVariable(lb=lb, ub=ub)
        if name:
            self._var_names[name] = var
        return var

    def add_constr(self, constraint: Any) -> None:
        self._h.addConstr(constraint)

    def set_objective(self, expr: Any, sense: SolverSense) -> None:
        self._objective_sense = sense
        if sense == SolverSense.MAXIMIZE:
            self._h.maximize(expr)
        else:
            self._h.minimize(expr)

    def optimize(self) -> SolverStatus:
        self._h.run()
        status = self._h.getModelStatus()

        if status == self._highspy.HighsModelStatus.kOptimal:
            return SolverStatus.OPTIMAL
        elif status == self._highspy.HighsModelStatus.kInfeasible:
            return SolverStatus.INFEASIBLE
        elif status == self._highspy.HighsModelStatus.kUnbounded:
            return SolverStatus.UNBOUNDED
        elif status in (
            self._highspy.HighsModelStatus.kObjectiveBound,
            self._highspy.HighsModelStatus.kObjectiveTarget,
            self._highspy.HighsModelStatus.kTimeLimit,
            self._highspy.HighsModelStatus.kIterationLimit,
            self._highspy.HighsModelStatus.kSolutionLimit,
        ):
            # These statuses indicate a feasible (but possibly not optimal) solution was found
            info = self._h.getInfo()
            if info.primal_solution_status == self._highspy.HighsSolutionStatus.kSolutionStatusFeasible:
                return SolverStatus.FEASIBLE
            return SolverStatus.ERROR
        else:
            return SolverStatus.ERROR

    def get_value(self, var: Any) -> float:
        return float(self._h.val(var))

    def get_objective_value(self) -> float:
        return float(self._h.getInfo().objective_function_value)

    def get_var_by_name(self, name: str) -> Any:
        return self._var_names.get(name)

    def get_gap(self) -> float:
        info = self._h.getInfo()
        return float(info.mip_gap) if hasattr(info, "mip_gap") else 0.0


class MipSolver(Solver):
    """python-mip solver implementation."""

    def __init__(
        self,
        verbose: bool = False,
        seed: int | None = None,
        time_limit: float | None = None,
        mip_gap: float | None = None,
    ):
        """Create a new python-mip solver instance.

        Args:
            verbose: If True, enable solver output
            seed: Random seed for reproducibility
            time_limit: Maximum solve time in seconds
            mip_gap: Acceptable MIP gap (e.g., 0.1 for 10%)
        """
        import mip

        self._mip = mip
        # Default to MAXIMIZE, will be changed when set_objective is called
        self._model = mip.Model()
        self._model.verbose = 1 if verbose else 0

        if seed is not None:
            self._model.seed = seed
        else:
            self._model.seed = random_provider().randint()

        if time_limit is not None:
            self._model.max_seconds = time_limit

        if mip_gap is not None:
            self._model.max_mip_gap = mip_gap

        self._status: SolverStatus = SolverStatus.ERROR

    def add_binary_var(self, name: str = "") -> Any:
        return self._model.add_var(var_type=self._mip.BINARY, name=name if name else None)

    def add_integer_var(self, lb: float = 0.0, ub: float | None = None, name: str = "") -> Any:
        if ub is None:
            ub = self._mip.INF
        return self._model.add_var(var_type=self._mip.INTEGER, lb=lb, ub=ub, name=name if name else None)

    def add_continuous_var(self, lb: float = 0.0, ub: float = 1.0, name: str = "") -> Any:
        return self._model.add_var(var_type=self._mip.CONTINUOUS, lb=lb, ub=ub, name=name if name else None)

    def add_constr(self, constraint: Any) -> None:
        self._model.add_constr(constraint)

    def set_objective(self, expr: Any, sense: SolverSense) -> None:
        if sense == SolverSense.MAXIMIZE:
            self._model.sense = self._mip.MAXIMIZE
        else:
            self._model.sense = self._mip.MINIMIZE
        self._model.objective = expr

    def optimize(self) -> SolverStatus:
        status = self._model.optimize()

        if status == self._mip.OptimizationStatus.OPTIMAL:
            self._status = SolverStatus.OPTIMAL
        elif status == self._mip.OptimizationStatus.INFEASIBLE:
            self._status = SolverStatus.INFEASIBLE
        elif status == self._mip.OptimizationStatus.UNBOUNDED:
            self._status = SolverStatus.UNBOUNDED
        elif status == self._mip.OptimizationStatus.FEASIBLE:
            self._status = SolverStatus.FEASIBLE
        else:
            self._status = SolverStatus.ERROR

        return self._status

    def get_value(self, var: Any) -> float:
        return float(var.x)

    def get_objective_value(self) -> float:
        return float(self._model.objective_value)

    def get_var_by_name(self, name: str) -> Any:
        return self._model.var_by_name(name)

    def get_gap(self) -> float:
        return float(self._model.gap)


def create_solver(
    backend: str = "highspy",
    verbose: bool = False,
    seed: int | None = None,
    time_limit: float | None = None,
    mip_gap: float | None = None,
) -> Solver:
    """Create a solver instance with the specified backend.

    Args:
        backend: Solver backend to use ("highspy" or "mip")
        verbose: If True, enable solver output
        seed: Random seed for reproducibility
        time_limit: Maximum solve time in seconds
        mip_gap: Acceptable MIP gap (e.g., 0.1 for 10%)

    Returns:
        A Solver instance

    Raises:
        ValueError: If an unknown backend is specified
    """
    if backend == "highspy":
        return HighsSolver(verbose=verbose, seed=seed, time_limit=time_limit, mip_gap=mip_gap)
    elif backend == "mip":
        return MipSolver(verbose=verbose, seed=seed, time_limit=time_limit, mip_gap=mip_gap)
    else:
        raise ValueError(f"Unknown solver backend: {backend}")


def solver_sum(terms: Any) -> Any:
    """Sum a collection of solver expressions.

    This provides a backend-agnostic way to sum terms.
    Both highspy and mip support Python's built-in sum() for their expression objects.

    Args:
        terms: An iterable of solver expressions/variables

    Returns:
        A sum expression
    """
    return sum(terms)
