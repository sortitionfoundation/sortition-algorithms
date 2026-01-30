# ABOUTME: Unit tests for the solver abstraction layer.
# ABOUTME: Tests both HighsSolver and MipSolver implementations.

import pytest

from sortition_algorithms.committee_generation.solver import (
    HighsSolver,
    MipSolver,
    Solver,
    SolverSense,
    SolverStatus,
    create_solver,
    solver_sum,
)


class TestSolverFactory:
    """Tests for the create_solver factory function."""

    def test_create_highspy_solver(self) -> None:
        """Test creating a HiGHS solver."""
        solver = create_solver(backend="highspy")
        assert isinstance(solver, HighsSolver)

    def test_create_mip_solver(self) -> None:
        """Test creating a python-mip solver."""
        solver = create_solver(backend="mip")
        assert isinstance(solver, MipSolver)

    def test_create_unknown_solver_raises(self) -> None:
        """Test that an unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown solver backend"):
            create_solver(backend="unknown")

    def test_create_solver_with_options(self) -> None:
        """Test creating a solver with various options."""
        solver = create_solver(
            backend="highspy",
            verbose=False,
            seed=42,
            time_limit=60.0,
            mip_gap=0.1,
        )
        assert isinstance(solver, HighsSolver)


class TestSolverBasicOperations:
    """Test basic solver operations on both backends."""

    @pytest.fixture(params=["highspy", "mip"])
    def solver(self, request: pytest.FixtureRequest) -> Solver:
        """Parametrized fixture returning both solver types."""
        return create_solver(backend=request.param, seed=42)

    def test_add_binary_var(self, solver: Solver) -> None:
        """Test adding a binary variable."""
        var = solver.add_binary_var(name="x")
        assert var is not None

    def test_add_integer_var(self, solver: Solver) -> None:
        """Test adding an integer variable."""
        var = solver.add_integer_var(lb=0.0, ub=10.0, name="y")
        assert var is not None

    def test_add_continuous_var(self, solver: Solver) -> None:
        """Test adding a continuous variable."""
        var = solver.add_continuous_var(lb=0.0, ub=1.0, name="z")
        assert var is not None

    def test_simple_maximize(self, solver: Solver) -> None:
        """Test a simple maximization problem."""
        x = solver.add_continuous_var(lb=0.0, ub=10.0, name="x")
        y = solver.add_continuous_var(lb=0.0, ub=10.0, name="y")

        # Maximize x + y subject to x + y <= 5
        solver.add_constr(x + y <= 5)
        solver.set_objective(x + y, SolverSense.MAXIMIZE)

        status = solver.optimize()
        assert status == SolverStatus.OPTIMAL

        # Optimal solution should be x + y = 5
        obj_value = solver.get_objective_value()
        assert abs(obj_value - 5.0) < 1e-5

    def test_simple_minimize(self, solver: Solver) -> None:
        """Test a simple minimization problem."""
        x = solver.add_continuous_var(lb=0.0, ub=10.0, name="x")

        # Minimize x subject to x >= 3
        solver.add_constr(x >= 3)
        solver.set_objective(x, SolverSense.MINIMIZE)

        status = solver.optimize()
        assert status == SolverStatus.OPTIMAL

        x_val = solver.get_value(x)
        assert abs(x_val - 3.0) < 1e-5

    def test_binary_ilp(self, solver: Solver) -> None:
        """Test a binary integer linear program."""
        x1 = solver.add_binary_var(name="x1")
        x2 = solver.add_binary_var(name="x2")
        x3 = solver.add_binary_var(name="x3")

        # Maximize x1 + 2*x2 + 3*x3 subject to x1 + x2 + x3 <= 2
        solver.add_constr(x1 + x2 + x3 <= 2)
        solver.set_objective(x1 + 2 * x2 + 3 * x3, SolverSense.MAXIMIZE)

        status = solver.optimize()
        assert status == SolverStatus.OPTIMAL

        # Optimal: x1=0, x2=1, x3=1 gives value 5
        obj_value = solver.get_objective_value()
        assert abs(obj_value - 5.0) < 1e-5

    def test_infeasible_problem(self, solver: Solver) -> None:
        """Test detection of infeasible problems."""
        x = solver.add_continuous_var(lb=0.0, ub=5.0, name="x")

        # x <= 5 (from bounds) and x >= 10 is infeasible
        solver.add_constr(x >= 10)
        solver.set_objective(x, SolverSense.MINIMIZE)

        status = solver.optimize()
        assert status == SolverStatus.INFEASIBLE

    def test_get_var_by_name(self, solver: Solver) -> None:
        """Test retrieving variables by name."""
        solver.add_binary_var(name="my_var")
        solver.add_continuous_var(name="other_var")

        retrieved_x = solver.get_var_by_name("my_var")
        retrieved_y = solver.get_var_by_name("other_var")
        retrieved_none = solver.get_var_by_name("nonexistent")

        assert retrieved_x is not None
        assert retrieved_y is not None
        assert retrieved_none is None


class TestSolverSum:
    """Test the solver_sum utility function."""

    @pytest.fixture(params=["highspy", "mip"])
    def solver(self, request: pytest.FixtureRequest) -> Solver:
        return create_solver(backend=request.param, seed=42)

    def test_sum_variables(self, solver: Solver) -> None:
        """Test summing variables."""
        binary_vars = [solver.add_binary_var(name=f"x{i}") for i in range(5)]

        # Sum all variables and constrain to equal 3
        total = solver_sum(binary_vars)
        solver.add_constr(total == 3)
        solver.set_objective(total, SolverSense.MAXIMIZE)

        status = solver.optimize()
        assert status == SolverStatus.OPTIMAL

        # Exactly 3 variables should be 1
        selected_count = sum(1 for v in binary_vars if solver.get_value(v) > 0.5)
        assert selected_count == 3

    def test_weighted_sum(self, solver: Solver) -> None:
        """Test weighted sum of variables."""
        x = solver.add_continuous_var(lb=0.0, ub=10.0, name="x")
        y = solver.add_continuous_var(lb=0.0, ub=10.0, name="y")

        weights = {x: 2.0, y: 3.0}
        weighted_expr = solver_sum(w * v for v, w in weights.items())

        solver.add_constr(x + y <= 5)
        solver.set_objective(weighted_expr, SolverSense.MAXIMIZE)

        status = solver.optimize()
        assert status == SolverStatus.OPTIMAL

        # Should maximize 3y, so y=5, x=0, giving value 15
        obj_value = solver.get_objective_value()
        assert abs(obj_value - 15.0) < 1e-5


class TestSolverEquivalence:
    """Test that both solvers produce equivalent results."""

    def test_same_optimal_value(self) -> None:
        """Test that both solvers find the same optimal value."""
        # Solve: maximize x + y
        # subject to: x + 2y <= 14, 3x + y <= 14, 0 <= x,y <= 10
        # Optimal at intersection of x + 2y = 14 and 3x + y = 14
        # Solving: x = 2.8, y = 5.6, so x + y = 8.4
        for backend in ["highspy", "mip"]:
            solver = create_solver(backend=backend, seed=42)

            x = solver.add_continuous_var(lb=0.0, ub=10.0)
            y = solver.add_continuous_var(lb=0.0, ub=10.0)

            solver.add_constr(x + 2 * y <= 14)
            solver.add_constr(3 * x + y <= 14)
            solver.set_objective(x + y, SolverSense.MAXIMIZE)

            status = solver.optimize()
            assert status == SolverStatus.OPTIMAL

            obj_value = solver.get_objective_value()
            # Both solvers should find the same optimal value
            assert abs(obj_value - 8.4) < 1e-4

    def test_committee_selection_like_problem(self) -> None:
        """Test a problem similar to committee selection."""
        for backend in ["highspy", "mip"]:
            solver = create_solver(backend=backend, seed=42)

            # 5 people, select 2
            people = ["alice", "bob", "charlie", "diana", "eve"]
            person_vars = {p: solver.add_binary_var(name=p) for p in people}

            # Must select exactly 2
            solver.add_constr(solver_sum(person_vars.values()) == 2)

            # Gender constraint: at least 1 female (alice, diana, eve)
            females = ["alice", "diana", "eve"]
            solver.add_constr(solver_sum(person_vars[p] for p in females) >= 1)

            # Maximize selection (any feasible solution is fine)
            solver.set_objective(solver_sum(person_vars.values()), SolverSense.MAXIMIZE)

            status = solver.optimize()
            assert status == SolverStatus.OPTIMAL

            # Check exactly 2 selected
            selected = [p for p in people if solver.get_value(person_vars[p]) > 0.5]
            assert len(selected) == 2

            # Check at least 1 female
            selected_females = [p for p in selected if p in females]
            assert len(selected_females) >= 1
