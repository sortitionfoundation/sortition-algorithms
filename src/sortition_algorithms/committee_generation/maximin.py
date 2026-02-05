# ABOUTME: Maximin algorithm for committee generation.
# ABOUTME: Maximizes the minimum selection probability across all agents.

import logging
from typing import Any

from sortition_algorithms.committee_generation.common import (
    EPS,
    generate_initial_committees,
    ilp_results_to_committee,
    setup_committee_generation,
)
from sortition_algorithms.committee_generation.solver import (
    Solver,
    SolverSense,
    SolverStatus,
    create_solver,
    solver_sum,
)
from sortition_algorithms.features import FeatureCollection
from sortition_algorithms.people import People
from sortition_algorithms.utils import RunReport


def _find_maximin_primal(
    committees: list[frozenset[str]],
    covered_agents: frozenset[str],
    solver_backend: str = "highspy",
) -> list[float]:
    """Find the optimal probabilities for committees that maximize the minimum selection probability.

    Args:
        committees: list of feasible committees
        covered_agents: frozenset of agents who can be selected
        solver_backend: solver backend to use ("highspy" or "mip")

    Returns:
        list of probabilities for each committee (same order as input)
    """
    solver = create_solver(backend=solver_backend)

    committee_variables = [solver.add_continuous_var(lb=0.0, ub=1.0) for _ in committees]
    solver.add_constr(solver_sum(committee_variables) == 1)

    agent_panel_variables: dict[str, list[Any]] = {agent_id: [] for agent_id in covered_agents}
    for committee, var in zip(committees, committee_variables, strict=False):
        for agent_id in committee:
            if agent_id in covered_agents:
                agent_panel_variables[agent_id].append(var)

    lower = solver.add_continuous_var(lb=0.0, ub=1.0)

    for agent_variables in agent_panel_variables.values():
        solver.add_constr(lower <= solver_sum(agent_variables))
    solver.set_objective(lower, SolverSense.MAXIMIZE)
    solver.optimize()

    probabilities = [solver.get_value(var) for var in committee_variables]
    probabilities = [max(p, 0) for p in probabilities]
    sum_probabilities = sum(probabilities)
    return [p / sum_probabilities for p in probabilities]


def _setup_maximin_incremental_model(
    committees: set[frozenset[str]],
    covered_agents: frozenset[str],
    solver_backend: str = "highspy",
) -> tuple[Solver, dict[str, Any], Any]:
    """Set up the incremental LP model for maximin optimization.

    The incremental model is an LP with a variable y_e for each entitlement e and one more variable z.
    For an agent i, let e(i) denote her entitlement. Then, the LP is:

    minimize  z
    s.t.      Σ_{i ∈ B} y_{e(i)} ≤ z   ∀ feasible committees B (*)
              Σ_e y_e = 1
              y_e ≥ 0                  ∀ e

    At any point in time, constraint (*) is only enforced for the committees in `committees`.

    Args:
        committees: set of initial committees
        covered_agents: agents that can be included in some committee
        solver_backend: solver backend to use ("highspy" or "mip")

    Returns:
        tuple of (incremental_solver, incr_agent_vars, upper_bound_var)
    """
    incremental_solver = create_solver(backend=solver_backend)

    # variable z - upper bound, no upper limit
    upper_bound = incremental_solver.add_continuous_var(lb=0.0, ub=float("inf"))
    # variables y_e
    incr_agent_vars = {agent_id: incremental_solver.add_continuous_var(lb=0.0, ub=1.0) for agent_id in covered_agents}

    # Σ_e y_e = 1
    incremental_solver.add_constr(solver_sum(incr_agent_vars.values()) == 1)
    # minimize z (will be set when we optimize)

    for committee in committees:
        committee_sum = solver_sum([incr_agent_vars[agent_id] for agent_id in committee])
        # Σ_{i ∈ B} y_{e(i)} ≤ z   ∀ B ∈ `committees`
        incremental_solver.add_constr(committee_sum <= upper_bound)

    return incremental_solver, incr_agent_vars, upper_bound


def _run_maximin_heuristic_for_additional_committees(
    new_committee_solver: Solver,
    agent_vars: dict[str, Any],
    incremental_solver: Solver,
    incr_agent_vars: dict[str, Any],
    upper_bound_var: Any,
    committees: set[frozenset[str]],
    covered_agents: frozenset[str],
    entitlement_weights: dict[str, float],
    upper: float,
    value: float,
) -> int:
    """Run heuristic to find additional committees without re-optimizing the incremental model.

    Because optimizing `incremental_solver` takes a long time, we would like to get multiple committees out
    of a single run of `incremental_solver`. Rather than reoptimizing for optimal y_e and z, we find some
    feasible values y_e and z by modifying the old solution.
    This heuristic only adds more committees, and does not influence correctness.

    Args:
        new_committee_solver: Solver for finding new committees
        agent_vars: agent variables in the committee model
        incremental_solver: the incremental LP solver
        incr_agent_vars: agent variables in incremental model
        upper_bound_var: upper bound variable in incremental model
        committees: set of committees (modified in-place)
        covered_agents: agents that can be included
        entitlement_weights: current entitlement weights (modified in-place)
        upper: current upper bound value
        value: current objective value

    Returns:
        number of additional committees found
    """
    counter = 0
    new_set = None  # Initialize to avoid UnboundLocalError

    for _ in range(10):
        # scale down the y_{e(i)} for i ∈ `new_set` to make Σ_{i ∈ `new_set`} y_{e(i)} ≤ z true
        if new_set is not None:  # Only scale if we have a new_set from previous iteration
            for agent_id in new_set:
                entitlement_weights[agent_id] *= upper / value

        # This will change Σ_e y_e to be less than 1. We rescale the y_e and z
        sum_weights = sum(entitlement_weights.values())
        if sum_weights < EPS:
            break
        for agent_id in entitlement_weights:
            entitlement_weights[agent_id] /= sum_weights
        upper /= sum_weights

        new_committee_solver.set_objective(
            solver_sum(entitlement_weights[agent_id] * agent_vars[agent_id] for agent_id in covered_agents),
            SolverSense.MAXIMIZE,
        )
        new_committee_solver.optimize()
        new_set = ilp_results_to_committee(new_committee_solver, agent_vars)
        value = sum(entitlement_weights[agent_id] for agent_id in new_set)
        if value <= upper + EPS or new_set in committees:
            break
        committees.add(new_set)
        incremental_solver.add_constr(solver_sum(incr_agent_vars[agent_id] for agent_id in new_set) <= upper_bound_var)
        counter += 1

    return counter


def _add_report_update(report: RunReport, value: float, upper: float, committees: set[frozenset[str]]) -> None:
    """Complex formatting, so pull out of the main flow."""
    at_most = f"{value:.2%}"
    upper_str = f"{upper:.2%}"
    num_committees = len(committees)
    gap_str = f"{value - upper:.2%}{'≤' if value - upper <= EPS else '>'}{EPS:%}"
    report.add_message_and_log(
        "maximin_is_at_most",
        log_level=logging.DEBUG,
        at_most=at_most,
        upper_str=upper_str,
        num_committees=num_committees,
        gap_str=gap_str,
    )


def _run_maximin_optimization_loop(
    new_committee_solver: Solver,
    agent_vars: dict[str, Any],
    incremental_solver: Solver,
    incr_agent_vars: dict[str, Any],
    upper_bound_var: Any,
    committees: set[frozenset[str]],
    covered_agents: frozenset[str],
    report: RunReport,
    solver_backend: str = "highspy",
) -> tuple[list[frozenset[str]], list[float], RunReport]:
    """Run the main maximin optimization loop with column generation.

    Args:
        new_committee_solver: Solver for finding new committees
        agent_vars: agent variables in the committee model
        incremental_solver: the incremental LP solver
        incr_agent_vars: agent variables in incremental model
        upper_bound_var: upper bound variable in incremental model
        committees: set of committees (modified in-place)
        covered_agents: agents that can be included
        output_lines: list of output messages (modified in-place)

    Returns:
        tuple of (committees, probabilities, output_lines)
    """
    while True:
        incremental_solver.set_objective(upper_bound_var, SolverSense.MINIMIZE)
        status = incremental_solver.optimize()
        assert status == SolverStatus.OPTIMAL

        # currently optimal values for y_e
        entitlement_weights = {
            agent_id: incremental_solver.get_value(incr_agent_vars[agent_id]) for agent_id in covered_agents
        }
        upper = incremental_solver.get_value(upper_bound_var)  # currently optimal value for z

        # For these fixed y_e, find the feasible committee B with maximal Σ_{i ∈ B} y_{e(i)}
        new_committee_solver.set_objective(
            solver_sum(entitlement_weights[agent_id] * agent_vars[agent_id] for agent_id in covered_agents),
            SolverSense.MAXIMIZE,
        )
        new_committee_solver.optimize()
        new_set = ilp_results_to_committee(new_committee_solver, agent_vars)
        value = sum(entitlement_weights[agent_id] for agent_id in new_set)

        _add_report_update(report, value, upper, committees)
        if value <= upper + EPS:
            # No feasible committee B violates Σ_{i ∈ B} y_{e(i)} ≤ z (at least up to EPS, to prevent rounding errors)
            # Thus, we have enough committees
            committee_list = list(committees)
            probabilities = _find_maximin_primal(committee_list, covered_agents, solver_backend)
            return committee_list, probabilities, report

        # Some committee B violates Σ_{i ∈ B} y_{e(i)} ≤ z. We add B to `committees` and recurse
        assert new_set not in committees
        committees.add(new_set)
        incremental_solver.add_constr(solver_sum(incr_agent_vars[agent_id] for agent_id in new_set) <= upper_bound_var)

        # Run heuristic to find additional committees
        counter = _run_maximin_heuristic_for_additional_committees(
            new_committee_solver,
            agent_vars,
            incremental_solver,
            incr_agent_vars,
            upper_bound_var,
            committees,
            covered_agents,
            entitlement_weights,
            upper,
            value,
        )
        if counter > 0:
            report.add_message_and_log("heuristic_generated_committees", logging.INFO, count=counter)


def find_distribution_maximin(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    check_same_address_columns: list[str],
    solver_backend: str = "highspy",
) -> tuple[list[frozenset[str]], list[float], RunReport]:
    """Find a distribution over feasible committees that maximizes the minimum probability of an agent being selected.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        check_same_address_columns: Address columns for household identification, or empty
                                    if no address checking to be done.
        solver_backend: solver backend to use ("highspy" or "mip")

    Returns:
        tuple of (committees, probabilities, output_lines)
        - committees: list of feasible committees (frozenset of agent IDs)
        - probabilities: list of probabilities for each committee
        - output_lines: list of debug strings
    """
    report = RunReport()
    report.add_message_and_log("using_maximin_algorithm", logging.INFO)

    # Set up an ILP that can be used for discovering new feasible committees
    new_committee_solver, agent_vars = setup_committee_generation(
        features, people, number_people_wanted, check_same_address_columns, solver_backend
    )

    # Find initial committees that cover every possible agent
    committees, covered_agents, init_report = generate_initial_committees(
        new_committee_solver, agent_vars, people.count
    )
    report.add_report(init_report)

    # Set up the incremental LP model for column generation
    incremental_solver, incr_agent_vars, upper_bound_var = _setup_maximin_incremental_model(
        committees, covered_agents, solver_backend
    )

    # Run the main optimization loop
    return _run_maximin_optimization_loop(
        new_committee_solver,
        agent_vars,
        incremental_solver,
        incr_agent_vars,
        upper_bound_var,
        committees,
        covered_agents,
        report,
        solver_backend,
    )
