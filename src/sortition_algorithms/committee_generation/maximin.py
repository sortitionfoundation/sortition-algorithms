import logging
from typing import Any

import mip

from sortition_algorithms.committee_generation.common import (
    EPS,
    generate_initial_committees,
    ilp_results_to_committee,
    setup_committee_generation,
)
from sortition_algorithms.features import FeatureCollection
from sortition_algorithms.people import People
from sortition_algorithms.utils import RunReport


def _find_maximin_primal(
    committees: list[frozenset[str]],
    covered_agents: frozenset[str],
) -> list[float]:
    """Find the optimal probabilities for committees that maximize the minimum selection probability.

    Args:
        committees: list of feasible committees
        covered_agents: frozenset of agents who can be selected

    Returns:
        list of probabilities for each committee (same order as input)
    """
    model = mip.Model(sense=mip.MAXIMIZE)
    model.verbose = 0

    committee_variables = [model.add_var(var_type=mip.CONTINUOUS, lb=0.0, ub=1.0) for _ in committees]
    model.add_constr(mip.xsum(committee_variables) == 1)

    agent_panel_variables: dict[str, list[Any]] = {agent_id: [] for agent_id in covered_agents}
    for committee, var in zip(committees, committee_variables, strict=False):
        for agent_id in committee:
            if agent_id in covered_agents:
                agent_panel_variables[agent_id].append(var)

    lower = model.add_var(var_type=mip.CONTINUOUS, lb=0.0, ub=1.0)

    for agent_variables in agent_panel_variables.values():
        model.add_constr(lower <= mip.xsum(agent_variables))
    model.objective = lower
    model.optimize()

    probabilities = [var.x for var in committee_variables]
    probabilities = [max(p, 0) for p in probabilities]
    sum_probabilities = sum(probabilities)
    return [p / sum_probabilities for p in probabilities]


def _setup_maximin_incremental_model(
    committees: set[frozenset[str]],
    covered_agents: frozenset[str],
) -> tuple[mip.model.Model, dict[str, mip.entities.Var], mip.entities.Var]:
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

    Returns:
        tuple of (incremental_model, incr_agent_vars, upper_bound_var)
    """
    incremental_model = mip.Model(sense=mip.MINIMIZE)
    incremental_model.verbose = 0

    upper_bound = incremental_model.add_var(
        var_type=mip.CONTINUOUS,
        lb=0.0,
        ub=mip.INF,
    )  # variable z
    # variables y_e
    incr_agent_vars = {
        agent_id: incremental_model.add_var(var_type=mip.CONTINUOUS, lb=0.0, ub=1.0) for agent_id in covered_agents
    }

    # Σ_e y_e = 1
    incremental_model.add_constr(mip.xsum(incr_agent_vars.values()) == 1)
    # minimize z
    incremental_model.objective = upper_bound

    for committee in committees:
        committee_sum = mip.xsum([incr_agent_vars[agent_id] for agent_id in committee])
        # Σ_{i ∈ B} y_{e(i)} ≤ z   ∀ B ∈ `committees`
        incremental_model.add_constr(committee_sum <= upper_bound)

    return incremental_model, incr_agent_vars, upper_bound


def _run_maximin_heuristic_for_additional_committees(
    new_committee_model: mip.model.Model,
    agent_vars: dict[str, mip.entities.Var],
    incremental_model: mip.model.Model,
    incr_agent_vars: dict[str, mip.entities.Var],
    upper_bound_var: mip.entities.Var,
    committees: set[frozenset[str]],
    covered_agents: frozenset[str],
    entitlement_weights: dict[str, float],
    upper: float,
    value: float,
) -> int:
    """Run heuristic to find additional committees without re-optimizing the incremental model.

    Because optimizing `incremental_model` takes a long time, we would like to get multiple committees out
    of a single run of `incremental_model`. Rather than reoptimizing for optimal y_e and z, we find some
    feasible values y_e and z by modifying the old solution.
    This heuristic only adds more committees, and does not influence correctness.

    Args:
        new_committee_model: MIP model for finding new committees
        agent_vars: agent variables in the committee model
        incremental_model: the incremental LP model
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

        new_committee_model.objective = mip.xsum(
            entitlement_weights[agent_id] * agent_vars[agent_id] for agent_id in covered_agents
        )
        new_committee_model.optimize()
        new_set = ilp_results_to_committee(agent_vars)
        value = sum(entitlement_weights[agent_id] for agent_id in new_set)
        if value <= upper + EPS or new_set in committees:
            break
        committees.add(new_set)
        incremental_model.add_constr(mip.xsum(incr_agent_vars[agent_id] for agent_id in new_set) <= upper_bound_var)
        counter += 1

    return counter


def _run_maximin_optimization_loop(
    new_committee_model: mip.model.Model,
    agent_vars: dict[str, mip.entities.Var],
    incremental_model: mip.model.Model,
    incr_agent_vars: dict[str, mip.entities.Var],
    upper_bound_var: mip.entities.Var,
    committees: set[frozenset[str]],
    covered_agents: frozenset[str],
    report: RunReport,
) -> tuple[list[frozenset[str]], list[float], RunReport]:
    """Run the main maximin optimization loop with column generation.

    Args:
        new_committee_model: MIP model for finding new committees
        agent_vars: agent variables in the committee model
        incremental_model: the incremental LP model
        incr_agent_vars: agent variables in incremental model
        upper_bound_var: upper bound variable in incremental model
        committees: set of committees (modified in-place)
        covered_agents: agents that can be included
        output_lines: list of output messages (modified in-place)

    Returns:
        tuple of (committees, probabilities, output_lines)
    """
    while True:
        status = incremental_model.optimize()
        assert status == mip.OptimizationStatus.OPTIMAL

        # currently optimal values for y_e
        entitlement_weights = {agent_id: incr_agent_vars[agent_id].x for agent_id in covered_agents}
        upper = upper_bound_var.x  # currently optimal value for z

        # For these fixed y_e, find the feasible committee B with maximal Σ_{i ∈ B} y_{e(i)}
        new_committee_model.objective = mip.xsum(
            entitlement_weights[agent_id] * agent_vars[agent_id] for agent_id in covered_agents
        )
        new_committee_model.optimize()
        new_set = ilp_results_to_committee(agent_vars)
        value = sum(entitlement_weights[agent_id] for agent_id in new_set)

        report.add_line_and_log(
            f"Maximin is at most {value:.2%}, can do {upper:.2%} with {len(committees)} "
            f"committees. Gap {value - upper:.2%}{'≤' if value - upper <= EPS else '>'}{EPS:%}.",
            log_level=logging.DEBUG,
        )
        if value <= upper + EPS:
            # No feasible committee B violates Σ_{i ∈ B} y_{e(i)} ≤ z (at least up to EPS, to prevent rounding errors)
            # Thus, we have enough committees
            committee_list = list(committees)
            probabilities = _find_maximin_primal(committee_list, covered_agents)
            return committee_list, probabilities, report

        # Some committee B violates Σ_{i ∈ B} y_{e(i)} ≤ z. We add B to `committees` and recurse
        assert new_set not in committees
        committees.add(new_set)
        incremental_model.add_constr(mip.xsum(incr_agent_vars[agent_id] for agent_id in new_set) <= upper_bound_var)

        # Run heuristic to find additional committees
        counter = _run_maximin_heuristic_for_additional_committees(
            new_committee_model,
            agent_vars,
            incremental_model,
            incr_agent_vars,
            upper_bound_var,
            committees,
            covered_agents,
            entitlement_weights,
            upper,
            value,
        )
        if counter > 0:
            report.add_line_and_log(f"Heuristic successfully generated {counter} additional committees.", logging.INFO)


def find_distribution_maximin(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    check_same_address_columns: list[str],
) -> tuple[list[frozenset[str]], list[float], RunReport]:
    """Find a distribution over feasible committees that maximizes the minimum probability of an agent being selected.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        check_same_address_columns: Address columns for household identification, or empty
                                    if no address checking to be done.

    Returns:
        tuple of (committees, probabilities, output_lines)
        - committees: list of feasible committees (frozenset of agent IDs)
        - probabilities: list of probabilities for each committee
        - output_lines: list of debug strings
    """
    report = RunReport()
    report.add_line_and_log("Using maximin algorithm.", log_level=logging.INFO)

    # Set up an ILP that can be used for discovering new feasible committees
    new_committee_model, agent_vars = setup_committee_generation(
        features, people, number_people_wanted, check_same_address_columns
    )

    # Find initial committees that cover every possible agent
    committees, covered_agents, init_report = generate_initial_committees(new_committee_model, agent_vars, people.count)
    report.add_report(init_report)

    # Set up the incremental LP model for column generation
    incremental_model, incr_agent_vars, upper_bound_var = _setup_maximin_incremental_model(committees, covered_agents)

    # Run the main optimization loop
    return _run_maximin_optimization_loop(
        new_committee_model,
        agent_vars,
        incremental_model,
        incr_agent_vars,
        upper_bound_var,
        committees,
        covered_agents,
        report,
    )
