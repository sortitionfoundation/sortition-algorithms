# ABOUTME: Common utilities for committee generation algorithms.
# ABOUTME: Provides setup functions, constraint building, and multiplicative weights.

import copy
import logging
from collections.abc import Collection, Iterable
from typing import Any

from sortition_algorithms import errors
from sortition_algorithms.committee_generation.solver import (
    Solver,
    SolverSense,
    SolverStatus,
    create_solver,
    solver_sum,
)
from sortition_algorithms.features import FeatureCollection, feature_value_pairs, iterate_feature_collection
from sortition_algorithms.people import People
from sortition_algorithms.utils import RunReport, logger, random_provider

# Tolerance for numerical comparisons
EPS = 0.0005
EPS2 = 0.00000001


def setup_committee_generation(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    check_same_address_columns: list[str],
    solver_backend: str = "highspy",
) -> tuple[Solver, dict[str, Any]]:
    """Set up the integer linear program for committee generation.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        check_same_address_columns: columns to check for same address, or empty list if
                                    not checking addresses.
        solver_backend: solver backend to use ("highspy" or "mip")

    Returns:
        tuple of (Solver, dict mapping person_id to binary variables)

    Raises:
        InfeasibleQuotasError: If quotas are infeasible, includes suggested relaxations
        SelectionError: If solver fails for other reasons
    """
    solver = create_solver(backend=solver_backend)

    # Binary variable for each person (selected/not selected)
    agent_vars = {person_id: solver.add_binary_var() for person_id in people}

    # Must select exactly the desired number of people
    solver.add_constr(solver_sum(agent_vars.values()) == number_people_wanted)

    # Respect min/max quotas for each feature value
    for feature_name, fvalue_name, fv_minmax in iterate_feature_collection(features):
        # Count people with this feature-value who are selected
        number_feature_value_agents = solver_sum(
            agent_vars[person_id]
            for person_id, person_data in people.items()
            if person_data[feature_name].lower() == fvalue_name.lower()
        )

        # Add min/max constraints
        solver.add_constr(number_feature_value_agents >= fv_minmax.min)
        solver.add_constr(number_feature_value_agents <= fv_minmax.max)

    # Household constraints: at most 1 person per household
    if check_same_address_columns:
        for housemates in people.households(check_same_address_columns).values():
            if len(housemates) > 1:
                solver.add_constr(solver_sum(agent_vars[member_id] for member_id in housemates) <= 1)

    # Test feasibility by optimizing once (objective doesn't matter for feasibility check)
    solver.set_objective(solver_sum(agent_vars.values()), SolverSense.MAXIMIZE)
    status = solver.optimize()
    if status == SolverStatus.INFEASIBLE:
        relaxed_features, output_lines = _relax_infeasible_quotas(
            features, people, number_people_wanted, check_same_address_columns, solver_backend=solver_backend
        )
        raise errors.InfeasibleQuotasError(relaxed_features, output_lines)
    if status != SolverStatus.OPTIMAL:
        msg = f"No feasible committees found, solver returns status {status}."
        raise errors.SelectionError(msg)

    return solver, agent_vars


def _relax_infeasible_quotas(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    check_same_address_columns: list[str],
    ensure_inclusion: Collection[Iterable[str]] = ((),),
    solver_backend: str = "highspy",
) -> tuple[FeatureCollection, list[str]]:
    """Assuming that the quotas are not satisfiable, suggest a minimal relaxation that would be.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        check_same_address_columns: columns to check for same address, or empty list if
                                    not checking addresses.
        ensure_inclusion: allows to specify that some panels should contain specific sets of agents. for example,
            passing `(("a",), ("b", "c"))` means that the quotas should be relaxed such that some valid panel contains
            agent "a" and some valid panel contains both agents "b" and "c". the default of `((),)` just requires
            a panel to exist, without further restrictions.
        solver_backend: solver backend to use ("highspy" or "mip")

    Returns:
        tuple of (relaxed FeatureCollection, list of output messages)

    Raises:
        InfeasibleQuotasCantRelaxError: If quotas cannot be relaxed within min_flex/max_flex bounds
        SelectionError: If solver fails for other reasons
    """
    assert len(ensure_inclusion) > 0  # otherwise, the existence of a panel is not required

    solver = create_solver(backend=solver_backend)

    # Create relaxation variables and bounds constraints
    min_vars, max_vars = _create_relaxation_variables_and_bounds(solver, features)

    # For each inclusion set, create constraints to ensure a valid committee exists
    for inclusion_set in ensure_inclusion:
        _add_committee_constraints_for_inclusion_set(
            solver,
            inclusion_set,
            people,
            number_people_wanted,
            features,
            min_vars,
            max_vars,
            check_same_address_columns,
        )

    # Objective: minimize weighted sum of relaxations
    objective_expr = solver_sum(
        [_reduction_weight(features, *fv) * min_vars[fv] for fv in feature_value_pairs(features)]
        + [max_vars[fv] for fv in feature_value_pairs(features)]
    )
    solver.set_objective(objective_expr, SolverSense.MINIMIZE)

    # Solve the model and handle errors
    _solve_relaxation_model_and_handle_errors(solver)

    # Extract results and generate messages
    return _extract_relaxed_features_and_messages(solver, features, min_vars, max_vars)


def _reduction_weight(features: FeatureCollection, feature_name: str, value_name: str) -> float:
    """Make the algorithm more reluctant to reduce lower quotas that are already low.
    If the lower quota was 1, reducing it one more (to 0) is 3 times more salient than
    increasing a quota by 1. This bonus tapers off quickly, reducing from 10 is only
    1.2 times as salient as an increase."""
    # Find the current min quota for this feature-value
    # we assume we are only getting value feature/value names as this is called
    # while iterating through a loop
    old_quota = features[feature_name][value_name].min
    if old_quota == 0:
        return 0  # cannot be relaxed anyway
    return 1 + 2 / old_quota


def _create_relaxation_variables_and_bounds(
    solver: Solver,
    features: FeatureCollection,
) -> tuple[dict[tuple[str, str], Any], dict[tuple[str, str], Any]]:
    """Create relaxation variables and add bounds constraints for quota relaxation.

    Args:
        solver: Solver to add variables and constraints to
        features: FeatureCollection with min/max quotas and flex bounds

    Returns:
        tuple of (min_vars, max_vars) - dictionaries mapping feature-value pairs to relaxation variables
    """
    # Create relaxation variables for each feature-value pair
    min_vars = {fv: solver.add_integer_var(lb=0.0) for fv in feature_value_pairs(features)}
    max_vars = {fv: solver.add_integer_var(lb=0.0) for fv in feature_value_pairs(features)}

    # Add constraints ensuring relaxations stay within min_flex/max_flex bounds
    for feature_name, fvalue_name, fv_minmax in iterate_feature_collection(features):
        fv = (feature_name, fvalue_name)

        # Relaxed min cannot go below min_flex
        solver.add_constr(fv_minmax.min - min_vars[fv] >= fv_minmax.min_flex)

        # Relaxed max cannot go above max_flex
        solver.add_constr(fv_minmax.max + max_vars[fv] <= fv_minmax.max_flex)

    return min_vars, max_vars


def _add_committee_constraints_for_inclusion_set(
    solver: Solver,
    inclusion_set: Iterable[str],
    people: People,
    number_people_wanted: int,
    features: FeatureCollection,
    min_vars: dict[tuple[str, str], Any],
    max_vars: dict[tuple[str, str], Any],
    check_same_address_columns: list[str],
) -> None:
    """Add constraints to ensure a valid committee exists that includes the specified agents.

    Args:
        solver: Solver to add constraints to
        inclusion_set: agents that must be included in this committee
        people: People object with pool members
        number_people_wanted: desired size of the panel
        features: FeatureCollection with quotas
        min_vars: relaxation variables for minimum quotas
        max_vars: relaxation variables for maximum quotas
        check_same_address_columns: columns to check for same address, or empty list if
                                    not checking addresses.
    """
    # Binary variables for each person (selected/not selected)
    agent_vars = {person_id: solver.add_binary_var() for person_id in people}

    # Force inclusion of specified agents
    for agent in inclusion_set:
        solver.add_constr(agent_vars[agent] == 1)

    # Must select exactly the desired number of people
    solver.add_constr(solver_sum(agent_vars.values()) == number_people_wanted)

    # Respect relaxed quotas for each feature-value
    for feature_name, fvalue_name, fv_minmax in iterate_feature_collection(features):
        fv = (feature_name, fvalue_name)

        # Count people with this feature-value who are selected
        number_feature_value_agents = solver_sum(
            agent_vars[person_id]
            for person_id, person_data in people.items()
            if person_data[feature_name].lower() == fvalue_name.lower()
        )

        # Apply relaxed min/max constraints
        solver.add_constr(number_feature_value_agents >= fv_minmax.min - min_vars[fv])
        solver.add_constr(number_feature_value_agents <= fv_minmax.max + max_vars[fv])

    # Household constraints: at most 1 person per household
    if check_same_address_columns:
        for housemates in people.households(check_same_address_columns).values():
            if len(housemates) > 1:
                solver.add_constr(solver_sum(agent_vars[member_id] for member_id in housemates) <= 1)


def _solve_relaxation_model_and_handle_errors(solver: Solver) -> None:
    """Solve the relaxation model and handle any optimization errors.

    Args:
        solver: Solver to solve

    Raises:
        InfeasibleQuotasCantRelaxError: If quotas cannot be relaxed within flex bounds
        SelectionError: If solver fails for other reasons
    """
    status = solver.optimize()
    if status == SolverStatus.INFEASIBLE:
        msg = (
            "No feasible committees found, even with relaxing the quotas. Most "
            "likely, quotas would have to be relaxed beyond what the 'min_flex' and "
            "'max_flex' columns allow."
        )
        raise errors.InfeasibleQuotasCantRelaxError(msg)
    if status != SolverStatus.OPTIMAL:
        msg = f"No feasible committees found, solver returns status {status}. Either the pool is very bad or something is wrong with the solver."
        raise errors.SelectionError(msg)


def _extract_relaxed_features_and_messages(
    solver: Solver,
    features: FeatureCollection,
    min_vars: dict[tuple[str, str], Any],
    max_vars: dict[tuple[str, str], Any],
) -> tuple[FeatureCollection, list[str]]:
    """
    Extract relaxed quotas from solved model and generate recommendation messages.

    Args:
        solver: Solver with solved model
        features: Original FeatureCollection with quotas
        min_vars: solved relaxation variables for minimum quotas
        max_vars: solved relaxation variables for maximum quotas

    Returns:
        tuple of (relaxed FeatureCollection, list of recommendation messages)
    """
    # Create a copy of the features with relaxed quotas
    relaxed_features = copy.deepcopy(features)
    output_lines = []

    for feature_name, fvalue_name, fv_minmax in iterate_feature_collection(relaxed_features):
        fv = (feature_name, fvalue_name)
        min_relaxation = round(solver.get_value(min_vars[fv]))
        max_relaxation = round(solver.get_value(max_vars[fv]))

        original_min = fv_minmax.min
        original_max = fv_minmax.max

        new_min = original_min - min_relaxation
        new_max = original_max + max_relaxation

        # Update the values
        fv_minmax.min = new_min
        fv_minmax.max = new_max

        # Generate output messages
        if new_min < original_min:
            output_lines.append(f"Recommend lowering lower quota of '{feature_name}:{fvalue_name}' to {new_min}.")
        if new_max > original_max:
            output_lines.append(f"Recommend raising upper quota of '{feature_name}:{fvalue_name}' to {new_max}.")

    return relaxed_features, output_lines


def ilp_results_to_committee(solver: Solver, variables: dict[str, Any]) -> frozenset[str]:
    """Extract the selected committee from ILP solver variables.

    Args:
        solver: Solver with solved model
        variables: dict mapping person_id to binary variables

    Returns:
        frozenset of person_ids who are selected (have variable value > 0.5)

    Raises:
        ValueError: If variables don't have values (solver failed)
    """
    try:
        committee = frozenset(person_id for person_id in variables if solver.get_value(variables[person_id]) > 0.5)
    # unfortunately, solvers sometimes throw generic Exceptions rather than a subclass
    except Exception as error:
        msg = f"It seems like some variables do not have a value. Original exception: {error}."
        raise ValueError(msg, "variables_without_value", {"error": str(error)}) from error

    return committee


def _update_multiplicative_weights_after_committee_found(
    weights: dict[str, float],
    new_committee: frozenset[str],
    agent_vars: dict[str, Any],
    found_duplicate: bool,
) -> None:
    """Update multiplicative weights after finding a committee.

    Args:
        weights: current weights for each agent (modified in-place)
        new_committee: the committee that was just found
        agent_vars: dict mapping agent_id to binary variables
        found_duplicate: True if this committee was already found before
    """
    if not found_duplicate:
        # Decrease the weight of each agent in the new committee by a constant factor
        # As a result, future rounds will strongly prioritize including agents that appear in few committees
        for agent_id in new_committee:
            weights[agent_id] *= 0.8
    else:
        # If committee is already known, make all weights a bit more equal to mix things up
        for agent_id in agent_vars:
            weights[agent_id] = 0.9 * weights[agent_id] + 0.1

    # Rescale the weights to prevent floating point problems
    coefficient_sum = sum(weights.values())
    for agent_id in agent_vars:
        weights[agent_id] *= len(agent_vars) / coefficient_sum


def _run_multiplicative_weights_phase(
    solver: Solver,
    agent_vars: dict[str, Any],
    multiplicative_weights_rounds: int,
) -> tuple[set[frozenset[str]], set[str]]:
    """Run the multiplicative weights algorithm to find an initial diverse set of committees.

    Args:
        solver: Solver for finding committees
        agent_vars: dict mapping agent_id to binary variables
        multiplicative_weights_rounds: number of rounds to run

    Returns:
        tuple of (committees, covered_agents) - sets of committees found and agents covered
    """
    committees: set[frozenset[str]] = set()
    covered_agents: set[str] = set()

    # Each agent has a weight between 0.99 and 1
    # Note that if all start with weight `1` then we can end up with some committees having wrong number of results
    weights = {agent_id: random_provider().uniform(0.99, 1.0) for agent_id in agent_vars}

    for i in range(multiplicative_weights_rounds):
        # Find a feasible committee such that the sum of weights of its members is maximal
        solver.set_objective(
            solver_sum(weights[agent_id] * agent_vars[agent_id] for agent_id in agent_vars), SolverSense.MAXIMIZE
        )
        solver.optimize()
        new_committee = ilp_results_to_committee(solver, agent_vars)

        # Check if this is a new committee
        is_new_committee = new_committee not in committees
        if is_new_committee:
            committees.add(new_committee)
            for agent_id in new_committee:
                covered_agents.add(agent_id)

        # Update weights based on whether we found a new committee
        _update_multiplicative_weights_after_committee_found(weights, new_committee, agent_vars, not is_new_committee)

        logger.debug(
            f"Multiplicative weights phase, round {i + 1}/{multiplicative_weights_rounds}. "
            f"Discovered {len(committees)} committees so far."
        )

    return committees, covered_agents


def _find_committees_for_uncovered_agents(
    solver: Solver,
    agent_vars: dict[str, Any],
    covered_agents: set[str],
) -> tuple[set[frozenset[str]], set[str], RunReport]:
    """Find committees that include any agents not yet covered by existing committees.

    Args:
        solver: Solver for finding committees
        agent_vars: dict mapping agent_id to binary variables
        covered_agents: agents already covered by existing committees (modified in-place)

    Returns:
        tuple of (new_committees, updated_covered_agents, output_lines)
    """
    new_committees: set[frozenset[str]] = set()
    report = RunReport()

    # Try to find a committee including each uncovered agent
    for agent_id, agent_var in agent_vars.items():
        if agent_id not in covered_agents:
            solver.set_objective(agent_var, SolverSense.MAXIMIZE)  # only care about this specific agent being included
            solver.optimize()
            new_committee = ilp_results_to_committee(solver, agent_vars)

            if agent_id in new_committee:
                new_committees.add(new_committee)
                for covered_agent_id in new_committee:
                    covered_agents.add(covered_agent_id)
            else:
                report.add_message_and_log("agent_not_in_feasible_committee", log_level=logging.INFO, agent_id=agent_id)

    return new_committees, covered_agents, report


def generate_initial_committees(
    solver: Solver,
    agent_vars: dict[str, Any],
    multiplicative_weights_rounds: int,
) -> tuple[set[frozenset[str]], frozenset[str], RunReport]:
    """To speed up the main iteration of the maximin and Nash algorithms, start from a diverse set of feasible
    committees. In particular, each agent that can be included in any committee will be included in at least one of
    these committees.

    Args:
        solver: Solver for finding committees
        agent_vars: dict mapping agent_id to binary variables
        multiplicative_weights_rounds: number of rounds for the multiplicative weights phase

    Returns:
        tuple of (committees, covered_agents, output_lines)
        - committees: set of feasible committees discovered
        - covered_agents: frozenset of all agents included in some committee
        - report: run report
        - output_lines: list of debug messages
    """
    report = RunReport()

    # Phase 1: Use multiplicative weights algorithm to find diverse committees
    committees, covered_agents = _run_multiplicative_weights_phase(solver, agent_vars, multiplicative_weights_rounds)

    # Phase 2: Find committees for any agents not yet covered
    additional_committees, covered_agents, coverage_report = _find_committees_for_uncovered_agents(
        solver, agent_vars, covered_agents
    )
    committees.update(additional_committees)
    report.add_report(coverage_report)

    # Validation and final output
    assert len(committees) >= 1  # We assume quotas are feasible at this stage

    if len(covered_agents) == len(agent_vars):
        report.add_message_and_log("all_agents_in_feasible_committees", logging.INFO)

    return committees, frozenset(covered_agents), report
