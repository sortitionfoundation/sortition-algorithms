import logging
from math import log
from typing import Any

import cvxpy as cp
import mip
import numpy as np

from sortition_algorithms.committee_generation.common import (
    EPS2,
    generate_initial_committees,
    ilp_results_to_committee,
    setup_committee_generation,
)
from sortition_algorithms.features import FeatureCollection
from sortition_algorithms.people import People
from sortition_algorithms.utils import RunReport, logger

# Tolerance for numerical comparisons
EPS_NASH = 0.1


def _define_entitlements(
    covered_agents: frozenset[str],
) -> tuple[list[str], dict[str, int]]:
    """Define entitlements mapping for agents who can be selected.

    Creates a mapping from agent IDs to indices for use in matrix operations.
    This is used by the fairness algorithms to track selection probabilities.

    Args:
        covered_agents: frozenset of agent IDs who can potentially be selected

    Returns:
        tuple of (entitlements list, contributes_to_entitlement mapping)
        - entitlements: list of agent IDs in a fixed order
        - contributes_to_entitlement: dict mapping agent_id -> index in entitlements list
    """
    entitlements = list(covered_agents)
    contributes_to_entitlement = {agent_id: entitlements.index(agent_id) for agent_id in covered_agents}

    return entitlements, contributes_to_entitlement


def _committees_to_matrix(
    committees: list[frozenset[str]],
    entitlements: list[str],
    contributes_to_entitlement: dict[str, int],
) -> np.ndarray:
    """Convert list of committees to a binary matrix for optimization algorithms.

    Creates a binary matrix where entry (i,j) indicates whether agent entitlements[i]
    is included in committee j. This matrix is used by fairness algorithms to optimize
    selection probabilities.

    Args:
        committees: list of committees, each committee is a frozenset of agent IDs
        entitlements: list of agent IDs in a fixed order (from _define_entitlements)
        contributes_to_entitlement: dict mapping agent_id -> index in entitlements

    Returns:
        numpy array of shape (len(entitlements), len(committees)) where entry (i,j)
        is 1 if agent entitlements[i] is in committee j, 0 otherwise
    """
    columns = []
    for committee in committees:
        column = [0 for _ in entitlements]
        for agent_id in committee:
            column[contributes_to_entitlement[agent_id]] += 1
        columns.append(np.array(column))

    return np.column_stack(columns)


def _solve_nash_welfare_optimization(
    committees: list[frozenset[str]],
    entitlements: list[str],
    contributes_to_entitlement: dict[str, int],
    start_lambdas: list[float],
    number_people_wanted: int,
    report: RunReport,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Solve the Nash welfare optimization problem for current committees.

    Args:
        committees: list of current committees
        entitlements: list of agent entitlements
        contributes_to_entitlement: mapping from agent_id to entitlement index
        start_lambdas: starting probabilities for committees
        number_people_wanted: desired committee size
        output_lines: list of output messages (modified in-place)

    Returns:
        tuple of (lambdas variable, entitled_reciprocals, differentials)
    """
    lambdas = cp.Variable(len(committees))  # probability of outputting a specific committee
    lambdas.value = start_lambdas

    # A is a binary matrix, whose (i,j)th entry indicates whether agent `entitlements[i]`
    # is included in committee j
    matrix = _committees_to_matrix(committees, entitlements, contributes_to_entitlement)
    assert matrix.shape == (len(entitlements), len(committees))

    objective = cp.Maximize(cp.sum(cp.log(matrix @ lambdas)))
    constraints = [lambdas >= 0, cp.sum(lambdas) == 1]
    problem = cp.Problem(objective, constraints)

    # Try SCS solver first, fall back to ECOS if it fails
    try:
        nash_welfare = problem.solve(solver=cp.SCS, warm_start=True)
    except cp.SolverError:
        # At least the ECOS solver in cvxpy crashes sometimes (numerical instabilities?).
        # In this case, try another solver. But hope that SCS is more stable.
        report.add_line_and_log("Had to switch to ECOS solver.", log_level=logging.INFO)
        nash_welfare = problem.solve(solver=cp.ECOS, warm_start=True)

    scaled_welfare = nash_welfare - len(entitlements) * log(number_people_wanted / len(entitlements))
    report.add_line_and_log(f"Scaled Nash welfare is now: {scaled_welfare}.", log_level=logging.INFO)

    assert lambdas.value.shape == (len(committees),)
    entitled_utilities = matrix.dot(lambdas.value)
    assert entitled_utilities.shape == (len(entitlements),)
    assert (entitled_utilities > EPS2).all()
    entitled_reciprocals = 1 / entitled_utilities
    assert entitled_reciprocals.shape == (len(entitlements),)
    differentials = entitled_reciprocals.dot(matrix)
    assert differentials.shape == (len(committees),)

    return lambdas, entitled_reciprocals, differentials


def _find_best_new_committee_for_nash(
    new_committee_model: mip.model.Model,
    agent_vars: dict[str, mip.entities.Var],
    entitled_reciprocals: np.ndarray,
    contributes_to_entitlement: dict[str, int],
    covered_agents: frozenset[str],
) -> tuple[frozenset[str], float]:
    """Find the committee that maximizes the Nash welfare objective.

    Args:
        new_committee_model: MIP model for finding committees
        agent_vars: agent variables in the committee model
        entitled_reciprocals: reciprocals of current utilities
        contributes_to_entitlement: mapping from agent_id to entitlement index
        covered_agents: agents that can be included

    Returns:
        tuple of (new_committee, objective_value)
    """
    obj = [
        entitled_reciprocals[contributes_to_entitlement[agent_id]] * agent_vars[agent_id] for agent_id in covered_agents
    ]
    new_committee_model.objective = mip.xsum(obj)
    new_committee_model.optimize()

    new_set = ilp_results_to_committee(agent_vars)
    value = sum(entitled_reciprocals[contributes_to_entitlement[agent_id]] for agent_id in new_set)

    return new_set, value


def _run_nash_optimization_loop(
    new_committee_model: mip.model.Model,
    agent_vars: dict[str, mip.entities.Var],
    committees: list[frozenset[str]],
    entitlements: list[str],
    contributes_to_entitlement: dict[str, int],
    covered_agents: frozenset[str],
    number_people_wanted: int,
    report: RunReport,
) -> tuple[list[frozenset[str]], list[float], RunReport]:
    """Run the main Nash welfare optimization loop.

    Args:
        new_committee_model: MIP model for finding committees
        agent_vars: agent variables in the committee model
        committees: list of committees (modified in-place)
        entitlements: list of agent entitlements
        contributes_to_entitlement: mapping from agent_id to entitlement index
        covered_agents: agents that can be included
        number_people_wanted: desired committee size
        output_lines: list of output messages (modified in-place)

    Returns:
        tuple of (committees, probabilities, output_lines)
    """
    start_lambdas = [1 / len(committees) for _ in committees]

    while True:
        # Solve Nash welfare optimization for current committees
        lambdas, entitled_reciprocals, differentials = _solve_nash_welfare_optimization(
            committees,
            entitlements,
            contributes_to_entitlement,
            start_lambdas,
            number_people_wanted,
            report,
        )

        # Find the best new committee
        new_set, value = _find_best_new_committee_for_nash(
            new_committee_model,
            agent_vars,
            entitled_reciprocals,
            contributes_to_entitlement,
            covered_agents,
        )

        # Check convergence condition
        if value <= differentials.max() + EPS_NASH:
            probabilities = np.array(lambdas.value).clip(0, 1)
            probabilities_normalised = list(probabilities / sum(probabilities))
            return committees, probabilities_normalised, report

        # Add new committee and continue
        logger.debug(
            f"nash committee: value: {value}, max differentials: {differentials.max()}, value - max: {value - differentials.max()}"
        )
        assert new_set not in committees
        committees.append(new_set)
        # Add 0 probability for new committee
        start_lambdas = [*list(np.array(lambdas.value)), 0]


def find_distribution_nash(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    check_same_address_columns: list[str],
) -> tuple[list[frozenset[str]], list[float], RunReport]:
    """Find a distribution over feasible committees that maximizes the Nash welfare, i.e., the product of
    selection probabilities over all persons.

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

    The algorithm maximizes the product of selection probabilities Πᵢ pᵢ by equivalently maximizing
    log(Πᵢ pᵢ) = Σᵢ log(pᵢ). If some person i is not included in any feasible committee, their pᵢ is 0, and
    this sum is -∞. We maximize Σᵢ log(pᵢ) where i is restricted to range over persons that can possibly be included.
    """
    report = RunReport()
    report.add_line_and_log("Using Nash algorithm.", log_level=logging.INFO)

    # Set up an ILP used for discovering new feasible committees
    new_committee_model, agent_vars = setup_committee_generation(
        features, people, number_people_wanted, check_same_address_columns
    )

    # Find initial committees that include every possible agent
    committee_set, covered_agents, initial_report = generate_initial_committees(
        new_committee_model, agent_vars, 2 * people.count
    )
    committees = list(committee_set)
    report.add_report(initial_report)

    # Map the covered agents to indices in a list for easier matrix representation
    entitlements, contributes_to_entitlement = _define_entitlements(covered_agents)

    # Run the main Nash welfare optimization loop
    return _run_nash_optimization_loop(
        new_committee_model,
        agent_vars,
        committees,
        entitlements,
        contributes_to_entitlement,
        covered_agents,
        number_people_wanted,
        report,
    )
