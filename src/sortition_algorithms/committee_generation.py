import copy
from collections.abc import Collection, Iterable
from math import log
from typing import Any

import cvxpy as cp
import mip
import numpy as np

try:
    import gurobipy as grb

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    grb = None

from sortition_algorithms import errors
from sortition_algorithms.features import FeatureCollection
from sortition_algorithms.people import People
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import secrets_uniform

# Tolerance for numerical comparisons
EPS = 0.0005
EPS2 = 0.00000001
EPS_NASH = 0.1


def _print(message: str) -> str:
    """Print and return a message for output collection."""
    # TODO: should we replace this with logging or similar?
    print(message)
    return message


def _reduction_weight(features: FeatureCollection, feature_name: str, value_name: str) -> float:
    """Make the algorithm more reluctant to reduce lower quotas that are already low.
    If the lower quota was 1, reducing it one more (to 0) is 3 times more salient than
    increasing a quota by 1. This bonus tapers off quickly, reducing from 10 is only
    1.2 times as salient as an increase."""
    # Find the current min quota for this feature-value
    for fname, vname, vcounts in features.feature_values_counts():
        if fname == feature_name and vname == value_name:
            old_quota = vcounts.min
            if old_quota == 0:
                return 0  # cannot be relaxed anyway
            return 1 + 2 / old_quota
    return 1  # fallback


def _create_relaxation_variables_and_bounds(
    model: mip.model.Model,
    features: FeatureCollection,
) -> tuple[dict[tuple[str, str], mip.entities.Var], dict[tuple[str, str], mip.entities.Var]]:
    """Create relaxation variables and add bounds constraints for quota relaxation.

    Args:
        model: MIP model to add variables and constraints to
        features: FeatureCollection with min/max quotas and flex bounds

    Returns:
        tuple of (min_vars, max_vars) - dictionaries mapping feature-value pairs to relaxation variables
    """
    # Get all feature-value pairs
    feature_values = [(feature_name, value_name) for feature_name, value_name, _ in features.feature_values_counts()]

    # Create relaxation variables for each feature-value pair
    min_vars = {fv: model.add_var(var_type=mip.INTEGER, lb=0.0) for fv in feature_values}
    max_vars = {fv: model.add_var(var_type=mip.INTEGER, lb=0.0) for fv in feature_values}

    # Add constraints ensuring relaxations stay within min_flex/max_flex bounds
    for feature_name, value_name, value_counts in features.feature_values_counts():
        fv = (feature_name, value_name)

        # Relaxed min cannot go below min_flex
        model.add_constr(value_counts.min - min_vars[fv] >= value_counts.min_flex)

        # Relaxed max cannot go above max_flex
        model.add_constr(value_counts.max + max_vars[fv] <= value_counts.max_flex)

    return min_vars, max_vars


def _add_committee_constraints_for_inclusion_set(
    model: mip.model.Model,
    inclusion_set: Iterable[str],
    people: People,
    number_people_wanted: int,
    features: FeatureCollection,
    min_vars: dict[tuple[str, str], mip.entities.Var],
    max_vars: dict[tuple[str, str], mip.entities.Var],
    settings: Settings,
) -> None:
    """Add constraints to ensure a valid committee exists that includes the specified agents.

    Args:
        model: MIP model to add constraints to
        inclusion_set: agents that must be included in this committee
        people: People object with pool members
        number_people_wanted: desired size of the panel
        features: FeatureCollection with quotas
        min_vars: relaxation variables for minimum quotas
        max_vars: relaxation variables for maximum quotas
        settings: Settings object containing household checking configuration
    """
    # Binary variables for each person (selected/not selected)
    agent_vars = {person_id: model.add_var(var_type=mip.BINARY) for person_id in people}

    # Force inclusion of specified agents
    for agent in inclusion_set:
        model.add_constr(agent_vars[agent] == 1)

    # Must select exactly the desired number of people
    model.add_constr(mip.xsum(agent_vars.values()) == number_people_wanted)

    # Respect relaxed quotas for each feature-value
    for feature_name, value_name, value_counts in features.feature_values_counts():
        fv = (feature_name, value_name)

        # Count people with this feature-value who are selected
        number_feature_value_agents = mip.xsum(
            agent_vars[person_id]
            for person_id, person_data in people.items()
            if person_data[feature_name] == value_name
        )

        # Apply relaxed min/max constraints
        model.add_constr(number_feature_value_agents >= value_counts.min - min_vars[fv])
        model.add_constr(number_feature_value_agents <= value_counts.max + max_vars[fv])

    # Household constraints: at most 1 person per household
    if settings.check_same_address:
        for housemates in people.households(settings.check_same_address_columns).values():
            if len(housemates) > 1:
                model.add_constr(mip.xsum(agent_vars[member_id] for member_id in housemates) <= 1)


def _solve_relaxation_model_and_handle_errors(model: mip.model.Model) -> None:
    """Solve the relaxation model and handle any optimization errors.

    Args:
        model: MIP model to solve

    Raises:
        InfeasibleQuotasCantRelaxError: If quotas cannot be relaxed within flex bounds
        SelectionError: If solver fails for other reasons
    """
    status = model.optimize()
    if status == mip.OptimizationStatus.INFEASIBLE:
        msg = (
            "No feasible committees found, even with relaxing the quotas. Most "
            "likely, quotas would have to be relaxed beyond what the 'min_flex' and "
            "'max_flex' columns allow."
        )
        raise errors.InfeasibleQuotasCantRelaxError(msg)
    if status != mip.OptimizationStatus.OPTIMAL:
        msg = (
            f"No feasible committees found, solver returns code {status} (see "
            f"https://docs.python-mip.com/en/latest/classes.html#optimizationstatus). Either the pool "
            f"is very bad or something is wrong with the solver."
        )
        raise errors.SelectionError(msg)


def _extract_relaxed_features_and_messages(
    features: FeatureCollection,
    min_vars: dict[tuple[str, str], mip.entities.Var],
    max_vars: dict[tuple[str, str], mip.entities.Var],
) -> tuple[FeatureCollection, list[str]]:
    """Extract relaxed quotas from solved model and generate recommendation messages.

    Args:
        features: Original FeatureCollection with quotas
        min_vars: solved relaxation variables for minimum quotas
        max_vars: solved relaxation variables for maximum quotas

    Returns:
        tuple of (relaxed FeatureCollection, list of recommendation messages)
    """
    # Create a copy of the features with relaxed quotas
    relaxed_features = copy.deepcopy(features)
    output_lines = []

    for (
        feature_name,
        value_name,
        value_counts,
    ) in relaxed_features.feature_values_counts():
        fv = (feature_name, value_name)
        min_relaxation = round(min_vars[fv].x)
        max_relaxation = round(max_vars[fv].x)

        original_min = value_counts.min
        original_max = value_counts.max

        new_min = original_min - min_relaxation
        new_max = original_max + max_relaxation

        # Update the values
        value_counts.min = new_min
        value_counts.max = new_max

        # Generate output messages
        if new_min < original_min:
            output_lines.append(f"Recommend lowering lower quota of {feature_name}:{value_name} to {new_min}.")
        if new_max > original_max:
            output_lines.append(f"Recommend raising upper quota of {feature_name}:{value_name} to {new_max}.")

    return relaxed_features, output_lines


def _relax_infeasible_quotas(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    settings: Settings,
    ensure_inclusion: Collection[Iterable[str]] = ((),),
) -> tuple[FeatureCollection, list[str]]:
    """Assuming that the quotas are not satisfiable, suggest a minimal relaxation that would be.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        settings: Settings object containing check_same_address and check_same_address_columns
        ensure_inclusion: allows to specify that some panels should contain specific sets of agents. for example,
            passing `(("a",), ("b", "c"))` means that the quotas should be relaxed such that some valid panel contains
            agent "a" and some valid panel contains both agents "b" and "c". the default of `((),)` just requires
            a panel to exist, without further restrictions.

    Returns:
        tuple of (relaxed FeatureCollection, list of output messages)

    Raises:
        InfeasibleQuotasCantRelaxError: If quotas cannot be relaxed within min_flex/max_flex bounds
        SelectionError: If solver fails for other reasons
    """
    model = mip.Model(sense=mip.MINIMIZE)
    model.verbose = 0  # TODO: get debug level from settings

    assert len(ensure_inclusion) > 0  # otherwise, the existence of a panel is not required

    # Create relaxation variables and bounds constraints
    min_vars, max_vars = _create_relaxation_variables_and_bounds(model, features)

    # Get feature-value pairs for objective function
    feature_values = [(feature_name, value_name) for feature_name, value_name, _ in features.feature_values_counts()]

    # For each inclusion set, create constraints to ensure a valid committee exists
    for inclusion_set in ensure_inclusion:
        _add_committee_constraints_for_inclusion_set(
            model, inclusion_set, people, number_people_wanted, features, min_vars, max_vars, settings
        )

    # Objective: minimize weighted sum of relaxations
    model.objective = mip.xsum(
        [_reduction_weight(features, *fv) * min_vars[fv] for fv in feature_values]
        + [max_vars[fv] for fv in feature_values]
    )

    # Solve the model and handle errors
    _solve_relaxation_model_and_handle_errors(model)

    # Extract results and generate messages
    return _extract_relaxed_features_and_messages(features, min_vars, max_vars)


def _setup_committee_generation(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    settings: Settings,
) -> tuple[mip.model.Model, dict[str, mip.entities.Var]]:
    """Set up the integer linear program for committee generation.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        settings: Settings object containing household checking configuration

    Returns:
        tuple of (MIP model, dict mapping person_id to binary variables)

    Raises:
        InfeasibleQuotasError: If quotas are infeasible, includes suggested relaxations
        SelectionError: If solver fails for other reasons
    """
    model = mip.Model(sense=mip.MAXIMIZE)
    model.verbose = 0  # TODO: get debug level from settings

    # Binary variable for each person (selected/not selected)
    agent_vars = {person_id: model.add_var(var_type=mip.BINARY) for person_id in people}

    # Must select exactly the desired number of people
    model.add_constr(mip.xsum(agent_vars.values()) == number_people_wanted)

    # Respect min/max quotas for each feature value
    for feature_name, value_name, value_counts in features.feature_values_counts():
        # Count people with this feature-value who are selected
        number_feature_value_agents = mip.xsum(
            agent_vars[person_id]
            for person_id, person_data in people.items()
            if person_data[feature_name] == value_name
        )

        # Add min/max constraints
        model.add_constr(number_feature_value_agents >= value_counts.min)
        model.add_constr(number_feature_value_agents <= value_counts.max)

    # Household constraints: at most 1 person per household
    if settings.check_same_address:
        for housemates in people.households(settings.check_same_address_columns).values():
            if len(housemates) > 1:
                model.add_constr(mip.xsum(agent_vars[member_id] for member_id in housemates) <= 1)

    # Test feasibility by optimizing once
    status = model.optimize()
    if status == mip.OptimizationStatus.INFEASIBLE:
        relaxed_features, output_lines = _relax_infeasible_quotas(features, people, number_people_wanted, settings)
        raise errors.InfeasibleQuotasError(relaxed_features, output_lines)
    if status != mip.OptimizationStatus.OPTIMAL:
        msg = (
            f"No feasible committees found, solver returns code {status} (see "
            "https://docs.python-mip.com/en/latest/classes.html#optimizationstatus)."
        )
        raise errors.SelectionError(msg)

    return model, agent_vars


def _ilp_results_to_committee(variables: dict[str, mip.entities.Var]) -> frozenset[str]:
    """Extract the selected committee from ILP solver variables.

    Args:
        variables: dict mapping person_id to binary MIP variables

    Returns:
        frozenset of person_ids who are selected (have variable value > 0.5)

    Raises:
        ValueError: If variables don't have values (solver failed)
    """
    try:
        committee = frozenset(person_id for person_id in variables if variables[person_id].x > 0.5)
    # unfortunately, MIP sometimes throws generic Exceptions rather than a subclass
    except Exception as error:
        msg = f"It seems like some variables do not have a value. Original exception: {error}."
        raise ValueError(msg) from error

    return committee


def _find_any_committee(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    settings: Settings,
) -> tuple[list[frozenset[str]], list[str]]:
    """Find any single feasible committee that satisfies the quotas.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        settings: Settings object containing configuration

    Returns:
        tuple of (list containing one committee as frozenset of person_ids, empty list of messages)

    Raises:
        InfeasibleQuotasError: If quotas are infeasible
        SelectionError: If solver fails for other reasons
    """
    model, agent_vars = _setup_committee_generation(features, people, number_people_wanted, settings)
    committee = _ilp_results_to_committee(agent_vars)
    return [committee], []


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


def _update_multiplicative_weights_after_committee_found(
    weights: dict[str, float],
    new_committee: frozenset[str],
    agent_vars: dict[str, mip.entities.Var],
    found_duplicate: bool,
) -> None:
    """Update multiplicative weights after finding a committee.

    Args:
        weights: current weights for each agent (modified in-place)
        new_committee: the committee that was just found
        agent_vars: dict mapping agent_id to binary MIP variables
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
    new_committee_model: mip.model.Model,
    agent_vars: dict[str, mip.entities.Var],
    multiplicative_weights_rounds: int,
) -> tuple[set[frozenset[str]], set[str]]:
    """Run the multiplicative weights algorithm to find an initial diverse set of committees.

    Args:
        new_committee_model: MIP model for finding committees
        agent_vars: dict mapping agent_id to binary MIP variables
        multiplicative_weights_rounds: number of rounds to run

    Returns:
        tuple of (committees, covered_agents) - sets of committees found and agents covered
    """
    committees: set[frozenset[str]] = set()
    covered_agents: set[str] = set()

    # Each agent has a weight between 0.99 and 1
    # Note that if all start with weight `1` then we can end up with some committees having wrong number of results
    weights = {agent_id: secrets_uniform(0.99, 1.0) for agent_id in agent_vars}

    for i in range(multiplicative_weights_rounds):
        # Find a feasible committee such that the sum of weights of its members is maximal
        new_committee_model.objective = mip.xsum(weights[agent_id] * agent_vars[agent_id] for agent_id in agent_vars)
        new_committee_model.optimize()
        new_committee = _ilp_results_to_committee(agent_vars)

        # Check if this is a new committee
        is_new_committee = new_committee not in committees
        if is_new_committee:
            committees.add(new_committee)
            for agent_id in new_committee:
                covered_agents.add(agent_id)

        # Update weights based on whether we found a new committee
        _update_multiplicative_weights_after_committee_found(weights, new_committee, agent_vars, not is_new_committee)

        print(
            f"Multiplicative weights phase, round {i + 1}/{multiplicative_weights_rounds}. "
            f"Discovered {len(committees)} committees so far."
        )

    return committees, covered_agents


def _find_committees_for_uncovered_agents(
    new_committee_model: mip.model.Model,
    agent_vars: dict[str, mip.entities.Var],
    covered_agents: set[str],
) -> tuple[set[frozenset[str]], set[str], list[str]]:
    """Find committees that include any agents not yet covered by existing committees.

    Args:
        new_committee_model: MIP model for finding committees
        agent_vars: dict mapping agent_id to binary MIP variables
        covered_agents: agents already covered by existing committees (modified in-place)

    Returns:
        tuple of (new_committees, updated_covered_agents, output_lines)
    """
    new_committees: set[frozenset[str]] = set()
    output_lines = []

    # Try to find a committee including each uncovered agent
    for agent_id, agent_var in agent_vars.items():
        if agent_id not in covered_agents:
            new_committee_model.objective = agent_var  # only care about this specific agent being included
            new_committee_model.optimize()
            new_committee = _ilp_results_to_committee(agent_vars)

            if agent_id in new_committee:
                new_committees.add(new_committee)
                for covered_agent_id in new_committee:
                    covered_agents.add(covered_agent_id)
            else:
                output_lines.append(_print(f"Agent {agent_id} not contained in any feasible committee."))

    return new_committees, covered_agents, output_lines


def _generate_initial_committees(
    new_committee_model: mip.model.Model,
    agent_vars: dict[str, mip.entities.Var],
    multiplicative_weights_rounds: int,
) -> tuple[set[frozenset[str]], frozenset[str], list[str]]:
    """To speed up the main iteration of the maximin and Nash algorithms, start from a diverse set of feasible
    committees. In particular, each agent that can be included in any committee will be included in at least one of
    these committees.

    Args:
        new_committee_model: MIP model for finding committees
        agent_vars: dict mapping agent_id to binary MIP variables
        multiplicative_weights_rounds: number of rounds for the multiplicative weights phase

    Returns:
        tuple of (committees, covered_agents, output_lines)
        - committees: set of feasible committees discovered
        - covered_agents: frozenset of all agents included in some committee
        - output_lines: list of debug messages
    """
    output_lines = []

    # Phase 1: Use multiplicative weights algorithm to find diverse committees
    committees, covered_agents = _run_multiplicative_weights_phase(
        new_committee_model, agent_vars, multiplicative_weights_rounds
    )

    # Phase 2: Find committees for any agents not yet covered
    additional_committees, covered_agents, coverage_output = _find_committees_for_uncovered_agents(
        new_committee_model, agent_vars, covered_agents
    )
    committees.update(additional_committees)
    output_lines.extend(coverage_output)

    # Validation and final output
    assert len(committees) >= 1  # We assume quotas are feasible at this stage

    if len(covered_agents) == len(agent_vars):
        output_lines.append(_print("All agents are contained in some feasible committee."))

    return committees, frozenset(covered_agents), output_lines


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


def find_distribution_maximin(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    settings: Settings,
) -> tuple[list[frozenset[str]], list[float], list[str]]:
    """Find a distribution over feasible committees that maximizes the minimum probability of an agent being selected.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        settings: Settings object containing configuration

    Returns:
        tuple of (committees, probabilities, output_lines)
        - committees: list of feasible committees (frozenset of agent IDs)
        - probabilities: list of probabilities for each committee
        - output_lines: list of debug strings
    """
    output_lines = [_print("Using maximin algorithm.")]

    # Set up an ILP that can be used for discovering new feasible committees maximizing some
    # sum of weights over the agents
    new_committee_model, agent_vars = _setup_committee_generation(features, people, number_people_wanted, settings)

    # Start by finding some initial committees, guaranteed to cover every agent that can be covered by some committee
    committees: set[frozenset[str]]  # set of feasible committees, add more over time
    covered_agents: frozenset[str]  # all agent ids for agents that can actually be included
    committees, covered_agents, new_output_lines = _generate_initial_committees(
        new_committee_model,
        agent_vars,
        people.count,
    )
    output_lines += new_output_lines

    # The incremental model is an LP with a variable y_e for each entitlement e and one more variable z.
    # For an agent i, let e(i) denote her entitlement. Then, the LP is:
    #
    # minimize  z
    # s.t.      Σ_{i ∈ B} y_{e(i)} ≤ z   ∀ feasible committees B (*)
    #           Σ_e y_e = 1
    #           y_e ≥ 0                  ∀ e
    #
    # At any point in time, constraint (*) is only enforced for the committees in `committees`. By linear-programming
    # duality, if the optimal solution with these reduced constraints satisfies all possible constraints, the committees
    # in `committees` are enough to find the maximin distribution among them.
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

    while True:
        status = incremental_model.optimize()
        assert status == mip.OptimizationStatus.OPTIMAL

        # currently optimal values for y_e
        entitlement_weights = {agent_id: incr_agent_vars[agent_id].x for agent_id in covered_agents}
        upper = upper_bound.x  # currently optimal value for z

        # For these fixed y_e, find the feasible committee B with maximal Σ_{i ∈ B} y_{e(i)}
        new_committee_model.objective = mip.xsum(
            entitlement_weights[agent_id] * agent_vars[agent_id] for agent_id in covered_agents
        )
        new_committee_model.optimize()
        new_set = _ilp_results_to_committee(agent_vars)
        value = sum(entitlement_weights[agent_id] for agent_id in new_set)

        output_lines.append(
            _print(
                f"Maximin is at most {value:.2%}, can do {upper:.2%} with {len(committees)} "
                f"committees. Gap {value - upper:.2%}{'≤' if value - upper <= EPS else '>'}{EPS:%}."
            )
        )
        if value <= upper + EPS:
            # No feasible committee B violates Σ_{i ∈ B} y_{e(i)} ≤ z (at least up to EPS, to prevent rounding errors)
            # Thus, we have enough committees
            committee_list = list(committees)
            probabilities = _find_maximin_primal(committee_list, covered_agents)
            return committee_list, probabilities, output_lines

        # Some committee B violates Σ_{i ∈ B} y_{e(i)} ≤ z. We add B to `committees` and recurse
        assert new_set not in committees
        committees.add(new_set)
        incremental_model.add_constr(mip.xsum(incr_agent_vars[agent_id] for agent_id in new_set) <= upper_bound)

        # Heuristic for better speed in practice:
        # Because optimizing `incremental_model` takes a long time, we would like to get multiple committees out
        # of a single run of `incremental_model`. Rather than reoptimizing for optimal y_e and z, we find some
        # feasible values y_e and z by modifying the old solution.
        # This heuristic only adds more committees, and does not influence correctness.
        counter = 0
        for _ in range(10):
            # scale down the y_{e(i)} for i ∈ `new_set` to make Σ_{i ∈ `new_set`} y_{e(i)} ≤ z true
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
            new_set = _ilp_results_to_committee(agent_vars)
            value = sum(entitlement_weights[agent_id] for agent_id in new_set)
            if value <= upper + EPS or new_set in committees:
                break
            committees.add(new_set)
            incremental_model.add_constr(mip.xsum(incr_agent_vars[agent_id] for agent_id in new_set) <= upper_bound)
            counter += 1
        if counter > 0:
            print(f"Heuristic successfully generated {counter} additional committees.")


def find_distribution_nash(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    settings: Settings,
) -> tuple[list[frozenset[str]], list[float], list[str]]:
    """Find a distribution over feasible committees that maximizes the Nash welfare, i.e., the product of
    selection probabilities over all persons.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        settings: Settings object containing configuration

    Returns:
        tuple of (committees, probabilities, output_lines)
        - committees: list of feasible committees (frozenset of agent IDs)
        - probabilities: list of probabilities for each committee
        - output_lines: list of debug strings

    The algorithm maximizes the product of selection probabilities Πᵢ pᵢ by equivalently maximizing
    log(Πᵢ pᵢ) = Σᵢ log(pᵢ). If some person i is not included in any feasible committee, their pᵢ is 0, and
    this sum is -∞. We maximize Σᵢ log(pᵢ) where i is restricted to range over persons that can possibly be included.
    """
    output_lines = [_print("Using Nash algorithm.")]

    # Set up an ILP used for discovering new feasible committees
    # We will use it many times, putting different weights on the inclusion of different agents to find many feasible
    # committees
    new_committee_model, agent_vars = _setup_committee_generation(features, people, number_people_wanted, settings)

    # Start by finding committees including every agent, and learn which agents cannot possibly be included
    committees: list[frozenset[str]]  # set of feasible committees, add more over time
    covered_agents: frozenset[str]  # all agent ids for agents that can actually be included
    committee_set, covered_agents, new_output_lines = _generate_initial_committees(
        new_committee_model,
        agent_vars,
        2 * people.count,
    )
    committees = list(committee_set)
    output_lines += new_output_lines

    # Map the covered agents to indices in a list for easier matrix representation
    entitlements, contributes_to_entitlement = _define_entitlements(covered_agents)

    # The algorithm proceeds iteratively. First, it finds probabilities for the committees already present in
    # `committees` that maximize the sum of logarithms. Then, reusing the old ILP, it finds the feasible committee
    # (possibly outside of `committees`) such that the partial derivative of the sum of logarithms with respect to the
    # probability of outputting this committee is maximal. If this partial derivative is less than the maximal partial
    # derivative of any committee already in `committees`, the Karush-Kuhn-Tucker conditions (which are sufficient in
    # this case) imply that the distribution is optimal even with all other committees receiving probability 0.
    start_lambdas = [1 / len(committees) for _ in committees]

    while True:
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
            output_lines.append(_print("Had to switch to ECOS solver."))
            nash_welfare = problem.solve(solver=cp.ECOS, warm_start=True)

        scaled_welfare = nash_welfare - len(entitlements) * log(number_people_wanted / len(entitlements))
        output_lines.append(_print(f"Scaled Nash welfare is now: {scaled_welfare}."))

        assert lambdas.value.shape == (len(committees),)  # type: ignore[union-attr]
        entitled_utilities = matrix.dot(lambdas.value)  # type: ignore[arg-type]
        assert entitled_utilities.shape == (len(entitlements),)
        assert (entitled_utilities > EPS2).all()
        entitled_reciprocals = 1 / entitled_utilities
        assert entitled_reciprocals.shape == (len(entitlements),)
        differentials = entitled_reciprocals.dot(matrix)
        assert differentials.shape == (len(committees),)

        obj = [
            entitled_reciprocals[contributes_to_entitlement[agent_id]] * agent_vars[agent_id]
            for agent_id in covered_agents
        ]
        new_committee_model.objective = mip.xsum(obj)
        new_committee_model.optimize()

        new_set = _ilp_results_to_committee(agent_vars)
        value = sum(entitled_reciprocals[contributes_to_entitlement[agent_id]] for agent_id in new_set)

        if value <= differentials.max() + EPS_NASH:
            probabilities = np.array(lambdas.value).clip(0, 1)
            probabilities_normalised = list(probabilities / sum(probabilities))
            return committees, probabilities_normalised, output_lines

        print(value, differentials.max(), value - differentials.max())
        assert new_set not in committees
        committees.append(new_set)
        start_lambdas = [
            *list(np.array(lambdas.value)),
            0,
        ]  # Add 0 probability for new committee


def _dual_leximin_stage(
    people: People,
    committees: set[frozenset[str]],
    fixed_probabilities: dict[str, float],
) -> tuple:
    """Implements the dual LP described in `find_distribution_leximin`, but where P only ranges over the panels
    in `committees` rather than over all feasible panels:
    minimize ŷ - Σ_{i in fixed_probabilities} fixed_probabilities[i] * yᵢ
    s.t.     Σ_{i ∈ P} yᵢ ≤ ŷ                              ∀ P
             Σ_{i not in fixed_probabilities} yᵢ = 1
             ŷ, yᵢ ≥ 0                                     ∀ i

    Args:
        people: People object with all agents
        committees: set of feasible committees
        fixed_probabilities: dict mapping agent_id to fixed probability

    Returns:
        tuple of (grb.Model, dict[str, grb.Var], grb.Var) - (model, agent_vars, cap_var)

    Raises:
        RuntimeError: If Gurobi is not available
    """
    if not GUROBI_AVAILABLE:
        msg = "Leximin algorithm requires Gurobi solver which is not available"
        raise RuntimeError(msg)

    assert len(committees) != 0

    model = grb.Model()
    agent_vars = {agent_id: model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.0) for agent_id in people}  # yᵢ
    cap_var = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.0)  # ŷ
    model.addConstr(
        grb.quicksum(agent_vars[agent_id] for agent_id in people if agent_id not in fixed_probabilities) == 1
    )
    for committee in committees:
        model.addConstr(grb.quicksum(agent_vars[agent_id] for agent_id in committee) <= cap_var)
    model.setObjective(
        cap_var
        - grb.quicksum(fixed_probabilities[agent_id] * agent_vars[agent_id] for agent_id in fixed_probabilities),
        grb.GRB.MINIMIZE,
    )

    # Change Gurobi configuration to encourage strictly complementary ("inner") solutions. These solutions will
    # typically allow to fix more probabilities per outer loop of the leximin algorithm.
    model.setParam("Method", 2)  # optimize via barrier only
    model.setParam("Crossover", 0)  # deactivate cross-over

    return model, agent_vars, cap_var


def find_distribution_leximin(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    settings: Settings,
) -> tuple[list[frozenset[str]], list[float], list[str]]:
    """Find a distribution over feasible committees that maximizes the minimum probability of an agent being selected
    (just like maximin), but breaks ties to maximize the second-lowest probability, breaks further ties to maximize the
    third-lowest probability and so forth.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        settings: Settings object containing configuration

    Returns:
        tuple of (committees, probabilities, output_lines)
        - committees: list of feasible committees (frozenset of agent IDs)
        - probabilities: list of probabilities for each committee
        - output_lines: list of debug strings

    Raises:
        RuntimeError: If Gurobi is not available
    """
    if not GUROBI_AVAILABLE:
        msg = "Leximin algorithm requires Gurobi solver which is not available"
        raise RuntimeError(msg)

    output_lines = [_print("Using leximin algorithm.")]
    grb.setParam("OutputFlag", 0)

    # Set up an ILP that can be used for discovering new feasible committees maximizing some
    # sum of weights over the agents
    new_committee_model, agent_vars = _setup_committee_generation(features, people, number_people_wanted, settings)

    # Start by finding some initial committees, guaranteed to cover every agent that can be covered by some committee
    committees: set[frozenset[str]]  # set of feasible committees, add more over time
    covered_agents: frozenset[str]  # all agent ids for agents that can actually be included
    committees, covered_agents, new_output_lines = _generate_initial_committees(
        new_committee_model,
        agent_vars,
        3 * people.count,
    )
    output_lines += new_output_lines

    # Over the course of the algorithm, the selection probabilities of more and more agents get fixed to a certain value
    fixed_probabilities: dict[str, float] = {}

    reduction_counter = 0

    # The outer loop maximizes the minimum of all unfixed probabilities while satisfying the fixed probabilities.
    # In each iteration, at least one more probability is fixed, but often more than one.
    while len(fixed_probabilities) < people.count:
        print(f"Fixed {len(fixed_probabilities)}/{people.count} probabilities.")

        dual_model, dual_agent_vars, dual_cap_var = _dual_leximin_stage(
            people,
            committees,
            fixed_probabilities,
        )
        # In the inner loop, there is a column generation for maximizing the minimum of all unfixed probabilities
        while True:
            """The primal LP being solved by column generation, with a variable x_P for each feasible panel P:

            maximize z
            s.t.     Σ_{P : i ∈ P} x_P ≥ z                         ∀ i not in fixed_probabilities
                     Σ_{P : i ∈ P} x_P ≥ fixed_probabilities[i]    ∀ i in fixed_probabilities
                     Σ_P x_P ≤ 1                                   (This should be thought of as equality, and wlog.
                                                                   optimal solutions have equality, but simplifies dual)
                     x_P ≥ 0                                       ∀ P

            We instead solve its dual linear program:
            minimize ŷ - Σ_{i in fixed_probabilities} fixed_probabilities[i] * yᵢ
            s.t.     Σ_{i ∈ P} yᵢ ≤ ŷ                              ∀ P
                     Σ_{i not in fixed_probabilities} yᵢ = 1
                     ŷ, yᵢ ≥ 0                                     ∀ i
            """
            dual_model.optimize()
            if dual_model.status != grb.GRB.OPTIMAL:
                # In theory, the LP is feasible in the first iterations, and we only add constraints (by fixing
                # probabilities) that preserve feasibility. Due to floating-point issues, however, it may happen that
                # Gurobi still cannot satisfy all the fixed probabilities in the primal (meaning that the dual will be
                # unbounded). In this case, we slightly relax the LP by slightly reducing all fixed probabilities.
                for agent_id in fixed_probabilities:
                    # Relax all fixed probabilities by a small constant
                    fixed_probabilities[agent_id] = max(0.0, fixed_probabilities[agent_id] - 0.0001)
                dual_model, dual_agent_vars, dual_cap_var = _dual_leximin_stage(
                    people,
                    committees,
                    fixed_probabilities,
                )
                print(dual_model.status, f"REDUCE PROBS for {reduction_counter}th time.")
                reduction_counter += 1
                continue

            # Find the panel P for which Σ_{i ∈ P} yᵢ is largest, i.e., for which Σ_{i ∈ P} yᵢ ≤ ŷ is tightest
            agent_weights = {agent_id: agent_var.x for agent_id, agent_var in dual_agent_vars.items()}
            new_committee_model.objective = mip.xsum(
                agent_weights[agent_id] * agent_vars[agent_id] for agent_id in people
            )
            new_committee_model.optimize()
            new_set = _ilp_results_to_committee(agent_vars)  # panel P
            value = new_committee_model.objective_value  # Σ_{i ∈ P} yᵢ

            upper = dual_cap_var.x  # ŷ
            dual_obj = dual_model.objVal  # ŷ - Σ_{i in fixed_probabilities} fixed_probabilities[i] * yᵢ

            output_lines.append(
                _print(
                    f"Maximin is at most {dual_obj - upper + value:.2%}, can do {dual_obj:.2%} with "
                    f"{len(committees)} committees. Gap {value - upper:.2%}."
                )
            )
            if value <= upper + EPS:
                # Within numeric tolerance, the panels in `committees` are enough to constrain the dual, i.e., they are
                # enough to support an optimal primal solution.
                for agent_id, agent_weight in agent_weights.items():
                    if agent_weight > EPS and agent_id not in fixed_probabilities:
                        # `agent_weight` is the dual variable yᵢ of the constraint "Σ_{P : i ∈ P} x_P ≥ z" for
                        # i = `agent_id` in the primal LP. If yᵢ is positive, this means that the constraint must be
                        # binding in all optimal solutions [1], and we can fix `agent_id`'s probability to the
                        # optimal value of the primal/dual LP.
                        # [1] Theorem 3.3 in: Renato Pelessoni. Some remarks on the use of the strict complementarity in
                        # checking coherence and extending coherent probabilities. 1998.
                        fixed_probabilities[agent_id] = max(0, dual_obj)
                break
            # Given that Σ_{i ∈ P} yᵢ > ŷ, the current solution to `dual_model` is not yet a solution to the dual.
            # Thus, add the constraint for panel P and recurse.
            assert new_set not in committees
            committees.add(new_set)
            dual_model.addConstr(grb.quicksum(dual_agent_vars[agent_id] for agent_id in new_set) <= dual_cap_var)

    # The previous algorithm computed the leximin selection probabilities of each agent and a set of panels such that
    # the selection probabilities can be obtained by randomizing over these panels. Here, such a randomization is found.
    primal = grb.Model()
    # Variables for the output probabilities of the different panels
    committee_vars = [primal.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.0) for _ in committees]
    # To avoid numerical problems, we formally minimize the largest downward deviation from the fixed probabilities.
    eps = primal.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.0)
    primal.addConstr(grb.quicksum(committee_vars) == 1)  # Probabilities add up to 1
    for agent_id, prob in fixed_probabilities.items():
        agent_probability = grb.quicksum(
            comm_var for committee, comm_var in zip(committees, committee_vars, strict=False) if agent_id in committee
        )
        primal.addConstr(agent_probability >= prob - eps)
    primal.setObjective(eps, grb.GRB.MINIMIZE)
    primal.optimize()

    # Bound variables between 0 and 1 and renormalize, because np.random.choice is sensitive to small deviations here
    probabilities = np.array([comm_var.x for comm_var in committee_vars]).clip(0, 1)
    probabilities_normalised = list(probabilities / sum(probabilities))

    return list(committees), probabilities_normalised, output_lines


def pipage_rounding(marginals: list[tuple[int, float]]) -> list[int]:
    """Pipage rounding algorithm for converting fractional solutions to integer solutions.

    Takes a list of (object, probability) pairs and randomly rounds them to a set of objects
    such that the expected number of times each object appears equals its probability.

    Args:
        marginals: list of (object, probability) pairs where probabilities sum to an integer

    Returns:
        list of objects that were selected
    """
    assert all(0.0 <= p <= 1.0 for _, p in marginals)

    outcomes: list[int] = []
    while True:
        if len(marginals) == 0:
            return outcomes
        if len(marginals) == 1:
            obj, prob = marginals[0]
            if secrets_uniform(0.0, 1.0) < prob:
                outcomes.append(obj)
            marginals = []
        else:
            obj0, prob0 = marginals[0]
            if prob0 > 1.0 - EPS2:
                outcomes.append(obj0)
                marginals = marginals[1:]
                continue
            if prob0 < EPS2:
                marginals = marginals[1:]
                continue

            obj1, prob1 = marginals[1]
            if prob1 > 1.0 - EPS2:
                outcomes.append(obj1)
                marginals = [marginals[0]] + marginals[2:]
                continue
            if prob1 < EPS2:
                marginals = [marginals[0]] + marginals[2:]
                continue

            inc0_dec1_amount = min(
                1.0 - prob0, prob1
            )  # maximal amount that prob0 can be increased and prob1 can be decreased
            dec0_inc1_amount = min(prob0, 1.0 - prob1)
            choice_probability = dec0_inc1_amount / (inc0_dec1_amount + dec0_inc1_amount)

            if secrets_uniform(0.0, 1.0) < choice_probability:  # increase prob0 and decrease prob1
                prob0 += inc0_dec1_amount
                prob1 -= inc0_dec1_amount
            else:
                prob0 -= dec0_inc1_amount
                prob1 += dec0_inc1_amount
            marginals = [(obj0, prob0), (obj1, prob1)] + marginals[2:]


def standardize_distribution(
    committees: list[frozenset[str]],
    probabilities: list[float],
) -> tuple[list[frozenset[str]], list[float]]:
    """Remove committees with zero probability and renormalize.

    Args:
        committees: list of committees
        probabilities: corresponding probabilities

    Returns:
        tuple of (filtered_committees, normalized_probabilities)
    """
    assert len(committees) == len(probabilities)
    new_committees = []
    new_probabilities = []
    for committee, prob in zip(committees, probabilities, strict=False):
        if prob >= EPS2:
            new_committees.append(committee)
            new_probabilities.append(prob)
    prob_sum = sum(new_probabilities)
    new_probabilities = [prob / prob_sum for prob in new_probabilities]
    return new_committees, new_probabilities


def lottery_rounding(
    committees: list[frozenset[str]],
    probabilities: list[float],
    number_selections: int,
) -> list[frozenset[str]]:
    """Convert probability distribution over committees to a discrete lottery.

    Args:
        committees: list of committees
        probabilities: corresponding probabilities (must sum to 1)
        number_selections: number of committees to return

    Returns:
        list of committees (may contain duplicates) of length number_selections
    """
    assert len(committees) == len(probabilities)
    assert number_selections >= 1

    num_copies: list[int] = []
    residuals: list[float] = []
    for _, prob in zip(committees, probabilities, strict=False):
        scaled_prob = prob * number_selections
        num_copies.append(int(scaled_prob))  # give lower quotas
        residuals.append(scaled_prob - int(scaled_prob))

    rounded_up_indices = pipage_rounding(list(enumerate(residuals)))
    for committee_index in rounded_up_indices:
        num_copies[committee_index] += 1

    committee_lottery: list[frozenset[str]] = []
    for committee, committee_copies in zip(committees, num_copies, strict=False):
        committee_lottery += [committee for _ in range(committee_copies)]

    return committee_lottery


def _distribution_stats(
    people: People,
    committees: list[frozenset[str]],
    probabilities: list[float],
) -> list[str]:
    """Generate statistics about the distribution over committees.

    Args:
        people: People object
        committees: list of committees
        probabilities: corresponding probabilities

    Returns:
        list of output lines with statistics
    """
    output_lines = []

    assert len(committees) == len(probabilities)
    num_non_zero = sum(1 for prob in probabilities if prob > 0)
    output_lines.append(
        f"Algorithm produced distribution over {len(committees)} committees, out of which "
        f"{num_non_zero} are chosen with positive probability."
    )

    individual_probabilities = dict.fromkeys(people, 0.0)
    containing_committees: dict[str, list[frozenset[str]]] = {agent_id: [] for agent_id in people}
    for committee, prob in zip(committees, probabilities, strict=False):
        if prob > 0:
            for agent_id in committee:
                individual_probabilities[agent_id] += prob
                containing_committees[agent_id].append(committee)

    table = [
        "<table border='1' cellpadding='5'><tr><th>Agent ID</th><th>Probability of selection</th><th>Included in # of committees</th></tr>"
    ]

    for _, agent_id in sorted((prob, agent_id) for agent_id, prob in individual_probabilities.items()):
        table.append(
            f"<tr><td>{agent_id}</td><td>{individual_probabilities[agent_id]:.4%}</td><td>{len(containing_committees[agent_id])}</td></tr>"
        )
    table.append("</table>")
    output_lines.append("".join(table))

    return output_lines


def find_random_sample(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    settings: Settings,
    selection_algorithm: str = "maximin",
    test_selection: bool = False,
    number_selections: int = 1,
) -> tuple[list[frozenset[str]], list[str]]:
    """Main algorithm to find one or multiple random committees.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        settings: Settings object containing configuration
        selection_algorithm: one of "legacy", "maximin", "leximin", or "nash"
        test_selection: if set, do not do a random selection, but just return some valid panel.
            Useful for quickly testing whether quotas are satisfiable, but should always be false for actual selection!
        number_selections: how many panels to return. Most of the time, this should be set to 1, which means that
            a single panel is chosen. When specifying a value n ≥ 2, the function will return a list of length n,
            containing multiple panels (some panels might be repeated in the list). In this case the eventual panel
            should be drawn uniformly at random from the returned list.

    Returns:
        tuple of (committee_lottery, output_lines)
        - committee_lottery: list of committees, where each committee is a frozen set of pool member ids
        - output_lines: list of debug strings

    Raises:
        InfeasibleQuotasError: if the quotas cannot be satisfied, which includes a suggestion for how to modify them
        SelectionError: in multiple other failure cases
        ValueError: for invalid parameters
        RuntimeError: if required solver is not available
    """
    # Input validation
    if test_selection and number_selections != 1:
        msg = (
            "Running the test selection does not support generating a transparent lottery, so, if "
            "`test_selection` is true, `number_selections` must be 1."
        )
        raise ValueError(msg)

    if selection_algorithm == "legacy" and number_selections != 1:
        msg = (
            "Currently, the legacy algorithm does not support generating a transparent lottery, "
            "so `number_selections` must be set to 1."
        )
        raise ValueError(msg)

    # Quick test selection using _find_any_committee
    if test_selection:
        print("Running test selection.")
        return _find_any_committee(features, people, number_people_wanted, settings)

    output_lines = []

    # Check if Gurobi is available for leximin
    if selection_algorithm == "leximin" and not GUROBI_AVAILABLE:
        output_lines.append(
            _print(
                "The leximin algorithm requires the optimization library Gurobi to be installed "
                "(commercial, free academic licenses available). Switching to the simpler "
                "maximin algorithm, which can be run using open source solvers."
            )
        )
        selection_algorithm = "maximin"

    # Route to appropriate algorithm
    if selection_algorithm == "legacy":
        # Import here to avoid circular imports
        from sortition_algorithms.find_sample import find_random_sample_legacy

        return find_random_sample_legacy(
            people,
            features,
            number_people_wanted,
            settings.check_same_address,
            settings.check_same_address_columns,
        )
    elif selection_algorithm == "leximin":
        committees, probabilities, new_output_lines = find_distribution_leximin(
            features, people, number_people_wanted, settings
        )
    elif selection_algorithm == "maximin":
        committees, probabilities, new_output_lines = find_distribution_maximin(
            features, people, number_people_wanted, settings
        )
    elif selection_algorithm == "nash":
        committees, probabilities, new_output_lines = find_distribution_nash(
            features, people, number_people_wanted, settings
        )
    else:
        msg = (
            f"Unknown selection algorithm {selection_algorithm!r}, must be either 'legacy', 'leximin', "
            f"'maximin', or 'nash'."
        )
        raise ValueError(msg)

    # Post-process the distribution
    committees, probabilities = standardize_distribution(committees, probabilities)
    if len(committees) > people.count:
        print(
            "INFO: The distribution over panels is what is known as a 'basic solution'. There is no reason for concern "
            "about the correctness of your output, but we'd appreciate if you could reach out to panelot"
            f"@paulgoelz.de with the following information: algorithm={selection_algorithm}, "
            f"num_panels={len(committees)}, num_agents={people.count}, min_probs={min(probabilities)}."
        )

    assert len(set(committees)) == len(committees)

    output_lines += new_output_lines
    output_lines += _distribution_stats(people, committees, probabilities)

    # Convert to lottery
    committee_lottery = lottery_rounding(committees, probabilities, number_selections)

    return committee_lottery, output_lines
