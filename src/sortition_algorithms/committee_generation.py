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
from sortition_algorithms.utils import print_ret, random_provider

# Tolerance for numerical comparisons
EPS = 0.0005
EPS2 = 0.00000001
EPS_NASH = 0.1


def _reduction_weight(features: FeatureCollection, feature_name: str, value_name: str) -> float:
    """Make the algorithm more reluctant to reduce lower quotas that are already low.
    If the lower quota was 1, reducing it one more (to 0) is 3 times more salient than
    increasing a quota by 1. This bonus tapers off quickly, reducing from 10 is only
    1.2 times as salient as an increase."""
    # Find the current min quota for this feature-value
    # we assume we are only getting value feature/value names as this is called
    # while iterating through a loop
    old_quota = features.get_counts(feature_name, value_name).min
    if old_quota == 0:
        return 0  # cannot be relaxed anyway
    return 1 + 2 / old_quota


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
    # Create relaxation variables for each feature-value pair
    min_vars = {fv: model.add_var(var_type=mip.INTEGER, lb=0.0) for fv in features.feature_value_pairs()}
    max_vars = {fv: model.add_var(var_type=mip.INTEGER, lb=0.0) for fv in features.feature_value_pairs()}

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

    # For each inclusion set, create constraints to ensure a valid committee exists
    for inclusion_set in ensure_inclusion:
        _add_committee_constraints_for_inclusion_set(
            model,
            inclusion_set,
            people,
            number_people_wanted,
            features,
            min_vars,
            max_vars,
            settings,
        )

    # Objective: minimize weighted sum of relaxations
    model.objective = mip.xsum(
        [_reduction_weight(features, *fv) * min_vars[fv] for fv in features.feature_value_pairs()]
        + [max_vars[fv] for fv in features.feature_value_pairs()]
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


def find_any_committee(
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
    weights = {agent_id: random_provider().uniform(0.99, 1.0) for agent_id in agent_vars}

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
                output_lines.append(print_ret(f"Agent {agent_id} not contained in any feasible committee."))

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
        output_lines.append(print_ret("All agents are contained in some feasible committee."))

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
        new_set = _ilp_results_to_committee(agent_vars)
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
    output_lines: list[str],
) -> tuple[list[frozenset[str]], list[float], list[str]]:
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
        new_set = _ilp_results_to_committee(agent_vars)
        value = sum(entitlement_weights[agent_id] for agent_id in new_set)

        output_lines.append(
            print_ret(
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
            print(f"Heuristic successfully generated {counter} additional committees.")


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
    output_lines = [print_ret("Using maximin algorithm.")]

    # Set up an ILP that can be used for discovering new feasible committees
    new_committee_model, agent_vars = _setup_committee_generation(features, people, number_people_wanted, settings)

    # Find initial committees that cover every possible agent
    committees, covered_agents, initial_output = _generate_initial_committees(
        new_committee_model, agent_vars, people.count
    )
    output_lines += initial_output

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
        output_lines,
    )


def _solve_nash_welfare_optimization(
    committees: list[frozenset[str]],
    entitlements: list[str],
    contributes_to_entitlement: dict[str, int],
    start_lambdas: list[float],
    number_people_wanted: int,
    output_lines: list[str],
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
        output_lines.append(print_ret("Had to switch to ECOS solver."))
        nash_welfare = problem.solve(solver=cp.ECOS, warm_start=True)

    scaled_welfare = nash_welfare - len(entitlements) * log(number_people_wanted / len(entitlements))
    output_lines.append(print_ret(f"Scaled Nash welfare is now: {scaled_welfare}."))

    assert lambdas.value.shape == (len(committees),)  # type: ignore[union-attr]
    entitled_utilities = matrix.dot(lambdas.value)  # type: ignore[arg-type]
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

    new_set = _ilp_results_to_committee(agent_vars)
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
    output_lines: list[str],
) -> tuple[list[frozenset[str]], list[float], list[str]]:
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
            output_lines,
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
            return committees, probabilities_normalised, output_lines

        # Add new committee and continue
        print(value, differentials.max(), value - differentials.max())
        assert new_set not in committees
        committees.append(new_set)
        start_lambdas = [
            *list(np.array(lambdas.value)),
            0,
        ]  # Add 0 probability for new committee


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
    output_lines = [print_ret("Using Nash algorithm.")]

    # Set up an ILP used for discovering new feasible committees
    new_committee_model, agent_vars = _setup_committee_generation(features, people, number_people_wanted, settings)

    # Find initial committees that include every possible agent
    committee_set, covered_agents, initial_output = _generate_initial_committees(
        new_committee_model, agent_vars, 2 * people.count
    )
    committees = list(committee_set)
    output_lines += initial_output

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
        output_lines,
    )


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


def _run_leximin_column_generation_loop(
    new_committee_model: mip.model.Model,
    agent_vars: dict[str, mip.entities.Var],
    dual_model: Any,  # grb.Model
    dual_agent_vars: dict[str, Any],  # dict[str, grb.Var]
    dual_cap_var: Any,  # grb.Var
    committees: set[frozenset[str]],
    fixed_probabilities: dict[str, float],
    people: People,
    reduction_counter: int,
    output_lines: list[str],
) -> tuple[bool, int]:
    """Run the column generation inner loop for leximin optimization.

    The primal LP being solved by column generation, with a variable x_P for each feasible panel P:

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

    Args:
        new_committee_model: MIP model for finding committees
        agent_vars: agent variables in the committee model
        dual_model: Gurobi model for dual LP
        dual_agent_vars: agent variables in dual model
        dual_cap_var: capacity variable in dual model
        committees: set of committees (modified in-place)
        fixed_probabilities: probabilities that have been fixed (modified in-place)
        people: People object with all agents
        reduction_counter: counter for probability reductions
        output_lines: list of output messages (modified in-place)

    Returns:
        tuple of (should_break_outer_loop, updated_reduction_counter)
    """
    while True:
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
        new_committee_model.objective = mip.xsum(agent_weights[agent_id] * agent_vars[agent_id] for agent_id in people)
        new_committee_model.optimize()
        new_set = _ilp_results_to_committee(agent_vars)  # panel P
        value = new_committee_model.objective_value  # Σ_{i ∈ P} yᵢ

        upper = dual_cap_var.x  # ŷ
        dual_obj = dual_model.objVal  # ŷ - Σ_{i in fixed_probabilities} fixed_probabilities[i] * yᵢ

        output_lines.append(
            print_ret(
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
            return True, reduction_counter  # Break outer loop

        # Given that Σ_{i ∈ P} yᵢ > ŷ, the current solution to `dual_model` is not yet a solution to the dual.
        # Thus, add the constraint for panel P and recurse.
        assert new_set not in committees
        committees.add(new_set)
        dual_model.addConstr(grb.quicksum(dual_agent_vars[agent_id] for agent_id in new_set) <= dual_cap_var)


def _solve_leximin_primal_for_final_probabilities(
    committees: set[frozenset[str]], fixed_probabilities: dict[str, float]
) -> list[float]:
    """Solve the final primal problem to get committee probabilities from fixed agent probabilities.

    The previous algorithm computed the leximin selection probabilities of each agent and a set of panels such that
    the selection probabilities can be obtained by randomizing over these panels. Here, such a randomization is found.

    Args:
        committees: set of committees
        fixed_probabilities: fixed probabilities for each agent

    Returns:
        list of normalized probabilities for each committee
    """
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
    return list(probabilities / sum(probabilities))


def _run_leximin_main_loop(
    new_committee_model: mip.model.Model,
    agent_vars: dict[str, mip.entities.Var],
    committees: set[frozenset[str]],
    people: People,
    output_lines: list[str],
) -> dict[str, float]:
    """Run the main leximin optimization loop that fixes probabilities iteratively.

    The outer loop maximizes the minimum of all unfixed probabilities while satisfying the fixed probabilities.
    In each iteration, at least one more probability is fixed, but often more than one.

    Args:
        new_committee_model: MIP model for finding committees
        agent_vars: agent variables in the committee model
        committees: set of committees (modified in-place)
        people: People object with all agents
        output_lines: list of output messages (modified in-place)

    Returns:
        dict mapping agent_id to fixed probability
    """
    fixed_probabilities: dict[str, float] = {}
    reduction_counter = 0

    while len(fixed_probabilities) < people.count:
        print(f"Fixed {len(fixed_probabilities)}/{people.count} probabilities.")

        dual_model, dual_agent_vars, dual_cap_var = _dual_leximin_stage(
            people,
            committees,
            fixed_probabilities,
        )

        # Run column generation inner loop
        should_break, reduction_counter = _run_leximin_column_generation_loop(
            new_committee_model,
            agent_vars,
            dual_model,
            dual_agent_vars,
            dual_cap_var,
            committees,
            fixed_probabilities,
            people,
            reduction_counter,
            output_lines,
        )
        if should_break:
            break

    return fixed_probabilities


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

    output_lines = [print_ret("Using leximin algorithm.")]
    grb.setParam("OutputFlag", 0)

    # Set up an ILP that can be used for discovering new feasible committees
    new_committee_model, agent_vars = _setup_committee_generation(features, people, number_people_wanted, settings)

    # Find initial committees that cover every possible agent
    committees, covered_agents, initial_output = _generate_initial_committees(
        new_committee_model, agent_vars, 3 * people.count
    )
    output_lines += initial_output

    # Run the main leximin optimization loop to fix agent probabilities
    fixed_probabilities = _run_leximin_main_loop(new_committee_model, agent_vars, committees, people, output_lines)

    # Convert fixed agent probabilities to committee probabilities
    probabilities_normalised = _solve_leximin_primal_for_final_probabilities(committees, fixed_probabilities)

    return list(committees), probabilities_normalised, output_lines


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
