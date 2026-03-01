# Committee Selection Algorithms: A Linear Walkthrough

_2026-03-01T21:05:15Z by Showboat 0.6.1_

<!-- showboat-id: 761d0e51-2c9f-45ac-ae0e-0e424525d2b6 -->

This document walks through every layer of the committee selection system — from the solver abstraction at the bottom, up through the shared setup utilities, and then through each of the five selection algorithms: Legacy, Maximin, Nash, Leximin, and Diversimax.

The system solves a hard problem: given a pool of people with demographic attributes, select a panel of a fixed size that satisfies minimum and maximum quotas on each attribute. The algorithms differ in _how fair_ the resulting selection is, not in _whether_ it satisfies the constraints.

All algorithms (except Legacy) produce a **probability distribution over committees** rather than a single committee. When a panel is needed, one committee is sampled from this distribution. This means every person in the pool has a well-defined probability of being selected.

## Layer 1: The Solver Abstraction

The file `src/sortition_algorithms/committee_generation/solver.py` defines an abstract `Solver` class that hides whether we are using HiGHS or python-mip as the underlying solver engine. All the algorithm code talks to this interface rather than to any specific solver, making backends interchangeable.

The key abstractions are:

- `add_binary_var()` — a 0/1 integer variable (person selected or not)
- `add_integer_var()` — a general integer variable (used in quota relaxation)
- `add_continuous_var()` — a continuous variable (used in LP relaxations and probability distributions)
- `add_constr()` — add a linear constraint
- `set_objective()` + `optimize()` — set the goal and solve
- `get_value()` — read a variable's value after solving

There are two concrete implementations: `HighsSolver` (default, open-source) and `MipSolver` (uses python-mip). The factory function `create_solver()` selects which one to instantiate.

```python
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
```

```python
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
        raise ConfigurationError(
            message=f"Unknown solver backend: {backend}",
            error_code="unknown_solver_backend",
            error_params={"backend": backend},
        )


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
```

## Layer 2: Common Setup — Building the Base ILP

Every algorithm (except Legacy) starts by calling `setup_committee_generation()` in `common.py`. This function constructs the Mixed-Integer Linear Program (MIP) that encodes the basic feasibility constraints, then verifies the constraints are satisfiable before returning.

The ILP contains:

1. **A binary variable per person** — `x_i ∈ {0,1}`, where `x_i = 1` means person `i` is selected.
2. **Panel size constraint** — `Σ x_i = k` (exactly `k` people selected).
3. **Quota constraints** — for every (feature, value) pair (e.g. gender=female), `min ≤ Σ_{i has this value} x_i ≤ max`.
4. **Household constraints** — at most one person per household: `Σ_{i in household} x_i ≤ 1`.

After adding all these constraints, the function does a test optimization to detect infeasibility early. If infeasible, it calls `_relax_infeasible_quotas()` to suggest the smallest quota adjustments that would make the problem solvable, then raises `InfeasibleQuotasError` with those suggestions.

```python
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

```

### Quota Relaxation

When the base ILP is infeasible, `_relax_infeasible_quotas()` is called. It builds a second MIP that asks: "what is the _smallest_ change to the min/max quotas that would make the problem feasible?"

It introduces relaxation variables `min_vars[fv]` and `max_vars[fv]` for each (feature, value) pair. These measure how much to reduce the minimum quota or increase the maximum quota. The objective minimizes a weighted sum of these relaxations, where the weights are designed to penalize reducing already-small minimums more heavily (via `_reduction_weight()`). The `min_flex` and `max_flex` columns in the feature data constrain how far any quota can be moved.

```python
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
```

## Layer 3: Initial Committee Generation (Shared by Maximin, Nash, Leximin)

Before the main optimization loop of the probabilistic algorithms, `generate_initial_committees()` populates a diverse starting set of feasible committees. This matters because the main loops use column generation (adding committees one at a time), and a better starting set means fewer iterations.

**Phase 1: Multiplicative Weights.** In each round, the solver maximizes a weighted sum of agent binary variables. Agents that have appeared in recently found committees get their weights reduced by a factor of 0.8, so future rounds naturally seek committees including _different_ people. If a duplicate committee is found, all weights are nudged slightly toward equality to break out of local optima. This continues for `multiplicative_weights_rounds` rounds (typically `people.count` or `2 * people.count`).

**Phase 2: Covering Uncovered Agents.** After Phase 1, some agents may still not appear in any discovered committee. For each such agent, the solver runs once more with the single objective of including that agent. If no feasible committee can include them, a warning is logged (they will have probability 0 in the final distribution).

```python
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
```

## Algorithm 1: Legacy

`legacy.py` implements the original greedy algorithm. Unlike the others, it produces a single committee directly — not a probability distribution — so there is no notion of individual selection probabilities.

**The loop:** On each of the `k` iterations (selecting `k` people total):

1. **Find the most urgent category.** `PeopleFeatures.find_max_ratio_category()` computes, for every (feature, value) pair, the ratio `(min_remaining - already_selected) / people_still_eligible`. The category with the highest ratio is the most "at risk" of not meeting its minimum quota and must be prioritised.

2. **Randomly select within that category.** A random position within the eligible people in that category is chosen uniformly.

3. **Remove housemates.** If address checking is on, all other household members of the selected person are removed from the pool.

4. **Handle full categories.** Once a category's maximum quota is met, anyone whose _only_ remaining categories are full is removed from the pool (they can no longer help meet any unmet quota).

This greedy approach is fast but produces no fairness guarantees — the same people are likely to be selected every time.

```python
def find_random_sample_legacy(
    people: People,
    features: FeatureCollection,
    number_people_wanted: int,
    check_same_address_columns: list[str] | None = None,
) -> tuple[list[frozenset[str]], RunReport]:
    """
    Legacy stratified random selection algorithm.

    Implements the original algorithm that uses greedy selection based on priority ratios.
    Always selects from the most urgently needed category first (highest ratio of
    (min-selected)/remaining), then randomly picks within that category.

    Args:
        people: People collection
        features: Feature definitions with min/max targets
        number_people_wanted: Number of people to select
        check_same_address_columns: Address columns for household identification, or empty
                                    if no address checking to be done.

    Returns:
        Tuple of (selected_committees, output_messages) where:
        - selected_committees: List containing one frozenset of selected person IDs
        - report: report containing log messages about the selection process

    Raises:
        SelectionError: If selection becomes impossible (not enough people, etc.)
    """
    report = RunReport()
    report.add_message("using_legacy_algorithm")
    people_selected: set[str] = set()

    # Create PeopleFeatures and initialize
    people_features = PeopleFeatures(people, features, check_same_address_columns or [])
    people_features.update_all_features_remaining()
    people_features.prune_for_feature_max_0()

    # Main selection loop
    for count in range(number_people_wanted):
        # Find the category with highest priority ratio
        try:
            ratio_result = people_features.find_max_ratio_category()
        except errors.SelectionError as e:
            msg = f"Selection failed on iteration {count + 1}: {e}"
            raise errors.RetryableSelectionError(msg) from e

        # Find the randomly selected person within that category
        target_feature = ratio_result.feature_name
        target_value = ratio_result.feature_value
        random_position = ratio_result.random_person_index

        selected_person_key = people_features.people.find_person_by_position_in_category(
            target_feature, target_value, random_position
        )

        # Should never select the same person twice
        assert selected_person_key not in people_selected, f"Person {selected_person_key} was already selected"

        # Select the person (this also removes household members if configured)
        people_selected.add(selected_person_key)
        selected_person_data = people_features.people.get_person_dict(selected_person_key)
        household_members_removed = people_features.select_person(selected_person_key)

        # Add output messages about household member removal
        if household_members_removed:
            report.add_line(
                f"Selected {selected_person_key}, also removed household members: "
                f"{', '.join(household_members_removed)}"
            )

        # Handle any categories that are now full after this selection
        try:
            category_report = people_features.handle_category_full_deletions(selected_person_data)
            report.add_report(category_report)
        except errors.SelectionError as e:
            msg = f"Selection failed after selecting {selected_person_key}: {e}"
            raise errors.RetryableSelectionError(msg) from e

        # Check if we're about to run out of people (but not on the last iteration)
        if count < (number_people_wanted - 1) and people_features.people.count == 0:
            msg = "Selection failed: Ran out of people before completing selection"
            raise errors.RetryableSelectionError(msg)

    # Return in legacy format: list containing single frozenset
    return [frozenset(people_selected)], report
```

## Algorithm 2: Maximin

`maximin.py` uses **column generation** to find a probability distribution over feasible committees that maximises the minimum selection probability across all agents. If there are `n` agents with probabilities `p_1, ..., p_n`, maximin maximises `min(p_1, ..., p_n)`.

### The Dual Formulation

The algorithm works in the "dual" space. For each agent `i`, introduce a dual weight `y_i >= 0` with `sum(y_i) = 1`. It solves:

    minimize  z
    s.t.      sum_{i in B} y_i <= z    for every feasible committee B
              sum_i y_i = 1
              y_i >= 0

The intuition: `z` ends up being the smallest "load" any committee can impose. By LP duality, the optimal `z` equals the optimal maximin probability.

### The Column Generation Loop

We cannot enumerate all feasible committees (there can be exponentially many). Instead:

1. Start with the initial committee set from `generate_initial_committees()`.
2. Solve the incremental LP (the dual above, but only over known committees).
3. Get the current optimal `y_i` values. Find the feasible committee `B` that maximally violates the constraint — the one maximising `sum_{i in B} y_i` — by solving the committee ILP with weights set to `y_i`.
4. If the maximum `sum_{i in B} y_i <= z + eps`, the dual is solved. Extract the final probabilities via `_find_maximin_primal()`.
5. Otherwise, add `B` to the committee set and go to step 2.

A heuristic also runs after each new committee is found: it tweaks the current `y_i` values to find additional committees cheaply, without re-optimising the full LP. This reduces the total number of LP solves.

```python
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
```

```python
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

```

## Algorithm 3: Nash

`nash.py` maximises the **Nash social welfare**: the product of all agents' selection probabilities, `p_1 * p_2 * ... * p_n`. Equivalently (since log is monotone), it maximises the sum of logs: `sum_i log(p_i)`.

Agents who cannot appear in any feasible committee are excluded (otherwise `log(0) = -inf` would dominate). Nash welfare is a classic concept in fair division: unlike maximin, it balances equality with the overall "size of the pie" — it won't sacrifice everyone else's chances just to squeeze out a tiny improvement for the worst-off person.

### The Optimisation

Given a set of committees `B_1, ..., B_m` and probabilities `lambda_1, ..., lambda_m` (with `sum lambda_j = 1`), agent `i`'s selection probability is `p_i = sum_{j: i in B_j} lambda_j`.

The objective `sum_i log(p_i)` is concave, so it can be solved via convex optimisation. The code uses **cvxpy** to set this up:

    maximize  sum(log(A @ lambda))
    s.t.      lambda >= 0,  sum(lambda) == 1

where `A` is the binary membership matrix (rows = agents, columns = committees).

### Column Generation Loop

Like maximin, Nash uses column generation:

1. Start with initial committees.
2. Solve the Nash convex problem for current committees to get `lambda*`.
3. Compute the reciprocals of current utilities: `r_i = 1 / p_i`. These are the gradients of the objective.
4. Find the committee maximising `sum_{i in B} r_i` — the ILP with weights `r_i`. This is the "most beneficial" committee to add.
5. If the best committee's value is within `EPS_NASH = 0.1` of the maximum differential already in the current set, converge and return.
6. Otherwise, add the new committee and go to step 2.

```python
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
        report.add_message_and_log("switched_to_ecos_solver", logging.INFO)
        nash_welfare = problem.solve(solver=cp.ECOS, warm_start=True)

    scaled_welfare = nash_welfare - len(entitlements) * log(number_people_wanted / len(entitlements))
    report.add_message_and_log("scaled_nash_welfare", log_level=logging.INFO, scaled_welfare=scaled_welfare)

    assert lambdas.value.shape == (len(committees),)
    entitled_utilities = matrix.dot(lambdas.value)
    assert entitled_utilities.shape == (len(entitlements),)
    assert (entitled_utilities > EPS2).all()
    entitled_reciprocals = 1 / entitled_utilities
    assert entitled_reciprocals.shape == (len(entitlements),)
    differentials = entitled_reciprocals.dot(matrix)
    assert differentials.shape == (len(committees),)

    return lambdas, entitled_reciprocals, differentials

```

```python
def _run_nash_optimization_loop(
    solver: Solver,
    agent_vars: dict[str, Any],
    committees: list[frozenset[str]],
    entitlements: list[str],
    contributes_to_entitlement: dict[str, int],
    covered_agents: frozenset[str],
    number_people_wanted: int,
    report: RunReport,
) -> tuple[list[frozenset[str]], list[float], RunReport]:
    """Run the main Nash welfare optimization loop.

    Args:
        solver: Solver for finding committees
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
            solver,
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


```

## Algorithm 4: Leximin

`leximin.py` is the most fairness-demanding algorithm. It maximises the **leximin** objective: first maximise the minimum probability, then (among all distributions achieving that minimum) maximise the second-lowest probability, then the third-lowest, and so on.

Leximin requires the **Gurobi** solver and is typically much slower than Maximin or Nash.

### The Primal LP (Column Generation)

The primal LP has a variable `x_P` for each feasible committee `P`:

    maximize  z
    s.t.      sum_{P : i in P} x_P >= z       for all i not in fixed_probabilities
              sum_{P : i in P} x_P >= fixed_probabilities[i]  for all i in fixed_probabilities
              sum_P x_P <= 1
              x_P >= 0

`z` is the minimum probability among agents whose probabilities haven't been fixed yet. The "fixed" agents are those whose optimal probability has already been determined in previous outer iterations.

### The Dual LP

Rather than working with the primal directly (which would require enumerating all committees), the algorithm uses the dual:

    minimize  y_cap - sum_{i in fixed} fixed_prob[i] * y_i
    s.t.      sum_{i in P} y_i <= y_cap    for all feasible P
              sum_{i not in fixed} y_i = 1
              y_cap, y_i >= 0

### The Outer Loop (Fixing Probabilities)

The outer loop runs until all agent probabilities are fixed:

1. Set up the dual LP for the current fixed set.
2. Run column generation: solve the dual LP, find the tightest-violated panel, add it, repeat until convergence.
3. When the dual converges, any agent with a positive dual variable `y_i` must have a binding constraint in the primal — their probability is exactly the optimal `z`. Fix those probabilities.
4. Go back to step 1 with the newly extended fixed set.

### Extracting Committee Probabilities

After all agent probabilities are fixed, a final primal solve finds committee probabilities `lambda_j` that realise those agent probabilities.

```python
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
        raise RuntimeError(msg, "gurobi_not_available", {})

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
```

```python
def _run_leximin_main_loop(
    new_committee_solver: Solver,
    agent_vars: dict[str, Any],
    committees: set[frozenset[str]],
    people: People,
    report: RunReport,
) -> dict[str, float]:
    """Run the main leximin optimization loop that fixes probabilities iteratively.

    The outer loop maximizes the minimum of all unfixed probabilities while satisfying the fixed probabilities.
    In each iteration, at least one more probability is fixed, but often more than one.

    Args:
        new_committee_solver: Solver for finding committees
        agent_vars: agent variables in the committee solver
        committees: set of committees (modified in-place)
        people: People object with all agents
        output_lines: list of output messages (modified in-place)

    Returns:
        dict mapping agent_id to fixed probability
    """
    fixed_probabilities: dict[str, float] = {}
    reduction_counter = 0

    while len(fixed_probabilities) < people.count:
        logger.debug(f"Fixed {len(fixed_probabilities)}/{people.count} probabilities.")

        dual_model, dual_agent_vars, dual_cap_var = _dual_leximin_stage(
            people,
            committees,
            fixed_probabilities,
        )

        # Run column generation inner loop
        should_break, reduction_counter = _run_leximin_column_generation_loop(
            new_committee_solver,
            agent_vars,
            dual_model,
            dual_agent_vars,
            dual_cap_var,
            committees,
            fixed_probabilities,
            people,
            reduction_counter,
            report,
        )
        if should_break:
            break

    return fixed_probabilities

```

## Algorithm 5: Diversimax

`diversimax.py` takes a completely different approach. Rather than optimising agent selection probabilities, it selects a **single committee** (no distribution) that maximises demographic diversity across all intersections of features.

Diversimax requires `pandas` and `scikit-learn` (optional dependencies).

### The Diversity Objective

Given features like gender (male/female) and age (young/old), the possible single-feature categories are `{male, female, young, old}`, and the two-feature intersection is `{male+young, male+old, female+young, female+old}`. For each such intersection, the algorithm measures how many selected panellists fall into it.

The ideal is **perfect equality** across an intersection: if there are 4 intersections and 12 people selected, the best outcome is 3 per intersection. The objective minimises the total absolute deviation from this ideal, summed over all intersections of all feature combinations.

### Building the MIP

The `DiversityOptimizer` class:

1. **`prepare_all_data()`** — computes all combinations of features (size 1, 2, 3, ...) and for each, determines which intersection every person belongs to.
2. **`create_all_one_hot_encodings()`** — for each feature combination that passes filtering (not too many intersections, pool not too sparse), creates a one-hot matrix: rows = people, columns = intersections. Capped at 50 matrices to bound computation.
3. **`optimize()`** — builds the MIP. For each intersection column in each OHE matrix, adds a continuous "absolute difference" support variable constrained by `abs_diff >= intersection_size - best_val` and `abs_diff >= best_val - intersection_size`. The objective minimises the sum of all these abs_diff variables.

The standard quota and household constraints are also added, so the solution is always a feasible committee.

```python
    def prepare_all_data(self) -> AllIntersectionsData:
        """
        For each combination of features, prepares the intersection data.
        The data is a dict where key is the combination of category names (e.g. (age, income, education level))
        and value is the IntersectionData for that combination.
        all_dims_combs is a list of all the intersections of all categories of all features.
        (e.g. [(age,), (income,), (education level,), (income, education level), (age, income, education level)])
        """
        all_dims_combs_iterators = [
            itertools.combinations(self.pool_members_df.columns, r=i)
            for i in range(1, self.pool_members_df.shape[1] + 1)
        ]
        all_dims_combs: list[InteractionNamesTuple] = [x for y in all_dims_combs_iterators for x in y]
        # randomize order within each size, so if we cutoff intersections at len(all_ohe) >= 50, we get a random sample
        rng = np.random.default_rng(random_provider().randint())
        rng.shuffle(all_dims_combs)
        all_dims_combs = sorted(all_dims_combs, key=len)
        data: dict[InteractionNamesTuple, IntersectionData] = {}
        for features_intersections in all_dims_combs:
            try:
                data[features_intersections] = self._prepare_features_data(features_intersections)
            except Exception as e:
                raise Exception(f"Error with {features_intersections}") from e
        return AllIntersectionsData(data=data, all_dims_combs=all_dims_combs)

    def create_all_one_hot_encodings(self) -> list[np.ndarray]:
        """
        For every intersection of features - one hot encode who is in which intersection.
        Rows are the different people. Columns are the different possible intersections.
        The values are 0/1 if the person is in that intersection or not.
        i.e. The number of columns is the number of combinations we have between the features: product of their sizes.
        """
        all_ohe: list[np.ndarray] = []
        for dims in self.intersections_data.all_dims_combs:
            intersection_data: IntersectionData = self.intersections_data.data[dims]
            possible_profiles = intersection_data.intersections_names
            # When there are too many intersections, each will have 0/1 people, and optimization is pointless + too slow
            if len(possible_profiles) > self.panel_size:
                continue
            if len(self.pool_members_df) / len(possible_profiles) < 2:
                continue
            if len(all_ohe) >= 50:  # limit to 50 different intersections to avoid too much computation
                break

            ohe = OneHotEncoder(categories=[possible_profiles], sparse_output=False)
            ohe_values = ohe.fit_transform(intersection_data.intersection_member_values.reshape(-1, 1))
            all_ohe.append(ohe_values)
        return all_ohe

```

```python

        # define the optimization goal
        all_objectives = []
        for ohe in self.all_ohe:  # for every set of intersections (one hot encoded) of features
            # how many selected to each intersection
            # We need to build expressions for each intersection column
            for col_idx in range(ohe.shape[1]):
                # For this intersection column, sum up the binary vars weighted by ohe membership
                intersection_size = solver_sum(
                    model_variables_list[row_idx] * ohe[row_idx, col_idx]
                    for row_idx in range(len(model_variables_list))
                )
                # the best value is if all intersections were equal size
                best_val = self.panel_size / ohe.shape[1]

                # set support variable that is the abs diff from intersection size to best_val
                abs_diff = solver.add_continuous_var(lb=0.0, ub=float("inf"))
                # constrain this support variable to be the abs diff from the intersection size
                solver.add_constr(abs_diff >= (intersection_size - best_val))
                solver.add_constr(abs_diff >= (best_val - intersection_size))

                all_objectives.append(abs_diff)

        obj = solver_sum(all_objectives)
        solver.set_objective(obj, SolverSense.MINIMIZE)

        logger.info(f"Diversimax: {len(model_variables)} binary vars, {len(all_objectives)} abs-diff vars")

        status = solver.optimize()
        gap = solver.get_gap()

        if status in [SolverStatus.OPTIMAL, SolverStatus.FEASIBLE]:
            selected_ids = []
            for person_id in df.index:
                var = solver.get_var_by_name(str(person_id))
                if var is not None and solver.get_value(var) >= 0.99:  # ==1, for numerical stability
                    selected_ids.append(person_id)
            selected_ids_set = frozenset(selected_ids)
            return status, selected_ids_set, gap
        return status, frozenset(), gap

```

## How the Algorithms Compare

| Algorithm      | Output           | Fairness criterion              | Solver needed      | Speed                   |
| -------------- | ---------------- | ------------------------------- | ------------------ | ----------------------- |
| **Legacy**     | Single committee | None (greedy)                   | None (pure Python) | Very fast               |
| **Maximin**    | Distribution     | Maximise minimum prob           | HiGHS or MIP       | Fast                    |
| **Nash**       | Distribution     | Maximise product of probs       | HiGHS/MIP + cvxpy  | Moderate                |
| **Leximin**    | Distribution     | Lexicographic min-max           | Gurobi (required)  | Slow                    |
| **Diversimax** | Single committee | Maximise intersection diversity | HiGHS or MIP       | Moderate (time-limited) |

### Return Types

- **Legacy and Diversimax** return `(frozenset[str], RunReport)` — a single committee.
- **Maximin, Nash, Leximin** return `(list[frozenset[str]], list[float], RunReport)` — a list of committees with corresponding probabilities. The caller samples one committee from this distribution using `numpy.random.choice`.

### Entry Points (from `__init__.py`)

- `find_random_sample_legacy()` — Legacy
- `find_distribution_maximin()` — Maximin
- `find_distribution_nash()` — Nash
- `find_distribution_leximin()` — Leximin
- `find_distribution_diversimax()` — Diversimax
- `find_any_committee()` — runs the base ILP and returns whatever feasible committee the solver finds first (no fairness objective)
- `standardize_distribution()` — utility to remove zero-probability committees and renormalise

```python
from sortition_algorithms.committee_generation.common import (
    EPS2,
    ilp_results_to_committee,
    setup_committee_generation,
)
from sortition_algorithms.committee_generation.diversimax import DIVERSIMAX_AVAILABLE
from sortition_algorithms.committee_generation.legacy import find_random_sample_legacy
from sortition_algorithms.committee_generation.leximin import GUROBI_AVAILABLE, find_distribution_leximin
from sortition_algorithms.committee_generation.maximin import find_distribution_maximin
from sortition_algorithms.committee_generation.nash import find_distribution_nash
from sortition_algorithms.features import FeatureCollection
from sortition_algorithms.people import People
from sortition_algorithms.utils import RunReport


def find_any_committee(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    check_same_address_columns: list[str],
    solver_backend: str = "highspy",
) -> tuple[list[frozenset[str]], RunReport]:
    """Find any single feasible committee that satisfies the quotas.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        check_same_address_columns: columns to check for same address, or empty list if
                                    not checking addresses.
        solver_backend: solver backend to use ("highspy" or "mip")

    Returns:
        tuple of (list containing one committee as frozenset of person_ids, empty report)

    Raises:
        InfeasibleQuotasError: If quotas are infeasible
        SelectionError: If solver fails for other reasons
    """
    solver, agent_vars = setup_committee_generation(
        features, people, number_people_wanted, check_same_address_columns, solver_backend
    )
    committee = ilp_results_to_committee(solver, agent_vars)
    return [committee], RunReport()


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


__all__ = (
    "DIVERSIMAX_AVAILABLE",
    "GUROBI_AVAILABLE",
    "find_any_committee",
    "find_distribution_leximin",
    "find_distribution_maximin",
    "find_distribution_nash",
    "find_random_sample_legacy",
    "standardize_distribution",
)
```
