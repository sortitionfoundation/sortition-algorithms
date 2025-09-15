import logging
from typing import Any

import mip
import numpy

try:
    import gurobipy as grb

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    grb = None

from sortition_algorithms.committee_generation.common import (
    EPS,
    generate_initial_committees,
    ilp_results_to_committee,
    setup_committee_generation,
)
from sortition_algorithms.features import FeatureCollection
from sortition_algorithms.people import People
from sortition_algorithms.utils import RunReport, logger


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
    report: RunReport,
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
            logger.debug(f"Status: {dual_model.status} REDUCE PROBS for {reduction_counter}th time.")
            reduction_counter += 1
            continue

        # Find the panel P for which Σ_{i ∈ P} yᵢ is largest, i.e., for which Σ_{i ∈ P} yᵢ ≤ ŷ is tightest
        agent_weights = {agent_id: agent_var.x for agent_id, agent_var in dual_agent_vars.items()}
        new_committee_model.objective = mip.xsum(agent_weights[agent_id] * agent_vars[agent_id] for agent_id in people)
        new_committee_model.optimize()
        new_set = ilp_results_to_committee(agent_vars)  # panel P
        value = new_committee_model.objective_value  # Σ_{i ∈ P} yᵢ

        upper = dual_cap_var.x  # ŷ
        dual_obj = dual_model.objVal  # ŷ - Σ_{i in fixed_probabilities} fixed_probabilities[i] * yᵢ

        report.add_line_and_log(
            f"Maximin is at most {dual_obj - upper + value:.2%}, can do {dual_obj:.2%} with "
            f"{len(committees)} committees. Gap {value - upper:.2%}.",
            log_level=logging.DEBUG,
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

    # Bound variables between 0 and 1 and renormalize, because numpy.random.choice is sensitive to small deviations here
    probabilities = numpy.array([comm_var.x for comm_var in committee_vars]).clip(0, 1)
    return list(probabilities / sum(probabilities))


def _run_leximin_main_loop(
    new_committee_model: mip.model.Model,
    agent_vars: dict[str, mip.entities.Var],
    committees: set[frozenset[str]],
    people: People,
    report: RunReport,
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
        logger.debug(f"Fixed {len(fixed_probabilities)}/{people.count} probabilities.")

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
            report,
        )
        if should_break:
            break

    return fixed_probabilities


def find_distribution_leximin(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    check_same_address_columns: list[str],
) -> tuple[list[frozenset[str]], list[float], RunReport]:
    """Find a distribution over feasible committees that maximizes the minimum probability of an agent being selected
    (just like maximin), but breaks ties to maximize the second-lowest probability, breaks further ties to maximize the
    third-lowest probability and so forth.

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

    Raises:
        RuntimeError: If Gurobi is not available
    """
    if not GUROBI_AVAILABLE:
        msg = "Leximin algorithm requires Gurobi solver which is not available"
        raise RuntimeError(msg)

    report = RunReport()
    report.add_line_and_log("Using leximin algorithm.", log_level=logging.INFO)
    grb.setParam("OutputFlag", 0)

    # Set up an ILP that can be used for discovering new feasible committees
    new_committee_model, agent_vars = setup_committee_generation(
        features, people, number_people_wanted, check_same_address_columns
    )

    # Find initial committees that cover every possible agent
    committees, covered_agents, initial_report = generate_initial_committees(
        new_committee_model, agent_vars, 3 * people.count
    )
    report.add_report(initial_report)

    # Run the main leximin optimization loop to fix agent probabilities
    fixed_probabilities = _run_leximin_main_loop(new_committee_model, agent_vars, committees, people, report)

    # Convert fixed agent probabilities to committee probabilities
    probabilities_normalised = _solve_leximin_primal_for_final_probabilities(committees, fixed_probabilities)

    return list(committees), probabilities_normalised, report
