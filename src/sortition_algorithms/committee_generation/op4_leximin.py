# ABOUTME: Leximin algorithm for committee generation.
# ABOUTME: Maximizes minimum probability, breaking ties by second-lowest, third-lowest, etc.
# Option 4: Heuristic Pricing (The "Good Enough" Approach)

import logging
from typing import Any

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
from sortition_algorithms.committee_generation.solver import Solver, SolverSense, solver_sum
from sortition_algorithms.features import FeatureCollection
from sortition_algorithms.people import People
from sortition_algorithms.utils import RunReport, logger


def _dual_leximin_stage(
    people: People,
    committees: set[frozenset[str]],
    fixed_probabilities: dict[str, float],
) -> tuple:
    if not GUROBI_AVAILABLE:
        msg = "Leximin algorithm requires Gurobi solver which is not available"
        raise RuntimeError(msg, "gurobi_not_available", {})

    assert len(committees) != 0

    model = grb.Model()
    agent_vars = {agent_id: model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.0) for agent_id in people}  
    cap_var = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.0)  
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

    # ---> OPTION 4 SPEEDUP: Let Gurobi use its default Simplex method <---
    # We are commenting out Paul's forced Barrier method.
    # model.setParam("Method", 2)  # optimize via barrier only
    # model.setParam("Crossover", 0)  # deactivate cross-over

    return model, agent_vars, cap_var


def _add_report_update(
    report: RunReport, dual_obj: float, value: float, upper: float, committees: set[frozenset[str]]
) -> None:
    at_most = f"{dual_obj - upper + value:.2%}"
    dual_obj_str = f"{dual_obj:.2%}"
    gap_str = f"{value - upper:.2%}"
    report.add_message_and_log(
        "leximin_is_at_most",
        log_level=logging.DEBUG,
        at_most=at_most,
        dual_obj_str=dual_obj_str,
        num_committees=len(committees),
        gap_str=gap_str,
    )


def _run_leximin_column_generation_loop(
    new_committee_solver: Solver,
    agent_vars: dict[str, Any],
    dual_model: Any,  
    dual_agent_vars: dict[str, Any],  
    dual_cap_var: Any,  
    committees: set[frozenset[str]],
    fixed_probabilities: dict[str, float],
    people: People,
    reduction_counter: int,
    report: RunReport,
) -> tuple[bool, int]:
    while True:
        dual_model.optimize()
        if dual_model.status != grb.GRB.OPTIMAL:
            for agent_id in fixed_probabilities:
                fixed_probabilities[agent_id] = max(0.0, fixed_probabilities[agent_id] - 0.0001)
            dual_model, dual_agent_vars, dual_cap_var = _dual_leximin_stage(
                people,
                committees,
                fixed_probabilities,
            )
            logger.debug(f"Status: {dual_model.status} REDUCE PROBS for {reduction_counter}th time.")
            reduction_counter += 1
            continue

        agent_weights = {agent_id: agent_var.x for agent_id, agent_var in dual_agent_vars.items()}
        new_committee_solver.set_objective(
            solver_sum(agent_weights[agent_id] * agent_vars[agent_id] for agent_id in people), SolverSense.MAXIMIZE
        )
        new_committee_solver.optimize()
        new_set = ilp_results_to_committee(new_committee_solver, agent_vars)  
        value = new_committee_solver.get_objective_value()  

        upper = dual_cap_var.x  
        dual_obj = dual_model.objVal  

        _add_report_update(report, dual_obj, value, upper, committees)
        if value <= upper + EPS:
            for agent_id, agent_weight in agent_weights.items():
                if agent_weight > EPS and agent_id not in fixed_probabilities:
                    fixed_probabilities[agent_id] = max(0, dual_obj)
            return True, reduction_counter  

        assert new_set not in committees
        committees.add(new_set)
        dual_model.addConstr(grb.quicksum(dual_agent_vars[agent_id] for agent_id in new_set) <= dual_cap_var)


def _solve_leximin_primal_for_final_probabilities(
    committees: set[frozenset[str]], fixed_probabilities: dict[str, float]
) -> list[float]:
    primal = grb.Model()
    committee_vars = [primal.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.0) for _ in committees]
    eps = primal.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.0)
    primal.addConstr(grb.quicksum(committee_vars) == 1)  
    for agent_id, prob in fixed_probabilities.items():
        agent_probability = grb.quicksum(
            comm_var for committee, comm_var in zip(committees, committee_vars, strict=False) if agent_id in committee
        )
        primal.addConstr(agent_probability >= prob - eps)
    primal.setObjective(eps, grb.GRB.MINIMIZE)
    primal.optimize()

    probabilities = numpy.array([comm_var.x for comm_var in committee_vars]).clip(0, 1)
    return list(probabilities / sum(probabilities))


def _run_leximin_main_loop(
    new_committee_solver: Solver,
    agent_vars: dict[str, Any],
    committees: set[frozenset[str]],
    people: People,
    report: RunReport,
) -> dict[str, float]:
    fixed_probabilities: dict[str, float] = {}
    reduction_counter = 0

    while len(fixed_probabilities) < people.count:
        logger.debug(f"Fixed {len(fixed_probabilities)}/{people.count} probabilities.")

        dual_model, dual_agent_vars, dual_cap_var = _dual_leximin_stage(
            people,
            committees,
            fixed_probabilities,
        )

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


def find_distribution_leximin(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    check_same_address_columns: list[str],
    solver_backend: str = "highspy",
) -> tuple[list[frozenset[str]], list[float], RunReport]:
    if not GUROBI_AVAILABLE:
        msg = "Leximin algorithm requires Gurobi solver which is not available"
        raise RuntimeError(msg, "gurobi_not_available", {})

    report = RunReport()
    report.add_message_and_log("using_leximin_algorithm", logging.INFO)
    grb.setParam("OutputFlag", 0)

    new_committee_solver, agent_vars = setup_committee_generation(
        features, people, number_people_wanted, check_same_address_columns, solver_backend
    )

    # We keep our Option 3 win here (capped at 100)
    committees, covered_agents, initial_report = generate_initial_committees(
        new_committee_solver, agent_vars, 100
    )
    report.add_report(initial_report)

    fixed_probabilities = _run_leximin_main_loop(new_committee_solver, agent_vars, committees, people, report)

    probabilities_normalised = _solve_leximin_primal_for_final_probabilities(committees, fixed_probabilities)

    return list(committees), probabilities_normalised, report