from sortition_algorithms.committee_generation.common import (
    EPS2,
    ilp_results_to_committee,
    setup_committee_generation,
)
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
) -> tuple[list[frozenset[str]], RunReport]:
    """Find any single feasible committee that satisfies the quotas.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        check_same_address_columns: columns to check for same address, or empty list if
                                    not checking addresses.

    Returns:
        tuple of (list containing one committee as frozenset of person_ids, empty report)

    Raises:
        InfeasibleQuotasError: If quotas are infeasible
        SelectionError: If solver fails for other reasons
    """
    _, agent_vars = setup_committee_generation(features, people, number_people_wanted, check_same_address_columns)
    committee = ilp_results_to_committee(agent_vars)
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
    "GUROBI_AVAILABLE",
    "find_any_committee",
    "find_distribution_leximin",
    "find_distribution_maximin",
    "find_distribution_nash",
    "find_random_sample_legacy",
    "standardize_distribution",
)
