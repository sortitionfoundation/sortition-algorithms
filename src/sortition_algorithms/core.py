from sortition_algorithms.committee_generation import (
    EPS2,
    GUROBI_AVAILABLE,
    find_any_committee,
    find_distribution_leximin,
    find_distribution_maximin,
    find_distribution_nash,
    standardize_distribution,
)
from sortition_algorithms.features import FeatureCollection
from sortition_algorithms.people import People
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import print_ret, secrets_uniform


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

    # Quick test selection using find_any_committee
    if test_selection:
        print("Running test selection.")
        return find_any_committee(features, people, number_people_wanted, settings)

    output_lines = []

    # Check if Gurobi is available for leximin
    if selection_algorithm == "leximin" and not GUROBI_AVAILABLE:
        output_lines.append(
            print_ret(
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
