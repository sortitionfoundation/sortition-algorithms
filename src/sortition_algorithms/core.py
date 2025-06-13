from collections.abc import Iterable
from copy import deepcopy

from sortition_algorithms import errors
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
from sortition_algorithms.people_features import simple_add_selected
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import print_ret, random_provider, set_random_provider


def multi_selection_to_table(multi_selections: list[frozenset[str]]) -> list[list[str]]:
    header_row = [f"Assembly {index}" for index in range(len(multi_selections))]
    # put all the assemblies in columns of the output
    return [header_row, *(list(selection_keys) for selection_keys in multi_selections)]


def person_list_to_table(
    person_keys: Iterable[str],
    people: People,
    features: FeatureCollection,
    settings: Settings,
) -> list[list[str]]:
    cols_to_use = settings.columns_to_keep[:]
    # we want to avoid duplicate columns if they are in both features and columns_to_keep
    extra_features = [name for name in features.feature_names if name not in cols_to_use]
    cols_to_use += extra_features
    rows = [[settings.id_column, *cols_to_use]]
    for pkey in person_keys:
        person_dict = people.get_person_dict(pkey)
        rows.append([pkey, *(person_dict[col] for col in cols_to_use)])
    return rows


def selected_remaining_tables(
    full_people: People,
    people_selected: frozenset[str],
    features: FeatureCollection,
    settings: Settings,
) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """
    write some text

    people_selected is a single frozenset[str] - it must be unwrapped before being passed
    to this function.
    """
    people_working = deepcopy(full_people)
    output_lines: list[str] = []

    people_selected_rows = person_list_to_table(people_selected, people_working, features, settings)

    # now delete the selected people (and maybe also those at the same address)
    num_same_address_deleted = 0
    for pkey in people_selected:
        # if check address then delete all those at this address (will NOT delete the one we want as well)
        if settings.check_same_address:
            pkey_to_delete = list(people_working.matching_address(pkey, settings.check_same_address_columns))
            num_same_address_deleted += len(pkey_to_delete) + 1
            # then delete this/these people at the same address from the reserve/remaining pool
            people_working.remove_many([pkey, *pkey_to_delete])
        else:
            people_working.remove(pkey)

    # add the columns to keep into remaining people
    # as above all these values are all in people_working but this is tidier...
    people_remaining_rows = person_list_to_table(people_working, people_working, features, settings)
    return people_selected_rows, people_remaining_rows, output_lines

    # TODO: put this code somewhere more suitable
    # maybe in strat app only?
    """
    dupes = self._output_selected_remaining(
        settings,
        people_selected_rows,
        people_remaining_rows,
    )
    if settings.check_same_address and self.gen_rem_tab == "on":
        output_lines.append(
            f"Deleted {num_same_address_deleted} people from remaining file who had the same "
            f"address as selected people.",
        )
        m = min(30, len(dupes))
        output_lines.append(
            f"In the remaining tab there are {len(dupes)} people who share the same address as "
            f"someone else in the tab. We highlighted the first {m} of these. "
            f"The full list of lines is {dupes}",
        )
    """


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
            if random_provider().uniform(0.0, 1.0) < prob:
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

            if random_provider().uniform(0.0, 1.0) < choice_probability:  # increase prob0 and decrease prob1
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


def _initial_print_category_info(
    features: FeatureCollection,
    people: People,
) -> list[str]:
    """Generate HTML table showing category/feature statistics.

    Args:
        features: FeatureCollection with min/max targets
        people: People object with pool members
        number_people_wanted: Target number of people to select

    Returns:
        List containing HTML table as single string
    """
    # Build HTML table header
    report_msg = [
        "<table border='1' cellpadding='5'><tr><th colspan='2'>Category</th><th>Initially</th><th>Want</th></tr>"
    ]
    # Make a working copy and update counts
    features_working = deepcopy(features)
    simple_add_selected(people, people, features_working)

    # Generate table rows
    for feature, value, fv_counts in features_working.feature_values_counts():
        report_msg.append(
            f"<tr><td>{feature}</td><td>{value}</td>"
            f"<td>{fv_counts.selected}</td><td>[{fv_counts.min},{fv_counts.max}]</td></tr>"
        )

    report_msg.append("</table>")
    return ["".join(report_msg)]


def _print_category_info(
    features: FeatureCollection,
    people: People,
    people_selected: list[frozenset[str]],
    number_people_wanted: int,
) -> list[str]:
    """Generate HTML table showing category/feature statistics.

    Args:
        features: FeatureCollection with min/max targets
        people: People object with pool members
        people_selected: List of selected committees (empty for initial state)
        number_people_wanted: Target number of people to select

    Returns:
        List containing HTML table as single string
    """
    if len(people_selected) != 1:
        return [
            "<p>We do not calculate target details for multiple selections - please see your output files.</p>",
        ]

    # Build HTML table header
    report_msg = [
        "<table border='1' cellpadding='5'><tr><th colspan='2'>Category</th><th>Selected</th><th>Want</th></tr>"
    ]

    # Make a working copy and update counts
    features_working = deepcopy(features)
    simple_add_selected(people_selected[0], people, features_working)

    # Generate table rows
    for feature, value, fv_counts in features_working.feature_values_counts():
        percent_selected = fv_counts.percent_selected(number_people_wanted)
        report_msg.append(
            f"<tr><td>{feature}</td><td>{value}</td>"
            f"<td>{fv_counts.selected} ({percent_selected:.2f}%)</td>"
            f"<td>[{fv_counts.min},{fv_counts.max}]</td></tr>"
        )

    report_msg.append("</table>")
    return ["".join(report_msg)]


def _check_category_selected(
    features: FeatureCollection,
    people: People,
    people_selected: list[frozenset[str]],
    number_selections: int,
) -> tuple[bool, list[str]]:
    """Check if selected committee meets all feature value targets.

    Args:
        features: FeatureCollection with min/max targets
        people: People object with pool members
        people_selected: List of selected committees
        number_selections: Number of selections made

    Returns:
        Tuple of (success, output_messages)
    """
    if number_selections > 1:
        return True, [
            "<p>No target checks done for multiple selections - please see your output files.</p>",
        ]

    if len(people_selected) != 1:
        return True, [""]

    hit_targets = True
    last_feature_fail = ""

    # Make working copy and count selected people
    from copy import deepcopy

    features_working = deepcopy(features)

    simple_add_selected(people_selected[0], people, features_working)

    # Check if quotas are met
    for (
        feature_name,
        value_name,
        value_counts,
    ) in features_working.feature_values_counts():
        if value_counts.selected < value_counts.min or value_counts.selected > value_counts.max:
            hit_targets = False
            last_feature_fail = f"{feature_name}: {value_name}"

    report_msg = (
        ""
        if hit_targets
        else f"<p>Failed to get minimum or got more than maximum in (at least) category: {last_feature_fail}</p>"
    )
    return hit_targets, [report_msg]


def run_stratification(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    settings: Settings,
    test_selection: bool = False,
    number_selections: int = 1,
) -> tuple[bool, list[frozenset[str]], list[str]]:
    """Run stratified random selection with retry logic.

    Args:
        features: FeatureCollection with min/max quotas for each feature value
        people: People object containing the pool of candidates
        number_people_wanted: Desired size of the panel
        settings: Settings object containing configuration
        test_selection: If True, don't randomize (for testing only)
        number_selections: Number of panels to return

    Returns:
        Tuple of (success, selected_committees, output_lines)
        - success: Whether selection succeeded within max attempts
        - selected_committees: List of committees (frozensets of person IDs)
        - output_lines: Debug and status messages

    Raises:
        Exception: If number_people_wanted is outside valid range for any feature
        ValueError: For invalid parameters
        RuntimeError: If required solver is not available
        InfeasibleQuotasError: If quotas cannot be satisfied
    """
    # Check if desired number is within feature constraints
    features.check_desired(number_people_wanted)

    # Set random seed if specified
    # If the seed is zero or None, we use the secrets module, as it is better
    # from a security point of view
    set_random_provider(settings.random_number_seed)

    success = False
    output_lines = []

    if test_selection:
        output_lines.append(
            "<b style='color: red'>WARNING: Panel is not selected at random! Only use for testing!</b><br>",
        )

    output_lines.append("<b>Initial: (selected = 0)</b>")
    output_lines += _initial_print_category_info(
        features,
        people,
    )
    people_selected: list[frozenset[str]] = []

    tries = 0
    for tries in range(settings.max_attempts):
        people_selected = []

        output_lines.append(f"<b>Trial number: {tries}</b>")

        try:
            people_selected, new_output_lines = find_random_sample(
                features,
                people,
                number_people_wanted,
                settings,
                settings.selection_algorithm,
                test_selection,
                number_selections,
            )
            output_lines += new_output_lines

            # Check if targets were met (only works for number_selections = 1)
            new_output_lines = _print_category_info(
                features,
                people,
                people_selected,
                number_people_wanted,
            )
            success, check_output_lines = _check_category_selected(
                features,
                people,
                people_selected,
                number_selections,
            )

            if success:
                output_lines.append("<b>SUCCESS!!</b> Final:")
                output_lines += new_output_lines + check_output_lines
                break

        except (ValueError, RuntimeError) as err:
            output_lines.append(str(err))
            break
        except errors.InfeasibleQuotasError as err:
            output_lines += err.output
            break
        except errors.SelectionError as serr:
            output_lines.append(f"Failed: Selection Error thrown: {serr}")

    if not success:
        output_lines.append(f"Failed {tries} times... gave up.")

    return success, people_selected, output_lines
