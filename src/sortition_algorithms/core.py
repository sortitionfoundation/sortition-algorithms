import logging
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
    find_random_sample_legacy,
    standardize_distribution,
)
from sortition_algorithms.features import FeatureCollection, check_desired
from sortition_algorithms.people import People
from sortition_algorithms.people_features import (
    iterate_select_collection,
    select_from_feature_collection,
    simple_add_selected,
)
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import ReportLevel, RunReport, logger, random_provider, set_random_provider


def multi_selection_to_table(multi_selections: list[frozenset[str]]) -> list[list[str]]:
    header_row = [f"Assembly {index}" for index in range(len(multi_selections))]
    # put all the assemblies in columns of the output
    return [header_row, *(list(selection_keys) for selection_keys in multi_selections)]


def columns_for_table(features: FeatureCollection, settings: Settings, include_id_column: bool = True) -> list[str]:
    cols_to_use = settings.full_columns_to_keep[:]
    # we want to avoid duplicate columns if they are in both features and columns_to_keep
    extra_features = [name for name in features if name not in cols_to_use]
    cols_to_use += extra_features
    if include_id_column:
        return [settings.id_column, *cols_to_use]
    return cols_to_use


def person_list_to_table(
    person_keys: Iterable[str],
    people: People,
    features: FeatureCollection,
    settings: Settings,
) -> list[list[str]]:
    cols_to_use = columns_for_table(features, settings, include_id_column=False)
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
    if settings.check_same_address and self.gen_rem_tab:
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
) -> RunReport:
    """Generate statistics about the distribution over committees.

    Args:
        people: People object
        committees: list of committees
        probabilities: corresponding probabilities

    Returns:
        list of output lines with statistics
    """
    report = RunReport()

    assert len(committees) == len(probabilities)
    num_non_zero = sum(1 for prob in probabilities if prob > 0)
    report.add_line(
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

    headers = ["Agent ID", "Probability of selection", "Included in # of committees"]

    data: list[list[str | int | float]] = []
    for _, agent_id in sorted((prob, agent_id) for agent_id, prob in individual_probabilities.items()):
        data.append([agent_id, f"{individual_probabilities[agent_id]:.4%}", len(containing_committees[agent_id])])

    report.add_table(headers, data)

    return report


def find_random_sample(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    check_same_address_columns: list[str],
    selection_algorithm: str = "maximin",
    test_selection: bool = False,
    number_selections: int = 1,
) -> tuple[list[frozenset[str]], RunReport]:
    """Main algorithm to find one or multiple random committees.

    Args:
        features: FeatureCollection with min/max quotas
        people: People object with pool members
        number_people_wanted: desired size of the panel
        check_same_address_columns: columns for the address to check, or empty list if no check required
        selection_algorithm: one of "legacy", "maximin", "leximin", or "nash"
        test_selection: if set, do not do a random selection, but just return some valid panel.
            Useful for quickly testing whether quotas are satisfiable, but should always be false for actual selection!
        number_selections: how many panels to return. Most of the time, this should be set to 1, which means that
            a single panel is chosen. When specifying a value n ≥ 2, the function will return a list of length n,
            containing multiple panels (some panels might be repeated in the list). In this case the eventual panel
            should be drawn uniformly at random from the returned list.

    Returns:
        tuple of (committee_lottery, report)
        - committee_lottery: list of committees, where each committee is a frozen set of pool member ids
        - report: report with debug strings

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
        logger.info("Running test selection.")
        return find_any_committee(features, people, number_people_wanted, check_same_address_columns)

    report = RunReport()

    # Check if Gurobi is available for leximin
    if selection_algorithm == "leximin" and not GUROBI_AVAILABLE:
        msg = (
            "The leximin algorithm requires the optimization library Gurobi to be installed "
            "(commercial, free academic licenses available). Switching to the simpler "
            "maximin algorithm, which can be run using open source solvers."
        )
        report.add_line(msg)
        selection_algorithm = "maximin"

    # Route to appropriate algorithm
    if selection_algorithm == "legacy":
        return find_random_sample_legacy(
            people,
            features,
            number_people_wanted,
            check_same_address_columns,
        )
    elif selection_algorithm == "leximin":
        committees, probabilities, new_report = find_distribution_leximin(
            features, people, number_people_wanted, check_same_address_columns
        )
    elif selection_algorithm == "maximin":
        committees, probabilities, new_report = find_distribution_maximin(
            features, people, number_people_wanted, check_same_address_columns
        )
    elif selection_algorithm == "nash":
        committees, probabilities, new_report = find_distribution_nash(
            features, people, number_people_wanted, check_same_address_columns
        )
    else:
        msg = (
            f"Unknown selection algorithm {selection_algorithm!r}, must be either 'legacy', 'leximin', "
            f"'maximin', or 'nash'."
        )
        raise ValueError(msg)

    report.add_report(new_report)

    # Post-process the distribution
    committees, probabilities = standardize_distribution(committees, probabilities)
    if len(committees) > people.count:
        report.add_line_and_log(
            "INFO: The distribution over panels is what is known as a 'basic solution'. There is no reason for concern "
            "about the correctness of your output, but we'd appreciate if you could reach out to panelot"
            f"@paulgoelz.de with the following information: algorithm={selection_algorithm}, "
            f"num_panels={len(committees)}, num_agents={people.count}, min_probs={min(probabilities)}.",
            logging.WARNING,
        )

    assert len(set(committees)) == len(committees)

    stats_report = _distribution_stats(people, committees, probabilities)
    report.add_report(stats_report)

    # Convert to lottery
    committee_lottery = lottery_rounding(committees, probabilities, number_selections)

    return committee_lottery, report


def _initial_category_info_table(
    features: FeatureCollection,
    people: People,
) -> RunReport:
    """Generate HTML table showing category/feature statistics.

    Args:
        features: FeatureCollection with min/max targets
        people: People object with pool members
        number_people_wanted: Target number of people to select

    Returns:
        Report containing table
    """
    # Build HTML table header
    headers = ["Category", "Category Value", "Initially", "Want"]
    # Make a working copy and update counts
    select_collection = select_from_feature_collection(features)
    simple_add_selected(people, people, select_collection)

    # Generate table rows
    data: list[list[str | int | float]] = []
    for feature_name, fvalue_name, fv_counts in iterate_select_collection(select_collection):
        data.append([
            feature_name,
            fvalue_name,
            fv_counts.selected,
            f"[{fv_counts.min_max.min},{fv_counts.min_max.max}]",
        ])

    report = RunReport()
    report.add_table(headers, data)
    return report


def _category_info_table(
    features: FeatureCollection,
    people: People,
    people_selected: list[frozenset[str]],
    number_people_wanted: int,
) -> RunReport:
    """Generate HTML table showing category/feature statistics.

    Args:
        features: FeatureCollection with min/max targets
        people: People object with pool members
        people_selected: List of selected committees (empty for initial state)
        number_people_wanted: Target number of people to select

    Returns:
        List containing HTML table as single string
    """
    report = RunReport()
    if len(people_selected) != 1:
        msg = "We do not calculate target details for multiple selections - please see your output files."
        report.add_line(msg)
        return report

    # Build HTML table header
    headers = ["Category", "Category Value", "Selected", "Want"]

    # Make a working copy and update counts
    select_collection = select_from_feature_collection(features)
    simple_add_selected(people_selected[0], people, select_collection)

    data: list[list[str | int | float]] = []
    # Generate table rows
    for feature_name, fvalue_name, fv_counts in iterate_select_collection(select_collection):
        percent_selected = fv_counts.percent_selected(number_people_wanted)
        data.append([
            feature_name,
            fvalue_name,
            f"{fv_counts.selected} ({percent_selected:.2f}%)",
            f"[{fv_counts.min_max.min},{fv_counts.min_max.max}]",
        ])

    report.add_table(headers, data)
    return report


def _check_category_selected(
    features: FeatureCollection,
    people: People,
    people_selected: list[frozenset[str]],
    number_selections: int,
) -> None:
    """Check if selected committee meets all feature value targets.

    Args:
        features: FeatureCollection with min/max targets
        people: People object with pool members
        people_selected: List of selected committees
        number_selections: Number of selections made

    Returns:
        None. Raises error if targets not hit.
    """
    report = RunReport()
    if number_selections > 1:
        report.add_line("No target checks done for multiple selections - please see your output files.")
        return

    if len(people_selected) != 1:
        return

    # Make working copy and count selected people
    select_collection = select_from_feature_collection(features)
    simple_add_selected(people_selected[0], people, select_collection)

    # Check if quotas are met
    feature_fails: list[str] = []
    for feature_name, fvalue_name, fv_counts in iterate_select_collection(select_collection):
        if not fv_counts.hit_target:
            feature_fails.append(
                f"{feature_name}/{fvalue_name} actual: {fv_counts.selected} "
                f"min: {fv_counts.min_max.min} max: {fv_counts.min_max.max}"
            )

    if not feature_fails:
        return

    raise errors.SelectionMultilineError([
        "Failed to get minimum or got more than maximum in categories:",
        *feature_fails,
    ])


def run_stratification(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    settings: Settings,
    test_selection: bool = False,
    number_selections: int = 1,
) -> tuple[bool, list[frozenset[str]], RunReport]:
    """Run stratified random selection with retry logic.

    Args:
        features: FeatureCollection with min/max quotas for each feature value
        people: People object containing the pool of candidates
        number_people_wanted: Desired size of the panel
        settings: Settings object containing configuration
        test_selection: If True, don't randomize (for testing only)
        number_selections: Number of panels to return

    Returns:
        Tuple of (success, selected_committees, report)
        - success: Whether selection succeeded within max attempts
        - selected_committees: List of committees (frozensets of person IDs)
        - report: contains debug and status messages

    Raises:
        Exception: If number_people_wanted is outside valid range for any feature
        ValueError: For invalid parameters
        RuntimeError: If required solver is not available
        InfeasibleQuotasError: If quotas cannot be satisfied
    """
    # Check if desired number is within feature constraints
    check_desired(features, number_people_wanted)

    # Set random seed if specified
    # If the seed is zero or None, we use the secrets module, as it is better
    # from a security point of view
    set_random_provider(settings.random_number_seed)

    success = False
    report = RunReport()

    if test_selection:
        report.add_line("WARNING: Panel is not selected at random! Only use for testing!", ReportLevel.CRITICAL)

    report.add_line("Initial: (selected = 0)", ReportLevel.IMPORTANT)
    report.add_report(_initial_category_info_table(features, people))
    people_selected: list[frozenset[str]] = []

    tries = 0
    for tries in range(settings.max_attempts):
        people_selected = []

        report.add_line_and_log(f"Trial number: {tries + 1}", logging.WARNING)

        try:
            people_selected, new_report = find_random_sample(
                features,
                people,
                number_people_wanted,
                settings.normalised_address_columns,
                settings.selection_algorithm,
                test_selection,
                number_selections,
            )
            report.add_report(new_report)

            # Check if targets were met (only works for number_selections = 1)
            # This raises an error if we did not select properly
            _check_category_selected(features, people, people_selected, number_selections)

            report.add_line("SUCCESS!! Final:", ReportLevel.IMPORTANT)
            report.add_report(_category_info_table(features, people, people_selected, number_people_wanted))
            success = True
            break

        # these are all fatal errors
        except (ValueError, RuntimeError, errors.InfeasibleQuotasError, errors.InfeasibleQuotasCantRelaxError) as err:
            report.add_error(err)
            break
        except errors.SelectionError as serr:
            report.add_error(serr, is_fatal=False)
            report.add_line(f"Failed one attempt. Selection Error raised - will retry. {serr}")
            # we do not break here, we try again.

    if not success:
        report.add_line(f"Failed {tries + 1} times. Gave up.", level=ReportLevel.IMPORTANT)

    return success, people_selected, report
