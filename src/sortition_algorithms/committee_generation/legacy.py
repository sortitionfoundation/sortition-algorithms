"""Selection algorithms for stratified sampling."""

from sortition_algorithms import errors
from sortition_algorithms.features import FeatureCollection
from sortition_algorithms.people import People
from sortition_algorithms.people_features import PeopleFeatures
from sortition_algorithms.utils import RunReport


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
    report.add_line("Using legacy algorithm.")
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
            raise errors.SelectionError(msg) from e

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
            category_messages = people_features.handle_category_full_deletions(selected_person_data)
            report.add_lines(category_messages)
        except errors.SelectionError as e:
            msg = f"Selection failed after selecting {selected_person_key}: {e}"
            raise errors.SelectionError(msg) from e

        # Check if we're about to run out of people (but not on the last iteration)
        if count < (number_people_wanted - 1) and people_features.people.count == 0:
            msg = "Selection failed: Ran out of people before completing selection"
            raise errors.SelectionError(msg)

    # Return in legacy format: list containing single frozenset
    return [frozenset(people_selected)], report
