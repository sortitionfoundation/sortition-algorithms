"""Selection algorithms for stratified sampling."""

from collections.abc import MutableMapping
from copy import deepcopy

from attrs import define

from sortition_algorithms import errors
from sortition_algorithms.features import FeatureCollection, iterate_feature_collection
from sortition_algorithms.people import People
from sortition_algorithms.people_features import iterate_select_collection, select_from_feature_collection
from sortition_algorithms.utils import RunReport, random_provider


@define(kw_only=True, slots=True)
class MaxRatioResult:
    """Result from finding the category with maximum selection ratio."""

    feature_name: str
    feature_value: str
    random_person_index: int


class PeopleFeatures:
    """
    This class manipulates people and features together, making a deepcopy on init.

    It is only used by the legacy algorithm.
    """

    # TODO: consider naming: maybe SelectionState

    def __init__(
        self,
        people: People,
        features: FeatureCollection,
        check_same_address_columns: list[str] | None = None,
    ) -> None:
        self.people = deepcopy(people)
        self.features = features
        self.select_collection = select_from_feature_collection(self.features)
        self.check_same_address_columns = check_same_address_columns or []

    def update_features_remaining(self, person_key: str) -> None:
        # this will blow up if the person does not exist
        person = self.people.get_person_dict(person_key)
        for feature_name in self.features:
            feature_value = person[feature_name]
            self.select_collection[feature_name][feature_value].add_remaining()

    def update_all_features_remaining(self) -> None:
        for person_key in self.people:
            self.update_features_remaining(person_key)

    def delete_all_with_feature_value(self, feature_name: str, feature_value: str) -> tuple[int, int]:
        """
        When a feature/value is "full" we delete everyone else in it.
        "Full" means that the number selected equals the "max" amount - that
        is detected elsewhere and then this method is called.
        Returns count of those deleted, and count of those left
        """
        # when a category is full we want to delete everyone in it
        people_to_delete: list[str] = []
        for pkey, person in self.people.items():
            if person[feature_name].lower() == feature_value.lower():
                people_to_delete.append(pkey)
                for feature in self.features:
                    current_feature_value = person[feature]
                    try:
                        self.select_collection[feature][current_feature_value].remove_remaining()
                    except errors.SelectionError as e:
                        msg = (
                            f"SELECTION IMPOSSIBLE: FAIL in delete_all_in_feature_value() "
                            f"as after previous deletion no one/not enough left in {feature} "
                            f"{person[feature]}. Tried to delete: {len(people_to_delete)}"
                        )
                        raise errors.RetryableSelectionError(msg) from e

        self.people.remove_many(people_to_delete)
        # return the number of people deleted and the number of people left
        return len(people_to_delete), self.people.count

    def prune_for_feature_max_0(self) -> list[str]:
        """
        Check if any feature_value.max is set to zero. if so delete everyone with that feature value
        NOT DONE: could then check if anyone is left.
        """
        msg: list[str] = []
        msg.append(f"Number of people: {self.people.count}.")
        total_num_deleted = 0
        for feature_name, fvalue_name, fv_minmax in iterate_feature_collection(self.features):
            if fv_minmax.max == 0:  # we don't want any of these people
                # pass the message in as deleting them might throw an exception
                msg.append(f"Feature/value {feature_name}/{fvalue_name} full - deleting people...")
                num_deleted, num_left = self.delete_all_with_feature_value(feature_name, fvalue_name)
                # if no exception was thrown above add this bit to the end of the previous message
                msg[-1] += f" Deleted {num_deleted}, {num_left} left."
                total_num_deleted += num_deleted
        # if the total number of people deleted is lots then we're probably doing a replacement selection, which means
        # the 'remaining' file will be useless - remind the user of this!
        if total_num_deleted >= self.people.count / 2:
            msg.append(
                ">>> WARNING <<< That deleted MANY PEOPLE - are you doing a "
                "replacement? If so your REMAINING FILE WILL BE USELESS!!!"
            )
        return msg

    def select_person(self, person_key: str) -> list[str]:
        """
        Selecting a person means:
        - remove the person from our copy of People
        - update the `selected` and `remaining` counts of the FeatureCollection
        - if check_same_address_columns has columns, also remove household members (without adding to selected)

        Returns:
            List of additional people removed due to same address (empty if check_same_address_columns is empty)
        """
        # First, find household members if address checking is enabled (before removing the person)
        household_members_removed = []
        if self.check_same_address_columns:
            household_members_removed = list(self.people.matching_address(person_key, self.check_same_address_columns))

        # Handle the main person selection
        person = self.people.get_person_dict(person_key)
        for feature_name in self.features:
            feature_value = person[feature_name]
            self.select_collection[feature_name][feature_value].remove_remaining()
            self.select_collection[feature_name][feature_value].add_selected()
        self.people.remove(person_key)

        # Then remove household members if any were found
        for household_member_key in household_members_removed:
            household_member = self.people.get_person_dict(household_member_key)
            for feature_name in self.features:
                feature_value = household_member[feature_name]
                self.select_collection[feature_name][feature_value].remove_remaining()
                # Note: we don't call add_selected() for household members
            self.people.remove(household_member_key)

        return household_members_removed

    def find_max_ratio_category(self) -> MaxRatioResult:
        """
        Find the feature/value combination with the highest selection ratio.

        The ratio is calculated as: (min - selected) / remaining
        This represents how urgently we need people from this category.
        Higher ratio = more urgent need (fewer people available relative to what we still need).

        Returns:
            MaxRatioResult containing the feature name, value, and a random person index

        Raises:
            SelectionError: If insufficient people remain to meet minimum requirements
        """
        max_ratio = -100.0
        result_feature_name = ""
        result_feature_value = ""
        random_person_index = -1

        for feature_name, fvalue_name, select_counts in iterate_select_collection(self.select_collection):
            # Check if we have insufficient people to meet minimum requirements
            if not select_counts.sufficient_people():
                msg = (
                    f"SELECTION IMPOSSIBLE: Not enough people remaining in {feature_name}/{fvalue_name}. "
                    f"Need {select_counts.people_still_needed} more, but only {select_counts.remaining} remaining."
                )
                raise errors.SelectionError(msg)

            # Skip categories with no remaining people or max = 0
            if select_counts.remaining == 0 or select_counts.min_max.max == 0:
                continue

            # Calculate the priority ratio
            ratio = select_counts.people_still_needed / float(select_counts.remaining)

            # Track the highest ratio category
            if ratio > max_ratio:
                max_ratio = ratio
                result_feature_name = feature_name
                result_feature_value = fvalue_name
                # from 1 to remaining
                random_person_index = random_provider().randbelow(select_counts.remaining) + 1

        # If no valid category found, all categories must be at their max or have max=0
        if not result_feature_name:
            msg = "No valid categories found - all may be at maximum or have max=0"
            raise errors.SelectionError(msg)

        return MaxRatioResult(
            feature_name=result_feature_name,
            feature_value=result_feature_value,
            random_person_index=random_person_index,
        )

    def handle_category_full_deletions(self, selected_person_data: MutableMapping[str, str]) -> RunReport:
        """
        Check if any categories are now full after a selection and delete remaining people.

        When a person is selected, some categories may reach their maximum quota.
        This method identifies such categories and removes all remaining people from them.

        Args:
            selected_person_data: Dictionary of the selected person's feature values

        Returns:
            RunReport containing messages about categories that became full and people deleted

        Raises:
            SelectionError: If deletions would violate minimum constraints
        """
        report = RunReport()

        for feature_name, fvalue_name, select_counts in iterate_select_collection(self.select_collection):
            if (
                fvalue_name.lower() == selected_person_data[feature_name].lower()
                and select_counts.selected == select_counts.min_max.max
            ):
                num_deleted, num_left = self.delete_all_with_feature_value(feature_name, fvalue_name)
                if num_deleted > 0:
                    report.add_line(
                        f"Category {feature_name}/{fvalue_name} full - deleted {num_deleted} people, {num_left} left."
                    )

        return report


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
