import pytest

from sortition_algorithms import errors
from sortition_algorithms.features import iterate_feature_collection, read_in_features
from sortition_algorithms.people_features import (
    MaxRatioResult,
    PeopleFeatures,
    iterate_select_collection,
    select_from_feature_collection,
)
from tests.helpers import (
    create_people_with_complex_households,
    create_simple_features,
    create_test_scenario,
    create_test_settings,
)


class TestSelectCollection:
    def test_feature_collection_add_remaining(self):
        """Test the add_remaining functionality."""
        # Create features using the new API
        features_data = [{"feature": "gender", "value": "male", "min": "1", "max": "3"}]
        head = ["feature", "value", "min", "max"]
        features, _, _ = read_in_features(head, features_data)

        select_collection = select_from_feature_collection(features)
        select_counts = select_collection["gender"]["male"]

        assert select_counts.remaining == 0
        select_counts.add_remaining()
        assert select_counts.remaining == 1


class TestPeopleFeatures:
    """Test the PeopleFeatures class."""

    def create_test_people_features(self):
        """Helper to create a test PeopleFeatures object."""
        features, people, settings = create_test_scenario(people_count=4)
        return PeopleFeatures(people, features)

    def test_people_features_creation_copies_data(self):
        """Test that PeopleFeatures creates deep copies of people and features."""
        people_features = self.create_test_people_features()

        # The objects should have the same data but be different instances
        assert people_features.people.count == 4
        assert len(people_features.features) == 2

    def test_update_features_remaining_single_person(self):
        """Test updating remaining counts for a single person."""
        people_features = self.create_test_people_features()

        # Initially, remaining counts should be 0
        for _, _, counts in iterate_select_collection(people_features.select_collection):
            assert counts.remaining == 0

        # Update for one person
        people_features.update_features_remaining("0")  # John: male, young

        # Check that the appropriate counts were incremented
        for (
            feature_name,
            value_name,
            counts,
        ) in iterate_select_collection(people_features.select_collection):
            if (feature_name == "gender" and value_name == "male") or (feature_name == "age" and value_name == "young"):
                assert counts.remaining == 1
            else:
                assert counts.remaining == 0

    def test_update_all_features_remaining(self):
        """Test updating remaining counts for all people."""
        people_features = self.create_test_people_features()

        people_features.update_all_features_remaining()

        # Check expected counts
        expected_counts = {
            ("gender", "male"): 2,  # John, Bob
            ("gender", "female"): 2,  # Jane, Alice
            ("age", "young"): 2,  # John, Jane
            ("age", "old"): 2,  # Bob, Alice
        }

        for (
            feature_name,
            value_name,
            counts,
        ) in iterate_select_collection(people_features.select_collection):
            expected = expected_counts[(feature_name, value_name)]
            assert counts.remaining == expected

    def test_delete_all_with_feature_value(self):
        """Test deleting all people with a specific feature value."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # first set the min target for "young" to zero
        people_features.features["age"]["young"].min = 0
        # Delete all young people
        deleted_count, remaining_count = people_features.delete_all_with_feature_value("age", "young")

        assert deleted_count == 2  # John and Jane were young
        assert remaining_count == 2  # Bob and Alice remain
        assert people_features.people.count == 2

        # Check that John and Jane are gone
        remaining_keys = set(people_features.people)
        assert remaining_keys == {"2", "3"}  # Bob and Alice

    def test_delete_all_with_feature_value_updates_remaining_counts(self):
        """Test that deleting people updates remaining counts correctly."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # first set the min target for "young" to zero
        people_features.features["gender"]["male"].min = 0
        # Delete all males
        people_features.delete_all_with_feature_value("gender", "male")

        # Check remaining counts
        for (
            feature_name,
            value_name,
            counts,
        ) in iterate_select_collection(people_features.select_collection):
            if feature_name == "gender" and value_name == "male":
                assert counts.remaining == 0
            elif feature_name == "gender" and value_name == "female":
                assert counts.remaining == 2  # Jane and Alice remain
            elif feature_name == "age" and value_name == "young":
                assert counts.remaining == 1  # Only Jane remains (John was deleted)
            elif feature_name == "age" and value_name == "old":
                assert counts.remaining == 1  # Only Alice remains (Bob was deleted)

    def test_delete_all_with_feature_value_selection_error(self):
        """Test that deleting people can raise SelectionError when constraints violated."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Set a high minimum for males to trigger the error
        for (
            feature_name,
            value_name,
            counts,
        ) in iterate_select_collection(people_features.select_collection):
            if feature_name == "gender" and value_name == "male":
                counts.min_max.min = 10  # Impossible to satisfy
                counts.selected = 0

        with pytest.raises(errors.SelectionError, match="not enough left in gender male"):
            people_features.delete_all_with_feature_value("gender", "male")

    def test_prune_for_feature_max_0(self):
        """Test pruning people when feature max is set to 0."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Set min and max to 0 for males (don't want any)
        for (
            feature_name,
            value_name,
            counts,
        ) in iterate_feature_collection(people_features.features):
            if feature_name == "gender" and value_name == "male":
                counts.min = 0
                counts.max = 0

        messages = people_features.prune_for_feature_max_0()

        # Should have deleted all males (John and Bob)
        assert people_features.people.count == 2
        remaining_keys = set(people_features.people)
        assert remaining_keys == {"1", "3"}  # Jane and Alice

        # Check messages
        assert any("Feature/value gender/male full - deleting people" in msg for msg in messages)
        assert any("Deleted 2, 2 left" in msg for msg in messages)

    def test_prune_for_feature_max_0_warning_for_many_deletions(self):
        """Test warning message when many people are deleted."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Set max to 0 for most people (delete 3 out of 4)
        for (
            feature_name,
            value_name,
            counts,
        ) in iterate_feature_collection(people_features.features):
            if feature_name == "gender" and value_name == "male":
                counts.min = 0
                counts.max = 0
            if feature_name == "age" and value_name == "young":
                counts.min = 0
                counts.max = 0

        messages = people_features.prune_for_feature_max_0()
        # Should warn about many deletions
        warning_messages = [msg for msg in messages if "WARNING" in msg and "replacement" in msg]
        assert len(warning_messages) == 1

    def test_find_max_ratio_category_basic(self):
        """Test basic functionality of find_max_ratio_category."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # All categories should have equal ratios initially (1-0)/2 = 0.5
        result = people_features.find_max_ratio_category()

        # Should return a valid result
        assert isinstance(result, MaxRatioResult)
        assert result.feature_name in ["gender", "age"]
        assert result.feature_value in ["male", "female", "young", "old"]
        assert 1 <= result.random_person_index <= 2  # 2 people per category

    def test_find_max_ratio_category_different_ratios(self):
        """Test that find_max_ratio_category selects the highest ratio."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Manually adjust counts to create different ratios
        # Make "female" more urgent by increasing its minimum
        for feature_name, value_name, counts in iterate_select_collection(people_features.select_collection):
            if feature_name == "gender" and value_name == "female":
                counts.min_max.min = 2  # Need 2 females, have 2 remaining -> ratio = 2/2 = 1.0
                counts.selected = 0
            elif feature_name == "gender" and value_name == "male":
                counts.min_max.min = 1  # Need 1 male, have 2 remaining -> ratio = 1/2 = 0.5
                counts.selected = 0
            else:
                counts.min_max.min = 1  # Need 1, have 2 remaining -> ratio = 1/2 = 0.5
                counts.selected = 0

        result = people_features.find_max_ratio_category()

        # Should select female due to highest ratio
        assert result.feature_name == "gender"
        assert result.feature_value == "female"
        assert 1 <= result.random_person_index <= 2

    def test_find_max_ratio_category_with_selections_made(self):
        """Test find_max_ratio_category after some people have been selected."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Select one male (this will update selected and remaining counts)
        household_members = people_features.select_person("0")  # John: male, young
        assert household_members == []  # No address checking configured

        # Now ratios should be different:
        # gender/male: (1-1)/1 = 0/1 = 0.0 (minimum already met)
        # gender/female: (1-0)/2 = 1/2 = 0.5 (still need 1 female)
        # age/young: (1-1)/1 = 0/1 = 0.0 (minimum already met)
        # age/old: (1-0)/2 = 1/2 = 0.5 (still need 1 old person)

        result = people_features.find_max_ratio_category()

        # Should select either female or old (both have ratio 0.5)
        assert (result.feature_name == "gender" and result.feature_value == "female") or (
            result.feature_name == "age" and result.feature_value == "old"
        )

    def test_find_max_ratio_category_skips_max_zero(self):
        """Test that categories with max=0 are skipped."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Set max=0 for males (don't want any)
        for feature_name, value_name, counts in iterate_feature_collection(people_features.features):
            if feature_name == "gender" and value_name == "male":
                counts.max = 0
                counts.min = 0

        result = people_features.find_max_ratio_category()

        # Should not select male
        assert not (result.feature_name == "gender" and result.feature_value == "male")

    def test_find_max_ratio_category_skips_no_remaining(self):
        """Test that categories with remaining=0 are skipped."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Set remaining to 0 for males AND reduce minimum to 0 (so it's not an error)
        for feature_name, value_name, counts in iterate_select_collection(people_features.select_collection):
            if feature_name == "gender" and value_name == "male":
                counts.remaining = 0
                counts.min_max.min = 0  # Don't need any males, so remaining=0 is okay

        result = people_features.find_max_ratio_category()

        # Should not select male
        assert not (result.feature_name == "gender" and result.feature_value == "male")

    def test_find_max_ratio_category_insufficient_people_error(self):
        """Test SelectionError when insufficient people remain to meet minimums."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Set impossible requirements: need 5 males but only have 2
        for feature_name, value_name, counts in iterate_select_collection(people_features.select_collection):
            if feature_name == "gender" and value_name == "male":
                counts.min_max.min = 5
                counts.selected = 0
                counts.remaining = 2

        with pytest.raises(errors.SelectionError, match="Not enough people remaining"):
            people_features.find_max_ratio_category()

    def test_find_max_ratio_category_no_valid_categories_error(self):
        """Test SelectionError when no valid categories are found."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Set all categories to max=0 or remaining=0
        for _feature_name, _value_name, counts in iterate_feature_collection(people_features.features):
            counts.max = 0
            counts.min = 0

        with pytest.raises(errors.SelectionError, match="No valid categories found"):
            people_features.find_max_ratio_category()

    def test_find_max_ratio_category_zero_ratio_valid(self):
        """Test that categories with ratio=0 (minimum already met) are still valid."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Set all minimums to 0 (all ratios will be 0)
        for _feature_name, _value_name, counts in iterate_select_collection(people_features.select_collection):
            counts.min_max.min = 0
            counts.selected = 0

        result = people_features.find_max_ratio_category()

        # Should still return a valid result (any category is fine when ratio=0)
        assert isinstance(result, MaxRatioResult)
        assert result.feature_name in ["gender", "age"]

    def test_find_max_ratio_category_random_index_bounds(self):
        """Test that random_person_index is within correct bounds."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Run multiple times to check random index bounds
        for _ in range(10):
            result = people_features.find_max_ratio_category()

            # Find the remaining count for the selected category
            remaining_count = None
            for feature_name, value_name, counts in iterate_select_collection(people_features.select_collection):
                if feature_name == result.feature_name and value_name == result.feature_value:
                    remaining_count = counts.remaining
                    break

            assert remaining_count is not None
            assert 1 <= result.random_person_index <= remaining_count

    def test_select_person_basic(self):
        """Test basic person selection without address checking."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Initial counts
        initial_people_count = people_features.people.count

        # Select John (male, young)
        household_members = people_features.select_person("0")

        # Should return empty list (no address checking)
        assert household_members == []

        # People count should decrease by 1
        assert people_features.people.count == initial_people_count - 1

        # John should be gone
        assert "0" not in people_features.people

        # Check that feature counts were updated correctly
        for feature_name, value_name, counts in iterate_select_collection(people_features.select_collection):
            if feature_name == "gender" and value_name == "male":
                assert counts.selected == 1
                assert counts.remaining == 1  # Bob is still there
            elif feature_name == "age" and value_name == "young":
                assert counts.selected == 1
                assert counts.remaining == 1  # Jane is still there
            else:
                assert counts.selected == 0
                assert counts.remaining == 2  # No change for female/old

    def create_test_people_features_with_addresses(self):
        """Helper to create PeopleFeatures with address data for testing."""
        features = create_simple_features()
        settings = create_test_settings(
            columns_to_keep=["name", "email", "address1", "address2"],
            check_same_address=True,
            check_same_address_columns=["address1", "address2"],
        )
        people = create_people_with_complex_households(features, settings)
        return PeopleFeatures(people, features, check_same_address_columns=["address1", "address2"])

    def test_select_person_with_address_checking(self):
        """Test person selection with automatic household member removal."""
        people_features = self.create_test_people_features_with_addresses()
        people_features.update_all_features_remaining()

        # Initial state: 5 people
        assert people_features.people.count == 5

        # Select John (at 123 Main St) - should also remove Jane and Carol
        household_members = people_features.select_person("0")  # John

        # Should return Jane and Carol as household members
        assert set(household_members) == {"1", "4"}  # Jane and Carol

        # People count should decrease by 3 (John + 2 household members)
        assert people_features.people.count == 2

        # Only Bob and Alice should remain
        remaining_people = set(people_features.people)
        assert remaining_people == {"2", "3"}  # Bob and Alice

        # Check feature counts
        for feature_name, value_name, counts in iterate_select_collection(people_features.select_collection):
            if feature_name == "gender" and value_name == "male":
                assert counts.selected == 1  # Only John was selected (not Bob)
                assert counts.remaining == 1  # Only Bob remains
            elif feature_name == "gender" and value_name == "female":
                assert counts.selected == 0  # Jane and Carol were removed, not selected
                assert counts.remaining == 1  # Only Alice remains
            elif feature_name == "age" and value_name == "young":
                assert counts.selected == 1  # Only John was selected
                assert counts.remaining == 0  # Jane was removed
            elif feature_name == "age" and value_name == "old":
                assert counts.selected == 0  # Bob and Alice remain, Carol was removed
                assert counts.remaining == 2  # Bob and Alice

    def test_select_person_no_household_members(self):
        """Test person selection when no household members exist."""
        people_features = self.create_test_people_features_with_addresses()
        people_features.update_all_features_remaining()

        # Select Bob (at unique address) - should not remove anyone else
        household_members = people_features.select_person("2")  # Bob

        # Should return empty list (no household members)
        assert household_members == []

        # People count should decrease by 1
        assert people_features.people.count == 4

        # Bob should be gone, others remain
        remaining_people = set(people_features.people)
        assert remaining_people == {"0", "1", "3", "4"}  # John, Jane, Alice, Carol

    def test_select_person_empty_address_columns(self):
        """Test that empty address columns disable address checking."""
        people_features = self.create_test_people_features_with_addresses()
        people_features.update_all_features_remaining()

        # Set empty address columns
        people_features.check_same_address_columns = []

        household_members = people_features.select_person("0")
        assert household_members == []

    def test_select_person_feature_counts_consistency(self):
        """Test that feature counts remain consistent after selection with address checking."""
        people_features = self.create_test_people_features_with_addresses()
        people_features.update_all_features_remaining()

        # Record initial remaining counts (what matters for consistency checking)
        initial_remaining = {}
        for feature_name, value_name, counts in iterate_select_collection(people_features.select_collection):
            key = (feature_name, value_name)
            initial_remaining[key] = counts.remaining

        # Select someone with household members
        people_features.select_person("0")  # John (removes Jane and Carol too)

        # Check remaining count changes
        for feature_name, value_name, counts in iterate_select_collection(people_features.select_collection):
            key = (feature_name, value_name)
            remaining_change = initial_remaining[key] - counts.remaining

            # Check expected changes in remaining counts
            if key == ("gender", "male"):
                # John was selected (male) -> remaining should decrease by 1
                assert remaining_change == 1, (
                    f"Expected male remaining to decrease by 1, was {initial_remaining[key]}, now {counts.remaining}"
                )
                assert counts.selected == 1, "John should be selected"
            elif key == ("gender", "female"):
                # Jane and Carol were removed (female) -> remaining should decrease by 2
                assert remaining_change == 2, (
                    f"Expected female remaining to decrease by 2, was {initial_remaining[key]}, now {counts.remaining}"
                )
                assert counts.selected == 0, "No females should be selected, only removed"
            elif key == ("age", "young"):
                # John selected, Jane removed (both young) -> remaining should decrease by 2
                assert remaining_change == 2, (
                    f"Expected young remaining to decrease by 2, was {initial_remaining[key]}, now {counts.remaining}"
                )
                assert counts.selected == 1, "Only John should be selected"
            elif key == ("age", "old"):
                # Carol removed (old) -> remaining should decrease by 1
                assert remaining_change == 1, (
                    f"Expected old remaining to decrease by 1, was {initial_remaining[key]}, now {counts.remaining}"
                )
                assert counts.selected == 0, "No old people should be selected, only removed"

    def test_select_person_updates_feature_counts_for_household_members(self):
        """Test that household member removal properly updates feature counts."""
        people_features = self.create_test_people_features_with_addresses()
        people_features.update_all_features_remaining()

        # Check initial remaining counts for females
        initial_female_remaining = None
        for feature_name, value_name, counts in iterate_select_collection(people_features.select_collection):
            if feature_name == "gender" and value_name == "female":
                initial_female_remaining = counts.remaining
                break

        assert initial_female_remaining == 3  # Jane, Alice, Carol

        # Select John - removes Jane and Carol (both female)
        people_features.select_person("0")

        # Check final counts for females
        for feature_name, value_name, counts in iterate_select_collection(people_features.select_collection):
            if feature_name == "gender" and value_name == "female":
                assert counts.selected == 0  # Household members aren't "selected"
                assert counts.remaining == 1  # Only Alice remains
                break


class TestCaseInsensitivePeopleFeatures:
    """Test that PeopleFeatures handles case-insensitive feature value matching."""

    def test_people_features_with_mismatched_case(self):
        """Test that people data with different case than features still works."""
        # Create features with uppercase values
        features_data = [
            {"feature": "gender", "value": "MALE", "min": "1", "max": "5"},
            {"feature": "gender", "value": "FEMALE", "min": "1", "max": "5"},
            {"feature": "age", "value": "YOUNG", "min": "1", "max": "3"},
            {"feature": "age", "value": "OLD", "min": "1", "max": "3"},
        ]
        head = ["feature", "value", "min", "max"]
        features, _, _ = read_in_features(head, features_data)

        # Create people with lowercase values
        settings = create_test_settings()
        people_data = [
            {"id": "0", "name": "John", "gender": "male", "age": "young"},
            {"id": "1", "name": "Jane", "gender": "female", "age": "young"},
            {"id": "2", "name": "Bob", "gender": "male", "age": "old"},
            {"id": "3", "name": "Alice", "gender": "female", "age": "old"},
        ]
        people_head = ["id", "name", "gender", "age"]
        from sortition_algorithms.people import read_in_people

        people, _ = read_in_people(people_head, people_data, features, settings)

        # Create PeopleFeatures and verify it works
        people_features = PeopleFeatures(people, features)
        people_features.update_all_features_remaining()

        # Verify counts are correct despite case mismatch
        expected_counts = {
            ("gender", "MALE"): 2,  # John, Bob
            ("gender", "FEMALE"): 2,  # Jane, Alice
            ("age", "YOUNG"): 2,  # John, Jane
            ("age", "OLD"): 2,  # Bob, Alice
        }

        for (
            feature_name,
            value_name,
            counts,
        ) in iterate_select_collection(people_features.select_collection):
            expected = expected_counts[(feature_name, value_name)]
            assert counts.remaining == expected, (
                f"Expected {feature_name}/{value_name} remaining to be {expected}, got {counts.remaining}"
            )

    def test_select_person_with_case_mismatch(self):
        """Test that selecting a person works when people data has different case."""
        # Features with mixed case
        features_data = [
            {"feature": "gender", "value": "Male", "min": "0", "max": "2"},
            {"feature": "gender", "value": "Female", "min": "0", "max": "2"},
        ]
        head = ["feature", "value", "min", "max"]
        features, _, _ = read_in_features(head, features_data)

        # People with different case
        settings = create_test_settings()
        people_data = [
            {"id": "0", "name": "John", "gender": "MALE"},
            {"id": "1", "name": "Jane", "gender": "female"},
        ]
        people_head = ["id", "name", "gender"]
        from sortition_algorithms.people import read_in_people

        people, _ = read_in_people(people_head, people_data, features, settings)

        # Create and use PeopleFeatures
        people_features = PeopleFeatures(people, features)
        people_features.update_all_features_remaining()

        # Verify initial remaining counts
        for feature_name, value_name, counts in iterate_select_collection(people_features.select_collection):
            if feature_name == "gender" and value_name.lower() == "male":
                assert counts.remaining == 1  # John
            elif feature_name == "gender" and value_name.lower() == "female":
                assert counts.remaining == 1  # Jane

        # Select John (who has "MALE" while features has "Male")
        people_features.select_person("0")

        # Verify the counts were updated correctly
        for feature_name, value_name, counts in iterate_select_collection(people_features.select_collection):
            if feature_name == "gender" and value_name.lower() == "male":
                assert counts.selected == 1
                assert counts.remaining == 0  # John was removed
            elif feature_name == "gender" and value_name.lower() == "female":
                assert counts.selected == 0
                assert counts.remaining == 1  # Jane still there
