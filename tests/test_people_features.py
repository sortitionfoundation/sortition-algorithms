import pytest

from sortition_algorithms import errors
from sortition_algorithms.features import FeatureCollection, FeatureValueCounts
from sortition_algorithms.people import People
from sortition_algorithms.people_features import MaxRatioResult, PeopleFeatures
from sortition_algorithms.utils import StrippedDict


class TestPeopleFeatures:
    """Test the PeopleFeatures class."""

    def create_test_people_features(self):
        """Helper to create a test PeopleFeatures object."""
        columns_to_keep = ["name", "email"]
        people = People(columns_to_keep)

        features = FeatureCollection()
        features.add_feature("gender", "male", FeatureValueCounts(min=1, max=5))
        features.add_feature("gender", "female", FeatureValueCounts(min=1, max=5))
        features.add_feature("age", "young", FeatureValueCounts(min=1, max=3))
        features.add_feature("age", "old", FeatureValueCounts(min=1, max=3))

        # Add some people
        for person_id, (name, gender, age) in enumerate([
            ("John", "male", "young"),
            ("Jane", "female", "young"),
            ("Bob", "male", "old"),
            ("Alice", "female", "old"),
        ]):
            person_data = StrippedDict({
                "id": str(person_id),
                "name": name,
                "email": f"{name.lower()}@example.com",
                "gender": gender,
                "age": age,
            })
            people.add(str(person_id), person_data, features)

        return PeopleFeatures(people, features)

    def test_people_features_creation_copies_data(self):
        """Test that PeopleFeatures creates deep copies of people and features."""
        people_features = self.create_test_people_features()

        # The objects should have the same data but be different instances
        assert people_features.people.count == 4
        assert len(people_features.features.feature_names) == 2

    def test_update_features_remaining_single_person(self):
        """Test updating remaining counts for a single person."""
        people_features = self.create_test_people_features()

        # Initially, remaining counts should be 0
        for _, _, counts in people_features.features.feature_values_counts():
            assert counts.remaining == 0

        # Update for one person
        people_features.update_features_remaining("0")  # John: male, young

        # Check that the appropriate counts were incremented
        for (
            feature_name,
            value_name,
            counts,
        ) in people_features.features.feature_values_counts():
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
        ) in people_features.features.feature_values_counts():
            expected = expected_counts[(feature_name, value_name)]
            assert counts.remaining == expected

    def test_delete_all_with_feature_value(self):
        """Test deleting all people with a specific feature value."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # first set the min target for "young" to zero
        people_features.features.collection["age"].feature_values["young"].min = 0
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
        people_features.features.collection["gender"].feature_values["male"].min = 0
        # Delete all males
        people_features.delete_all_with_feature_value("gender", "male")

        # Check remaining counts
        for (
            feature_name,
            value_name,
            counts,
        ) in people_features.features.feature_values_counts():
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
        ) in people_features.features.feature_values_counts():
            if feature_name == "gender" and value_name == "male":
                counts.min = 10  # Impossible to satisfy
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
        ) in people_features.features.feature_values_counts():
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
        ) in people_features.features.feature_values_counts():
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
        for feature_name, value_name, counts in people_features.features.feature_values_counts():
            if feature_name == "gender" and value_name == "female":
                counts.min = 2  # Need 2 females, have 2 remaining -> ratio = 2/2 = 1.0
                counts.selected = 0
            elif feature_name == "gender" and value_name == "male":
                counts.min = 1  # Need 1 male, have 2 remaining -> ratio = 1/2 = 0.5
                counts.selected = 0
            else:
                counts.min = 1  # Need 1, have 2 remaining -> ratio = 1/2 = 0.5
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
        people_features.select_person("0")  # John: male, young

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
        for feature_name, value_name, counts in people_features.features.feature_values_counts():
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
        for feature_name, value_name, counts in people_features.features.feature_values_counts():
            if feature_name == "gender" and value_name == "male":
                counts.remaining = 0
                counts.min = 0  # Don't need any males, so remaining=0 is okay

        result = people_features.find_max_ratio_category()

        # Should not select male
        assert not (result.feature_name == "gender" and result.feature_value == "male")

    def test_find_max_ratio_category_insufficient_people_error(self):
        """Test SelectionError when insufficient people remain to meet minimums."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Set impossible requirements: need 5 males but only have 2
        for feature_name, value_name, counts in people_features.features.feature_values_counts():
            if feature_name == "gender" and value_name == "male":
                counts.min = 5
                counts.selected = 0
                counts.remaining = 2

        with pytest.raises(errors.SelectionError, match="Not enough people remaining"):
            people_features.find_max_ratio_category()

    def test_find_max_ratio_category_no_valid_categories_error(self):
        """Test SelectionError when no valid categories are found."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Set all categories to max=0 or remaining=0
        for _feature_name, _value_name, counts in people_features.features.feature_values_counts():
            counts.max = 0
            counts.min = 0

        with pytest.raises(errors.SelectionError, match="No valid categories found"):
            people_features.find_max_ratio_category()

    def test_find_max_ratio_category_zero_ratio_valid(self):
        """Test that categories with ratio=0 (minimum already met) are still valid."""
        people_features = self.create_test_people_features()
        people_features.update_all_features_remaining()

        # Set all minimums to 0 (all ratios will be 0)
        for _feature_name, _value_name, counts in people_features.features.feature_values_counts():
            counts.min = 0
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
            for feature_name, value_name, counts in people_features.features.feature_values_counts():
                if feature_name == result.feature_name and value_name == result.feature_value:
                    remaining_count = counts.remaining
                    break

            assert remaining_count is not None
            assert 1 <= result.random_person_index <= remaining_count
