import pytest

from sortition_algorithms import errors
from sortition_algorithms.features import FeatureCollection, FeatureValueCounts
from sortition_algorithms.people import People
from sortition_algorithms.people_features import PeopleFeatures
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

        with pytest.raises(errors.SelectionError, match="Failed removing from gender/male"):
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
