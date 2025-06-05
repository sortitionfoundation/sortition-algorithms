import pytest

from sortition_algorithms import errors
from sortition_algorithms.features import FeatureCollection, FeatureValueCounts
from sortition_algorithms.find_sample import find_random_sample_legacy
from sortition_algorithms.people import People
from sortition_algorithms.utils import StrippedDict


class TestFindRandomSampleLegacy:
    """Test the find_random_sample_legacy function."""

    def create_test_data(self):
        """Create test people and features for selection testing."""
        columns_to_keep = ["name", "email"]
        people = People(columns_to_keep)

        features = FeatureCollection()
        features.add_feature("gender", "male", FeatureValueCounts(min=1, max=2))
        features.add_feature("gender", "female", FeatureValueCounts(min=1, max=2))
        features.add_feature("age", "young", FeatureValueCounts(min=1, max=2))
        features.add_feature("age", "old", FeatureValueCounts(min=1, max=2))

        # Add test people
        test_people = [
            ("John", "male", "young"),
            ("Jane", "female", "young"),
            ("Bob", "male", "old"),
            ("Alice", "female", "old"),
            ("Carol", "female", "young"),
            ("David", "male", "old"),
        ]

        for person_id, (name, gender, age) in enumerate(test_people):
            person_data = StrippedDict({
                "id": str(person_id),
                "name": name,
                "email": f"{name.lower()}@example.com",
                "gender": gender,
                "age": age,
            })
            people.add(str(person_id), person_data, features)

        return people, features

    def test_basic_selection(self):
        """Test basic selection without address checking."""
        people, features = self.create_test_data()

        committees, messages = find_random_sample_legacy(people, features, 2)

        # Should return one committee with 2 people
        assert len(committees) == 1
        assert len(committees[0]) == 2

        # All selected people should be valid person IDs
        selected_people = committees[0]
        for person_id in selected_people:
            assert person_id in ["0", "1", "2", "3", "4", "5"]

        # Should have output messages
        assert len(messages) >= 1
        assert messages[0] == "Using legacy algorithm."

    def test_selection_respects_quotas(self):
        """Test that selection respects min/max quotas."""
        people, features = self.create_test_data()

        # Run selection multiple times to check quota adherence
        for _ in range(5):
            committees, messages = find_random_sample_legacy(people, features, 4)
            selected_people = committees[0]

            # Count selections by category
            gender_counts = {"male": 0, "female": 0}
            age_counts = {"young": 0, "old": 0}

            # Need to check original people data to count selections
            for person_id in selected_people:
                # Find person in original data
                for pid, person_dict in people.items():
                    if pid == person_id:
                        gender_counts[person_dict["gender"]] += 1
                        age_counts[person_dict["age"]] += 1
                        break

            # Check that quotas are respected (min=1, max=2 for each category)
            for gender, count in gender_counts.items():
                assert 1 <= count <= 2, f"Gender {gender} count {count} violates quotas"
            for age, count in age_counts.items():
                assert 1 <= count <= 2, f"Age {age} count {count} violates quotas"

    def test_insufficient_people_error(self):
        """Test error when asking for more people than available."""
        people, features = self.create_test_data()

        # Try to select more people than exist
        with pytest.raises(errors.SelectionError, match="Selection failed"):
            find_random_sample_legacy(people, features, 10)

    def test_impossible_quotas_error(self):
        """Test error when quotas are impossible to satisfy."""
        columns_to_keep = ["name"]
        people = People(columns_to_keep)

        features = FeatureCollection()
        # Impossible: need 3 males minimum but only 2 people total
        features.add_feature("gender", "male", FeatureValueCounts(min=3, max=5))
        features.add_feature("gender", "female", FeatureValueCounts(min=1, max=2))

        # Add only 2 people, both male
        for i, name in enumerate(["John", "Bob"]):
            person_data = StrippedDict({
                "id": str(i),
                "name": name,
                "gender": "male",
            })
            people.add(str(i), person_data, features)

        with pytest.raises(errors.SelectionError):
            find_random_sample_legacy(people, features, 3)

    def create_test_data_with_addresses(self):
        """Create test data with address information for household testing."""
        columns_to_keep = ["name", "address1", "address2"]
        people = People(columns_to_keep)

        features = FeatureCollection()
        # More lenient constraints to handle household removals
        features.add_feature("gender", "male", FeatureValueCounts(min=0, max=3))
        features.add_feature("gender", "female", FeatureValueCounts(min=0, max=3))

        # Add people with some at same address
        test_people = [
            ("John", "male", "123 Main St", "12345"),  # Same address as Jane
            ("Jane", "female", "123 Main St", "12345"),  # Same address as John
            ("Bob", "male", "456 Oak Ave", "67890"),  # Different address
            ("Alice", "female", "789 Pine Rd", "11111"),  # Different address
            ("Carol", "female", "123 Main St", "12345"),  # Same address as John/Jane
        ]

        for person_id, (name, gender, addr1, addr2) in enumerate(test_people):
            person_data = StrippedDict({
                "id": str(person_id),
                "name": name,
                "address1": addr1,
                "address2": addr2,
                "gender": gender,
            })
            people.add(str(person_id), person_data, features)

        return people, features

    def test_address_checking_enabled(self):
        """Test selection with address checking enabled."""
        people, features = self.create_test_data_with_addresses()

        committees, messages = find_random_sample_legacy(
            people, features, 2, check_same_address=True, check_same_address_columns=["address1", "address2"]
        )

        selected_people = committees[0]
        assert len(selected_people) == 2

        # Check if any household removal messages were generated
        household_messages = [msg for msg in messages if "household members" in msg]

        # If someone from 123 Main St was selected, others should be removed
        selected_at_main_st = []
        for person_id in selected_people:
            if person_id in ["0", "1", "4"]:  # John, Jane, Carol at 123 Main St
                selected_at_main_st.append(person_id)

        # Should have at most 1 person from 123 Main St
        assert len(selected_at_main_st) <= 1, "Multiple people from same address selected"

        # If someone from 123 Main St was selected, there should be household removal messages
        if selected_at_main_st:
            assert len(household_messages) > 0, "Expected household removal messages when selecting from 123 Main St"

    def test_address_checking_disabled(self):
        """Test selection with address checking disabled."""
        people, features = self.create_test_data_with_addresses()

        committees, messages = find_random_sample_legacy(
            people, features, 2, check_same_address=False, check_same_address_columns=["address1", "address2"]
        )

        selected_people = committees[0]
        assert len(selected_people) == 2

        # Should not have household removal messages
        household_messages = [msg for msg in messages if "household members" in msg]
        assert len(household_messages) == 0

    def test_zero_people_wanted(self):
        """Test edge case of selecting zero people."""
        people, features = self.create_test_data()

        committees, messages = find_random_sample_legacy(people, features, 0)

        assert len(committees) == 1
        assert len(committees[0]) == 0
        assert messages == ["Using legacy algorithm."]

    def test_max_zero_pruning(self):
        """Test that people with max=0 categories are pruned."""
        columns_to_keep = ["name"]
        people = People(columns_to_keep)

        features = FeatureCollection()
        features.add_feature("gender", "male", FeatureValueCounts(min=1, max=2))
        features.add_feature("gender", "female", FeatureValueCounts(min=0, max=0))  # Don't want any females

        # Add people
        test_people = [
            ("John", "male"),
            ("Jane", "female"),  # Should be pruned
            ("Bob", "male"),
        ]

        for person_id, (name, gender) in enumerate(test_people):
            person_data = StrippedDict({
                "id": str(person_id),
                "name": name,
                "gender": gender,
            })
            people.add(str(person_id), person_data, features)

        committees, messages = find_random_sample_legacy(people, features, 2)

        # Should only select males (Jane should be pruned)
        selected_people = committees[0]
        assert len(selected_people) == 2
        assert selected_people == {"0", "2"}  # John and Bob

    def test_return_format_compatibility(self):
        """Test that return format matches legacy expectations."""
        people, features = self.create_test_data()

        committees, messages = find_random_sample_legacy(people, features, 1)

        # Should return list of frozensets (legacy format for multi-committee compatibility)
        assert isinstance(committees, list)
        assert len(committees) == 1
        assert isinstance(committees[0], frozenset)

        # Messages should be list of strings
        assert isinstance(messages, list)
        assert all(isinstance(msg, str) for msg in messages)

    def test_selection_is_random(self):
        """Test that selection has random variation."""
        people, features = self.create_test_data()

        # Run selection multiple times and check for variation
        all_selections = []
        for _ in range(10):
            committees, _ = find_random_sample_legacy(people, features, 2)
            all_selections.append(tuple(sorted(committees[0])))

        # Should have some variation in selections (not all identical)
        unique_selections = set(all_selections)
        assert len(unique_selections) > 1, "Selection should have random variation"

    def test_feature_constraints_validation(self):
        """Test that feature constraints are properly validated."""
        people, features = self.create_test_data()

        # This should work with the given quotas
        committees, messages = find_random_sample_legacy(people, features, 4)

        # Verify we got exactly 4 people
        assert len(committees[0]) == 4

        # Verify all selected people are unique
        selected_list = list(committees[0])
        assert len(selected_list) == len(set(selected_list))
