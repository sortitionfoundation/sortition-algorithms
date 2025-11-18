import pytest

from sortition_algorithms import errors
from sortition_algorithms.committee_generation.legacy import find_random_sample_legacy
from sortition_algorithms.features import read_in_features
from sortition_algorithms.utils import RunReport
from tests.helpers import (
    create_gender_only_features,
    create_people_with_legacy_addresses,
    create_simple_features,
    create_simple_people,
    create_test_settings,
)


class TestFindRandomSampleLegacy:
    """Test the find_random_sample_legacy function."""

    def create_test_data(self, person_count: int = 6):
        """Create test people and features for selection testing."""
        features = create_simple_features(gender_min=1, gender_max=2, age_min=1, age_max=2)
        settings = create_test_settings(columns_to_keep=["name", "email"])
        people = create_simple_people(features, settings, count=person_count)
        return people, features

    def test_basic_selection(self):
        """Test basic selection without address checking."""
        people, features = self.create_test_data()

        committees, report = find_random_sample_legacy(people, features, 2)

        # Should return one committee with 2 people
        assert len(committees) == 1
        assert len(committees[0]) == 2

        # All selected people should be valid person IDs
        selected_people = committees[0]
        for person_id in selected_people:
            assert person_id in ["0", "1", "2", "3", "4", "5"]

        # Should have output messages
        assert "Using legacy algorithm" in report.as_text()

    def test_selection_respects_quotas(self):
        """Test that selection respects min/max quotas."""
        # Run selection multiple times to check quota adherence
        for _ in range(5):
            # Create fresh test data each time since legacy function modifies it
            people, features = self.create_test_data(8)
            committees, _ = find_random_sample_legacy(people, features, 4)
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
        # Create impossible quotas: need 3 males minimum but only 2 people total
        features = create_gender_only_features(min_val=3, max_val=5)
        settings = create_test_settings(columns_to_keep=["name"])
        people = create_simple_people(features, settings, count=2)

        # Override both people to be male to match the test scenario
        for person_id in ["0", "1"]:
            person_data = people.get_person_dict(person_id)
            person_data["gender"] = "male"

        with pytest.raises(errors.SelectionError):
            find_random_sample_legacy(people, features, 3)

    def create_test_data_with_addresses(self):
        """Create test data with address information for household testing."""
        features = create_gender_only_features(min_val=0, max_val=3)
        settings = create_test_settings(columns_to_keep=["name", "address1", "address2"])
        people = create_people_with_legacy_addresses(features, settings)
        return people, features

    def test_address_checking_enabled(self):
        """Test selection with address checking enabled."""
        people, features = self.create_test_data_with_addresses()

        committees, report = find_random_sample_legacy(
            people,
            features,
            2,
            check_same_address_columns=["address1", "address2"],
        )

        selected_people = committees[0]
        assert len(selected_people) == 2

        # If someone from 123 Main St was selected, others should be removed
        selected_at_main_st = []
        for person_id in selected_people:
            if person_id in ["0", "1", "4"]:  # John, Jane, Carol at 123 Main St
                selected_at_main_st.append(person_id)

        # Should have at most 1 person from 123 Main St
        assert len(selected_at_main_st) <= 1, "Multiple people from same address selected"

        # If someone from 123 Main St was selected, there should be household removal messages
        if selected_at_main_st:
            assert "household members" in report.as_text(), (
                "Expected household removal messages when selecting from 123 Main St"
            )

    def test_address_checking_disabled(self):
        """Test selection with address checking disabled."""
        people, features = self.create_test_data_with_addresses()

        committees, report = find_random_sample_legacy(
            people,
            features,
            2,
            check_same_address_columns=[],
        )

        selected_people = committees[0]
        assert len(selected_people) == 2

        # Should not have household removal messages
        assert "household members" not in report.as_text()

    def test_zero_people_wanted(self):
        """Test edge case of selecting zero people."""
        people, features = self.create_test_data()

        committees, report = find_random_sample_legacy(people, features, 0)

        assert len(committees) == 1
        assert len(committees[0]) == 0
        assert report.as_text() == "Using legacy algorithm."

    def test_max_zero_pruning(self):
        """Test that people with max=0 categories are pruned."""

        # Create features with males allowed but females not wanted
        features_data = [
            {"feature": "gender", "value": "male", "min": "1", "max": "2"},
            {"feature": "gender", "value": "female", "min": "0", "max": "0"},  # Don't want any females
        ]
        head = ["feature", "value", "min", "max"]
        features, _, _ = read_in_features(head, features_data)

        settings = create_test_settings(columns_to_keep=["name"])
        people = create_simple_people(features, settings, count=3)

        # Override one person to be female (will be pruned)
        person_data = people.get_person_dict("1")
        person_data["gender"] = "female"

        committees, _ = find_random_sample_legacy(people, features, 2)

        # Should only select males (Jane should be pruned)
        selected_people = committees[0]
        assert len(selected_people) == 2
        assert selected_people == {"0", "2"}  # Person0 and Person2

    def test_return_format_compatibility(self):
        """Test that return format matches legacy expectations."""
        people, features = self.create_test_data()

        committees, report = find_random_sample_legacy(people, features, 1)

        # Should return list of frozensets (legacy format for multi-committee compatibility)
        assert isinstance(committees, list)
        assert len(committees) == 1
        assert isinstance(committees[0], frozenset)

        # Messages should be list of strings
        assert isinstance(report, RunReport)

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
        people, features = self.create_test_data(8)

        # This should work with the given quotas
        committees, _ = find_random_sample_legacy(people, features, 4)

        # Verify we got exactly 4 people
        assert len(committees[0]) == 4

        # Verify all selected people are unique
        selected_list = list(committees[0])
        assert len(selected_list) == len(set(selected_list))
