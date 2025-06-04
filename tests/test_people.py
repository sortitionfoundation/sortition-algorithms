from pathlib import Path

import pytest

from sortition_algorithms import errors
from sortition_algorithms.features import FeatureCollection, FeatureValueCounts
from sortition_algorithms.people import (
    People,
    PeopleFeatures,
    _check_columns_exist_or_multiple,
    _ensure_settings_keep_address_columns,
    read_in_people,
)
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import StrippedDict


class TestPeople:
    """Test the People class directly."""

    def test_people_creation_and_basic_properties(self):
        """Test creating a People object and its basic properties."""
        columns_to_keep = ["name", "email", "address"]
        people = People(columns_to_keep)

        assert people.count == 0
        assert list(people) == []

    def test_people_add_person(self):
        """Test adding a person to the People collection."""
        columns_to_keep = ["name", "email"]
        people = People(columns_to_keep)

        # Create mock features
        features = FeatureCollection()
        features.add_feature("gender", "male", FeatureValueCounts(min=1, max=5))
        features.add_feature("gender", "female", FeatureValueCounts(min=1, max=5))

        # Create person data
        person_data = StrippedDict({
            "id": "123",
            "name": "John Doe",
            "email": "john@example.com",
            "gender": "male",
            "age": "30",
        })

        people.add("123", person_data, features)

        assert people.count == 1
        # testing both methods, to be sure
        assert "123" in people

        person_dict = people.get_person_dict("123")
        assert person_dict["name"] == "John Doe"
        assert person_dict["email"] == "john@example.com"
        assert person_dict["gender"] == "male"

    def test_people_add_person_with_invalid_feature_value(self):
        """Test adding a person with an invalid feature value raises BadDataError."""
        columns_to_keep = ["name"]
        people = People(columns_to_keep)

        # Create features that only allow "male" and "female"
        features = FeatureCollection()
        features.add_feature("gender", "male", FeatureValueCounts(min=1, max=5))
        features.add_feature("gender", "female", FeatureValueCounts(min=1, max=5))

        # Try to add person with invalid gender
        person_data = StrippedDict({
            "id": "123",
            "name": "John Doe",
            "gender": "other",  # Not in allowed values
        })

        with pytest.raises(errors.BadDataError, match="has value 'other' not in feature gender"):
            people.add("123", person_data, features)

    def test_people_remove_person(self):
        """Test removing a person from the People collection."""
        columns_to_keep = ["name"]
        people = People(columns_to_keep)

        features = FeatureCollection()
        features.add_feature("gender", "male", FeatureValueCounts(min=1, max=5))

        person_data = StrippedDict({"id": "123", "name": "John", "gender": "male"})
        people.add("123", person_data, features)

        assert people.count == 1

        people.remove("123")

        assert people.count == 0
        assert "123" not in people

    def test_people_iteration(self):
        """Test iterating over people keys."""
        columns_to_keep = ["name"]
        people = People(columns_to_keep)

        features = FeatureCollection()
        features.add_feature("gender", "male", FeatureValueCounts(min=1, max=5))

        # Add multiple people
        for i, name in enumerate(["John", "Jane", "Bob"]):
            person_data = StrippedDict({"id": str(i), "name": name, "gender": "male"})
            people.add(str(i), person_data, features)

        # Test iteration
        person_keys = list(people)
        assert len(person_keys) == 3
        assert set(person_keys) == {"0", "1", "2"}


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


class TestHelperFunctions:
    """Test helper functions in people.py."""

    def test_check_columns_exist_or_multiple_valid(self):
        """Test _check_columns_exist_or_multiple with valid columns."""
        people_head = ["id", "name", "gender", "age", "email"]
        columns = ["name", "email"]

        # Should not raise any exception
        _check_columns_exist_or_multiple(people_head, columns, "test")

    def test_check_columns_exist_or_multiple_missing_column(self):
        """Test _check_columns_exist_or_multiple with missing column."""
        people_head = ["id", "name", "gender"]
        columns = ["name", "email"]  # email is missing

        with pytest.raises(errors.BadDataError, match="No 'email' column test found"):
            _check_columns_exist_or_multiple(people_head, columns, "test")

    def test_check_columns_exist_or_multiple_duplicate_column(self):
        """Test _check_columns_exist_or_multiple with duplicate column."""
        people_head = ["id", "name", "name", "gender"]  # name appears twice
        columns = ["name"]

        with pytest.raises(errors.BadDataError, match="MORE THAN 1 'name' column test found"):
            _check_columns_exist_or_multiple(people_head, columns, "test")

    def test_ensure_settings_keep_address_columns(self):
        """Test _ensure_settings_keep_address_columns adds missing columns."""
        settings = Settings(
            id_column="id",
            columns_to_keep=["name", "email"],
            check_same_address=True,
            check_same_address_columns=["address1", "postcode"],
            max_attempts=100,
            selection_algorithm="legacy",
            random_number_seed=0,
            json_file_path=Path("/path/to/test.json"),
        )

        _ensure_settings_keep_address_columns(settings)

        assert "address1" in settings.columns_to_keep
        assert "postcode" in settings.columns_to_keep
        assert "name" in settings.columns_to_keep
        assert "email" in settings.columns_to_keep

    def test_ensure_settings_keep_address_columns_no_duplicates(self):
        """Test _ensure_settings_keep_address_columns doesn't create duplicates."""
        settings = Settings(
            id_column="id",
            columns_to_keep=["name", "address1", "email"],  # address1 already present
            check_same_address=True,
            check_same_address_columns=["address1", "postcode"],
            max_attempts=100,
            selection_algorithm="legacy",
            random_number_seed=0,
            json_file_path=Path("/path/to/test.json"),
        )

        _ensure_settings_keep_address_columns(settings)

        # address1 should only appear once
        assert settings.columns_to_keep.count("address1") == 1
        assert "postcode" in settings.columns_to_keep


class TestReadInPeople:
    """Test the read_in_people function."""

    def create_test_data(self):
        """Helper to create test data for read_in_people."""
        features = FeatureCollection()
        features.add_feature("gender", "male", FeatureValueCounts(min=1, max=5))
        features.add_feature("gender", "female", FeatureValueCounts(min=1, max=5))
        features.add_feature("age", "young", FeatureValueCounts(min=1, max=3))
        features.add_feature("age", "old", FeatureValueCounts(min=1, max=3))

        settings = Settings(
            id_column="id",
            columns_to_keep=["name", "email"],
            check_same_address=False,
            check_same_address_columns=[],
            max_attempts=100,
            selection_algorithm="legacy",
            random_number_seed=0,
            json_file_path=Path("/path/to/test.json"),
        )

        people_head = ["id", "name", "email", "gender", "age"]
        people_body: list[dict[str, str | int]] = [
            {
                "id": "1",
                "name": "John Doe",
                "email": "john@example.com",
                "gender": "male",
                "age": "young",
            },
            {
                "id": "2",
                "name": "Jane Smith",
                "email": "jane@example.com",
                "gender": "female",
                "age": "old",
            },
            {
                "id": "3",
                "name": "Bob Johnson",
                "email": "bob@example.com",
                "gender": "male",
                "age": "old",
            },
        ]

        return features, settings, people_head, people_body

    def test_read_in_people_happy_path(self):
        """Test read_in_people with valid data."""
        features, settings, people_head, people_body = self.create_test_data()

        people, messages = read_in_people(people_head, people_body, features, settings)

        assert people.count == 3
        assert "1" in people
        assert "2" in people
        assert "3" in people

        # Check person data
        john = people.get_person_dict("1")
        assert john["name"] == "John Doe"
        assert john["email"] == "john@example.com"
        assert john["gender"] == "male"
        assert john["age"] == "young"

        assert len(messages) == 0  # No warning messages

    def test_read_in_people_with_blank_id_warning(self):
        """Test read_in_people handles blank IDs with warning."""
        features, settings, people_head, people_body = self.create_test_data()

        # Add a row with blank ID
        people_body.append({
            "id": "",
            "name": "Empty ID",
            "email": "empty@example.com",
            "gender": "male",
            "age": "young",
        })

        people, messages = read_in_people(people_head, people_body, features, settings)

        assert people.count == 3  # Blank ID row should be skipped
        assert len(messages) == 1
        assert "blank cell found in ID column" in messages[0]
        assert "row 3" in messages[0]  # 0-indexed, so row 3 is the 4th row

    def test_read_in_people_with_whitespace_stripping(self):
        """Test that read_in_people strips whitespace from data."""
        features, settings, people_head, people_body = self.create_test_data()

        # Add whitespace to test data
        people_body[0]["id"] = "  1  "
        people_body[0]["name"] = "  John Doe  "
        people_body[0]["gender"] = "  male  "

        people, _ = read_in_people(people_head, people_body, features, settings)

        john = people.get_person_dict("1")
        assert john["name"] == "John Doe"  # Should be stripped
        assert john["gender"] == "male"  # Should be stripped

    def test_read_in_people_invalid_feature_value(self):
        """Test read_in_people with invalid feature value."""
        features, settings, people_head, people_body = self.create_test_data()

        # Change gender to invalid value
        people_body[0]["gender"] = "unknown"

        with pytest.raises(errors.BadDataError, match="has value 'unknown' not in feature gender"):
            read_in_people(people_head, people_body, features, settings)

    def test_read_in_people_missing_id_column(self):
        """Test read_in_people with missing ID column."""
        features, settings, people_head, people_body = self.create_test_data()

        # Remove ID column from header
        people_head.remove("id")

        with pytest.raises(errors.BadDataError, match="No 'id' column \\(unique id\\) found"):
            read_in_people(people_head, people_body, features, settings)

    def test_read_in_people_missing_feature_column(self):
        """Test read_in_people with missing feature column."""
        features, settings, people_head, people_body = self.create_test_data()

        # Remove gender column from header
        people_head.remove("gender")

        with pytest.raises(errors.BadDataError, match="No 'gender' column \\(a feature\\) found"):
            read_in_people(people_head, people_body, features, settings)

    def test_read_in_people_missing_columns_to_keep(self):
        """Test read_in_people with missing columns_to_keep."""
        features, settings, people_head, people_body = self.create_test_data()

        # Remove email column from header
        people_head.remove("email")

        with pytest.raises(errors.BadDataError, match="No 'email' column \\(to keep\\) found"):
            read_in_people(people_head, people_body, features, settings)

    def test_read_in_people_duplicate_id_column(self):
        """Test read_in_people with duplicate ID column."""
        features, settings, people_head, people_body = self.create_test_data()

        # Add duplicate ID column
        people_head.append("id")

        with pytest.raises(errors.BadDataError, match="MORE THAN 1 'id' column \\(unique id\\) found"):
            read_in_people(people_head, people_body, features, settings)

    def test_read_in_people_duplicate_feature_column(self):
        """Test read_in_people with duplicate feature column."""
        features, settings, people_head, people_body = self.create_test_data()

        # Add duplicate gender column
        people_head.append("gender")

        with pytest.raises(
            errors.BadDataError,
            match="MORE THAN 1 'gender' column \\(a feature\\) found",
        ):
            read_in_people(people_head, people_body, features, settings)

    def test_read_in_people_with_address_checking(self):
        """Test read_in_people with address checking columns."""
        features, settings, people_head, people_body = self.create_test_data()

        # Add address checking
        settings.check_same_address_columns = ["address1", "postcode"]
        people_head.extend(["address1", "postcode"])

        for row in people_body:
            row["address1"] = "123 Main St"
            row["postcode"] = "12345"

        people, _ = read_in_people(people_head, people_body, features, settings)

        # Should succeed and address columns should be added to columns_to_keep
        assert "address1" in settings.columns_to_keep
        assert "postcode" in settings.columns_to_keep
        assert people.count == 3

    def test_read_in_people_missing_address_column(self):
        """Test read_in_people with missing address checking column."""
        features, settings, people_head, people_body = self.create_test_data()

        # Set address checking but don't include the columns
        settings.check_same_address_columns = ["address1", "postcode"]

        with pytest.raises(
            errors.BadDataError,
            match="No 'address1' column \\(to check same address\\) found",
        ):
            read_in_people(people_head, people_body, features, settings)

    def test_read_in_people_empty_body(self):
        """Test read_in_people with empty people body."""
        features, settings, people_head, _ = self.create_test_data()

        people, messages = read_in_people(people_head, [], features, settings)

        assert people.count == 0
        assert len(messages) == 0

    def test_read_in_people_with_numeric_ids(self):
        """Test read_in_people with numeric IDs that get converted to strings."""
        features, settings, people_head, people_body = self.create_test_data()

        # Use numeric IDs
        people_body[0]["id"] = 123
        people_body[1]["id"] = 456
        people_body[2]["id"] = 789

        people, _ = read_in_people(people_head, people_body, features, settings)

        assert people.count == 3
        # IDs should be converted to strings
        assert "123" in people
        assert "456" in people
        assert "789" in people
