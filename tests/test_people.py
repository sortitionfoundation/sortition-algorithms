import pytest

from sortition_algorithms import errors
from sortition_algorithms.features import FeatureCollection, read_in_features
from sortition_algorithms.people import (
    People,
    _check_columns_exist_or_multiple,
    check_for_duplicate_people,
    read_in_people,
)
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import StrippedDict


def create_simple_test_features() -> FeatureCollection:
    """Create a simple FeatureCollection with only gender for testing."""
    features_data = [
        {"feature": "gender", "value": "male", "min": "1", "max": "10"},
        {"feature": "gender", "value": "female", "min": "1", "max": "10"},
    ]
    head = ["feature", "value", "min", "max"]
    features, _, _ = read_in_features(head, features_data)
    return features


def create_test_features() -> FeatureCollection:
    """Create a FeatureCollection with gender and age for testing."""
    features_data = [
        {"feature": "gender", "value": "male", "min": "1", "max": "5"},
        {"feature": "gender", "value": "female", "min": "1", "max": "5"},
        {"feature": "age", "value": "young", "min": "1", "max": "3"},
        {"feature": "age", "value": "old", "min": "1", "max": "3"},
    ]
    head = ["feature", "value", "min", "max"]
    features, _, _ = read_in_features(head, features_data)
    return features


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
        features = create_simple_test_features()

        # Create person data
        person_data = StrippedDict({
            "id": "123",
            "name": "John Doe",
            "email": "john@example.com",
            "gender": "male",
        })

        people.add("123", person_data, features, 1)

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
        features = create_simple_test_features()

        # Try to add person with invalid gender
        person_data = StrippedDict({
            "id": "123",
            "name": "John Doe",
            "gender": "other",  # Not in allowed values
        })

        with pytest.raises(errors.ParseTableMultiError, match="Value 'other' not in feature gender"):
            people.add("123", person_data, features, 1)

    def test_people_remove_person(self):
        """Test removing a person from the People collection."""
        columns_to_keep = ["name"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        person_data = StrippedDict({"id": "123", "name": "John", "gender": "male"})
        people.add("123", person_data, features, 1)

        assert people.count == 1

        people.remove("123")

        assert people.count == 0
        assert "123" not in people

    def test_people_iteration(self):
        """Test iterating over people keys."""
        columns_to_keep = ["name"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        # Add multiple people
        for i, name in enumerate(["John", "Jane", "Bob"]):
            person_data = StrippedDict({"id": str(i), "name": name, "gender": "male"})
            people.add(str(i), person_data, features, i)

        # Test iteration
        person_keys = list(people)
        assert len(person_keys) == 3
        assert set(person_keys) == {"0", "1", "2"}


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

    def test_full_columns_to_keep_includes_address_columns(self):
        """Test _ensure_settings_keep_address_columns adds missing columns."""
        settings = Settings(
            id_column="id",
            columns_to_keep=["name", "email"],
            check_same_address=True,
            check_same_address_columns=["address1", "postcode"],
        )

        assert "address1" in settings.full_columns_to_keep
        assert "postcode" in settings.full_columns_to_keep
        assert "name" in settings.full_columns_to_keep
        assert "email" in settings.full_columns_to_keep

    def test_full_columns_to_keep_has_no_duplicates(self):
        """Test _ensure_settings_keep_address_columns doesn't create duplicates."""
        settings = Settings(
            id_column="id",
            columns_to_keep=["name", "address1", "email"],  # address1 already present
            check_same_address=True,
            check_same_address_columns=["address1", "postcode"],
        )

        # address1 should only appear once
        assert settings.full_columns_to_keep.count("address1") == 1
        assert "postcode" in settings.full_columns_to_keep


class TestReadInPeople:
    """Test the read_in_people function."""

    def create_test_data(self):
        """Helper to create test data for read_in_people."""
        features = create_test_features()

        settings = Settings(
            id_column="id",
            columns_to_keep=["name", "email"],
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

        people, report = read_in_people(people_head, people_body, features, settings)

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

        assert report.as_text() == ""  # No warning messages

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

        people, report = read_in_people(people_head, people_body, features, settings)

        assert people.count == 3  # Blank ID row should be skipped
        report_text = report.as_text()
        assert "blank cell found in ID column" in report_text
        assert "row 5" in report_text  # The 1st row is the header, so the 4th data row is row 5

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

        with pytest.raises(errors.ParseTableMultiError, match="Value 'unknown' not in feature gender"):
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
        assert "address1" in settings.full_columns_to_keep
        assert "postcode" in settings.full_columns_to_keep
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

        people, report = read_in_people(people_head, [], features, settings)

        assert people.count == 0
        assert report.as_text() == ""

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


class TestPeopleHouseholds:
    """Test the households() method of the People class."""

    def test_households_empty_people(self):
        """Test households method with no people."""
        columns_to_keep = ["name"]
        people = People(columns_to_keep)

        households = people.households(["address1", "postcode"])

        assert households == {}

    def test_households_single_person(self):
        """Test households method with a single person."""
        columns_to_keep = ["name", "address1", "postcode"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        person_data = StrippedDict({
            "id": "1",
            "name": "John Doe",
            "address1": "123 Main St",
            "postcode": "12345",
            "gender": "male",
        })
        people.add("1", person_data, features, 1)

        households = people.households(["address1", "postcode"])

        expected = {("123 Main St", "12345"): ["1"]}
        assert households == expected

    def test_households_multiple_people_same_address(self):
        """Test households method with multiple people at the same address."""
        columns_to_keep = ["name", "address1", "postcode"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        # Add people at the same address
        people_data = [
            ("1", "John Doe", "123 Main St", "12345", "male"),
            ("2", "Jane Doe", "123 Main St", "12345", "female"),
            ("3", "Bob Doe", "123 Main St", "12345", "male"),
        ]

        for person_id, name, address1, postcode, gender in people_data:
            person_data = StrippedDict({
                "id": person_id,
                "name": name,
                "address1": address1,
                "postcode": postcode,
                "gender": gender,
            })
            people.add(person_id, person_data, features, 1)

        households = people.households(["address1", "postcode"])

        expected = {("123 Main St", "12345"): ["1", "2", "3"]}
        assert households == expected

    def test_households_multiple_people_different_addresses(self):
        """Test households method with people at different addresses."""
        columns_to_keep = ["name", "address1", "postcode"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        # Add people at different addresses
        people_data = [
            ("1", "John Doe", "123 Main St", "12345", "male"),
            ("2", "Jane Smith", "456 Oak Ave", "67890", "female"),
            ("3", "Bob Johnson", "789 Pine Rd", "54321", "male"),
        ]

        for person_id, name, address1, postcode, gender in people_data:
            person_data = StrippedDict({
                "id": person_id,
                "name": name,
                "address1": address1,
                "postcode": postcode,
                "gender": gender,
            })
            people.add(person_id, person_data, features, 1)

        households = people.households(["address1", "postcode"])

        expected = {
            ("123 Main St", "12345"): ["1"],
            ("456 Oak Ave", "67890"): ["2"],
            ("789 Pine Rd", "54321"): ["3"],
        }
        assert households == expected

    def test_households_mixed_same_and_different_addresses(self):
        """Test households method with mix of same and different addresses."""
        columns_to_keep = ["name", "address1", "postcode"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        # Add people with some sharing addresses
        people_data = [
            ("1", "John Doe", "123 Main St", "12345", "male"),
            ("2", "Jane Doe", "123 Main St", "12345", "female"),  # Same as John
            ("3", "Bob Smith", "456 Oak Ave", "67890", "male"),
            ("4", "Alice Smith", "456 Oak Ave", "67890", "female"),  # Same as Bob
            ("5", "Charlie Johnson", "789 Pine Rd", "54321", "male"),  # Unique address
        ]

        for person_id, name, address1, postcode, gender in people_data:
            person_data = StrippedDict({
                "id": person_id,
                "name": name,
                "address1": address1,
                "postcode": postcode,
                "gender": gender,
            })
            people.add(person_id, person_data, features, 1)

        households = people.households(["address1", "postcode"])

        expected = {
            ("123 Main St", "12345"): ["1", "2"],
            ("456 Oak Ave", "67890"): ["3", "4"],
            ("789 Pine Rd", "54321"): ["5"],
        }
        assert households == expected

    def test_households_single_address_column(self):
        """Test households method with only one address column."""
        columns_to_keep = ["name", "address1"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        people_data = [
            ("1", "John Doe", "123 Main St", "male"),
            ("2", "Jane Doe", "123 Main St", "female"),  # Same address
            ("3", "Bob Smith", "456 Oak Ave", "male"),  # Different address
        ]

        for person_id, name, address1, gender in people_data:
            person_data = StrippedDict({
                "id": person_id,
                "name": name,
                "address1": address1,
                "gender": gender,
            })
            people.add(person_id, person_data, features, 1)

        households = people.households(["address1"])

        expected = {
            ("123 Main St",): ["1", "2"],
            ("456 Oak Ave",): ["3"],
        }
        assert households == expected

    def test_households_with_whitespace_in_addresses(self):
        """Test households method handles whitespace in addresses correctly."""
        columns_to_keep = ["name", "address1", "postcode"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        # Note: StrippedDict will strip whitespace, so these should be considered the same
        people_data = [
            ("1", "John Doe", "  123 Main St  ", "  12345  ", "male"),
            ("2", "Jane Doe", "123 Main St", "12345", "male"),
        ]

        for person_id, name, address1, postcode, gender in people_data:
            person_data = StrippedDict({
                "id": person_id,
                "name": name,
                "address1": address1,
                "postcode": postcode,
                "gender": gender,
            })
            people.add(person_id, person_data, features, 1)

        households = people.households(["address1", "postcode"])

        # Both should end up in the same household due to whitespace stripping
        expected = {("123 Main St", "12345"): ["1", "2"]}
        assert households == expected

    def test_households_empty_address_columns_list(self):
        """Test households method with empty address columns list."""
        columns_to_keep = ["name"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        person_data = StrippedDict({
            "id": "1",
            "name": "John Doe",
            "gender": "male",
        })
        people.add("1", person_data, features, 1)

        households = people.households([])

        # Empty tuple should group everyone together
        expected = {(): ["1"]}
        assert households == expected


class TestPeopleMatchingAddress:
    """Test the matching_address() method of the People class."""

    def test_matching_address_no_matches(self):
        """Test matching_address when no one else has the same address."""
        columns_to_keep = ["name", "address1", "postcode"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        # Add people with different addresses
        people_data = [
            ("1", "John Doe", "123 Main St", "12345", "male"),
            ("2", "Jane Smith", "456 Oak Ave", "67890", "female"),
            ("3", "Bob Johnson", "789 Pine Rd", "54321", "male"),
        ]

        for person_id, name, address1, postcode, gender in people_data:
            person_data = StrippedDict({
                "id": person_id,
                "name": name,
                "address1": address1,
                "postcode": postcode,
                "gender": gender,
            })
            people.add(person_id, person_data, features, 1)

        # John should have no matches
        matches = list(people.matching_address("1", ["address1", "postcode"]))
        assert matches == []

    def test_matching_address_with_matches(self):
        """Test matching_address when other people have the same address."""
        columns_to_keep = ["name", "address1", "postcode"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        # Add people with some sharing addresses
        people_data = [
            ("1", "John Doe", "123 Main St", "12345", "male"),
            ("2", "Jane Doe", "123 Main St", "12345", "female"),  # Same as John
            ("3", "Bob Doe", "123 Main St", "12345", "male"),  # Same as John
            ("4", "Alice Smith", "456 Oak Ave", "67890", "female"),  # Different address
        ]

        for person_id, name, address1, postcode, gender in people_data:
            person_data = StrippedDict({
                "id": person_id,
                "name": name,
                "address1": address1,
                "postcode": postcode,
                "gender": gender,
            })
            people.add(person_id, person_data, features, 1)

        # John should match Jane and Bob
        matches = list(people.matching_address("1", ["address1", "postcode"]))
        assert set(matches) == {"2", "3"}

        # Jane should match John and Bob
        matches = list(people.matching_address("2", ["address1", "postcode"]))
        assert set(matches) == {"1", "3"}

        # Alice should have no matches
        matches = list(people.matching_address("4", ["address1", "postcode"]))
        assert matches == []

    def test_matching_address_excludes_self(self):
        """Test that matching_address excludes the person being queried."""
        columns_to_keep = ["name", "address1", "postcode"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        # Add person
        person_data = StrippedDict({
            "id": "1",
            "name": "John Doe",
            "address1": "123 Main St",
            "postcode": "12345",
            "gender": "male",
        })
        people.add("1", person_data, features, 1)

        # Should not match himself
        matches = list(people.matching_address("1", ["address1", "postcode"]))
        assert matches == []

    def test_matching_address_single_address_column(self):
        """Test matching_address with only one address column."""
        columns_to_keep = ["name", "address1"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        people_data = [
            ("1", "John Doe", "123 Main St", "male"),
            ("2", "Jane Doe", "123 Main St", "female"),  # Same address
            ("3", "Bob Smith", "456 Oak Ave", "male"),  # Different address
        ]

        for person_id, name, address1, gender in people_data:
            person_data = StrippedDict({
                "id": person_id,
                "name": name,
                "address1": address1,
                "gender": gender,
            })
            people.add(person_id, person_data, features, 1)

        # John should match Jane
        matches = list(people.matching_address("1", ["address1"]))
        assert matches == ["2"]

        # Bob should have no matches
        matches = list(people.matching_address("3", ["address1"]))
        assert matches == []

    def test_matching_address_with_whitespace(self):
        """Test matching_address handles whitespace correctly."""
        columns_to_keep = ["name", "address1", "postcode"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        # Addresses with different whitespace should still match
        people_data = [
            ("1", "John Doe", "  123 Main St  ", "  12345  ", "male"),
            ("2", "Jane Doe", "123 Main St", "12345", "male"),
        ]

        for person_id, name, address1, postcode, gender in people_data:
            person_data = StrippedDict({
                "id": person_id,
                "name": name,
                "address1": address1,
                "postcode": postcode,
                "gender": gender,
            })
            people.add(person_id, person_data, features, 1)

        # Should match due to whitespace stripping
        matches = list(people.matching_address("1", ["address1", "postcode"]))
        assert matches == ["2"]

    def test_matching_address_empty_address_columns(self):
        """Test matching_address with empty address columns list."""
        columns_to_keep = ["name"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        people_data = [
            ("1", "John Doe", "male"),
            ("2", "Jane Doe", "female"),
            ("3", "Bob Smith", "male"),
        ]

        for person_id, name, gender in people_data:
            person_data = StrippedDict({
                "id": person_id,
                "name": name,
                "gender": gender,
            })
            people.add(person_id, person_data, features, 1)

        # With empty address columns, everyone should match everyone else
        matches = list(people.matching_address("1", []))
        assert set(matches) == {"2", "3"}

        matches = list(people.matching_address("2", []))
        assert set(matches) == {"1", "3"}

    def test_matching_address_partial_address_match(self):
        """Test that partial address matches don't count as matches."""
        columns_to_keep = ["name", "address1", "postcode"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        people_data = [
            ("1", "John Doe", "123 Main St", "12345", "male"),
            (
                "2",
                "Jane Doe",
                "123 Main St",
                "67890",
                "female",
            ),  # Same street, different postcode
            (
                "3",
                "Bob Smith",
                "456 Oak Ave",
                "12345",
                "male",
            ),  # Different street, same postcode
        ]

        for person_id, name, address1, postcode, gender in people_data:
            person_data = StrippedDict({
                "id": person_id,
                "name": name,
                "address1": address1,
                "postcode": postcode,
                "gender": gender,
            })
            people.add(person_id, person_data, features, 1)

        # John should have no matches (both address1 AND postcode must match)
        matches = list(people.matching_address("1", ["address1", "postcode"]))
        assert matches == []

    def test_matching_address_case_sensitivity(self):
        """Test that matching_address is case sensitive (or handles case as per StrippedDict)."""
        columns_to_keep = ["name", "address1", "postcode"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        people_data = [
            ("1", "John Doe", "123 Main St", "12345", "male"),
            ("2", "Jane Doe", "123 MAIN ST", "12345", "male"),  # Different case
        ]

        for person_id, name, address1, postcode, gender in people_data:
            person_data = StrippedDict({
                "id": person_id,
                "name": name,
                "address1": address1,
                "postcode": postcode,
                "gender": gender,
            })
            people.add(person_id, person_data, features, 1)

        # Should NOT match due to case difference (StrippedDict only strips, doesn't normalize case)
        matches = list(people.matching_address("1", ["address1", "postcode"]))
        assert matches == []

    def test_matching_address_nonexistent_person(self):
        """Test matching_address raises KeyError for non-existent person."""
        columns_to_keep = ["name", "address1"]
        people = People(columns_to_keep)

        with pytest.raises(KeyError):
            list(people.matching_address("nonexistent", ["address1"]))

    def test_matching_address_large_household(self):
        """Test matching_address with a large household."""
        columns_to_keep = ["name", "address1", "postcode"]
        people = People(columns_to_keep)

        features = create_simple_test_features()

        # Add 10 people at the same address
        for i in range(10):
            person_data = StrippedDict({
                "id": str(i),
                "name": f"Person {i}",
                "address1": "123 Main St",
                "postcode": "12345",
                "gender": "male" if i % 2 == 0 else "female",
            })
            people.add(str(i), person_data, features, 1)

        # Person 0 should match all others (1-9)
        matches = list(people.matching_address("0", ["address1", "postcode"]))
        expected_matches = [str(i) for i in range(1, 10)]
        assert set(matches) == set(expected_matches)

        # Person 5 should match all others except themselves
        matches = list(people.matching_address("5", ["address1", "postcode"]))
        expected_matches = [str(i) for i in range(10) if i != 5]
        assert set(matches) == set(expected_matches)

    def test_check_for_duplicate_people_with_no_dupes(self):
        settings = Settings(
            id_column="id",
            columns_to_keep=["name", "email"],
        )

        people_body = [
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
        stripped_people_body = [StrippedDict(row) for row in people_body]
        assert check_for_duplicate_people(stripped_people_body, settings) == []

    def test_check_for_duplicate_people_with_exact_dupes(self):
        """
        When there are duplicate rows with exactly the same data,
        there should be messages to be logged, but nothing raised.
        """
        settings = Settings(
            id_column="id",
            columns_to_keep=["name", "email"],
        )

        people_body = [
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
        stripped_people_body = [StrippedDict(row) for row in people_body]
        combined_messages = " ".join(check_for_duplicate_people(stripped_people_body, settings)).lower()
        assert "found 1 ids that have more than one row" in combined_messages
        assert "duplicated ids are: 2" in combined_messages
        assert "all duplicate rows have identical data" in combined_messages

    def test_check_for_duplicate_people_with_dupes_with_mismatching_data(self):
        """
        When there are duplicate rows with data that does not match then
        an error should be raised.
        """
        settings = Settings(
            id_column="id",
            columns_to_keep=["name", "email"],
        )

        people_body = [
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
            {
                "id": "3",
                "name": "Bob J. Johnson",
                "email": "bob42@example.com",
                "gender": "male",
                "age": "old",
            },
        ]
        stripped_people_body = [StrippedDict(row) for row in people_body]
        with pytest.raises(errors.SelectionMultilineError) as exc_context:
            check_for_duplicate_people(stripped_people_body, settings)
        combined_messages = str(exc_context.value).lower()

        assert "found 2 ids that have more than one row" in combined_messages
        assert "duplicated ids are: 2 3" in combined_messages
        assert "for id '3' one row of data is" in combined_messages
        assert "bob@example.com" in combined_messages
        assert "bob42@example.com" in combined_messages

        assert "jane@example.com" not in combined_messages
