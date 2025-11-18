"""ABOUTME: Test helper functions for creating common test objects and scenarios.
ABOUTME: Provides standardized ways to create Settings, People, and FeatureCollection objects.
"""

from pathlib import Path

import tomli_w

from sortition_algorithms.features import FeatureCollection, read_in_features
from sortition_algorithms.people import People, read_in_people
from sortition_algorithms.settings import Settings

test_path = Path(__file__).parent
features_csv_path = test_path / "fixtures/features.csv"
candidates_csv_path = test_path / "fixtures/candidates.csv"


def create_test_settings(
    id_column: str = "id",
    columns_to_keep: list[str] | None = None,
    check_same_address: bool = False,
    check_same_address_columns: list[str] | None = None,
    max_attempts: int = 100,
    selection_algorithm: str = "maximin",
    random_number_seed: int = 42,
    **kwargs,
) -> Settings:
    """Create Settings object with sensible test defaults.

    Args:
        id_column: Column name for person IDs
        columns_to_keep: List of columns to keep in output
        check_same_address: Whether to check for same address
        check_same_address_columns: Columns to use for address checking
        max_attempts: Maximum retry attempts
        selection_algorithm: Selection algorithm to use
        random_number_seed: Random seed for reproducible tests
        **kwargs: Additional arguments passed to Settings constructor

    Returns:
        Settings object configured for testing
    """
    columns_to_keep = columns_to_keep or []
    check_same_address_columns = check_same_address_columns or []

    return Settings(
        id_column=id_column,
        columns_to_keep=columns_to_keep,
        check_same_address=check_same_address,
        check_same_address_columns=check_same_address_columns,
        max_attempts=max_attempts,
        selection_algorithm=selection_algorithm,
        random_number_seed=random_number_seed,
        **kwargs,
    )


def get_settings_for_fixtures(algorithm="legacy"):
    columns_to_keep = [
        "first_name",
        "last_name",
        "mobile_number",
        "email",
        "primary_address1",
        "primary_address2",
        "primary_city",
        "primary_zip",
        "gender",
        "age_bracket",
        "geo_bucket",
        "edu_level",
    ]
    return Settings(
        id_column="nationbuilder_id",
        columns_to_keep=columns_to_keep,
        check_same_address=True,
        check_same_address_columns=["primary_address1", "primary_zip"],
        selection_algorithm=algorithm,
    )


def create_settings_file_for_fixtures(file_path: Path, algorithm="maximin") -> None:
    settings = get_settings_for_fixtures(algorithm)
    with file_path.open("wb") as file:
        tomli_w.dump(settings.as_dict(), file)


def create_simple_features(
    gender_min: int = 1,
    gender_max: int = 5,
    age_min: int = 1,
    age_max: int = 3,
) -> FeatureCollection:
    """Create basic gender/age features (most common test pattern).

    Args:
        gender_min: Minimum for each gender value
        gender_max: Maximum for each gender value
        age_min: Minimum for each age value
        age_max: Maximum for each age value

    Returns:
        FeatureCollection with gender (male/female) and age (young/old) features
    """
    features_data = [
        {
            "feature": "gender",
            "value": "male",
            "min": str(gender_min),
            "max": str(gender_max),
        },
        {
            "feature": "gender",
            "value": "female",
            "min": str(gender_min),
            "max": str(gender_max),
        },
        {"feature": "age", "value": "young", "min": str(age_min), "max": str(age_max)},
        {"feature": "age", "value": "old", "min": str(age_min), "max": str(age_max)},
    ]

    head = ["feature", "value", "min", "max"]
    features, _, _ = read_in_features(head, features_data)
    return features


def create_gender_only_features(min_val: int = 1, max_val: int = 5) -> FeatureCollection:
    """Create features with only gender (male/female).

    Args:
        min_val: Minimum for each gender value
        max_val: Maximum for each gender value

    Returns:
        FeatureCollection with gender feature only
    """
    features_data = [
        {
            "feature": "gender",
            "value": "male",
            "min": str(min_val),
            "max": str(max_val),
        },
        {
            "feature": "gender",
            "value": "female",
            "min": str(min_val),
            "max": str(max_val),
        },
    ]

    head = ["feature", "value", "min", "max"]
    features, _, _ = read_in_features(head, features_data)
    return features


def create_simple_people(
    features: FeatureCollection,
    settings: Settings,
    count: int = 6,
) -> People:
    """Create basic test people with gender/age data.

    Args:
        features: FeatureCollection to validate against
        settings: Settings object for configuration
        count: Number of people to create (default creates balanced set)

    Returns:
        People object with test data
    """
    # Create balanced test data
    people_data = []
    patterns = [
        ("male", "young"),
        ("female", "young"),
        ("male", "old"),
        ("female", "old"),
    ]

    for i in range(count):
        person_id = str(i)  # Start from 0 to match existing test expectations
        gender, age = patterns[i % len(patterns)]
        person_data = {
            "id": person_id,
            "name": f"Person{person_id}",
            "email": f"person{person_id}@example.com",
            "gender": gender,
            "age": age,
        }
        people_data.append(person_data)

    head = ["id", "name", "email", "gender", "age"]
    people, _ = read_in_people(head, people_data, features, settings)
    return people


def create_people_with_addresses(
    features: FeatureCollection,
    settings: Settings,
    include_households: bool = True,
) -> People:
    """Create test people with address data for household testing.

    Args:
        features: FeatureCollection to validate against
        settings: Settings object for configuration
        include_households: Whether to include people at same addresses

    Returns:
        People object with address data for testing same-address logic
    """
    people_data = [
        {
            "id": "0",
            "name": "John",
            "email": "john@example.com",
            "gender": "male",
            "age": "young",
            "address1": "123 Main St",
            "postcode": "12345",
        },
        {
            "id": "1",
            "name": "Jane",
            "email": "jane@example.com",
            "gender": "female",
            "age": "young",
            "address1": "123 Main St" if include_households else "456 Oak Ave",
            "postcode": "12345" if include_households else "67890",
        },
        {
            "id": "2",
            "name": "Bob",
            "email": "bob@example.com",
            "gender": "male",
            "age": "old",
            "address1": "456 Oak Ave",
            "postcode": "67890",
        },
        {
            "id": "3",
            "name": "Alice",
            "email": "alice@example.com",
            "gender": "female",
            "age": "old",
            "address1": "789 Pine Rd",
            "postcode": "11111",
        },
    ]

    head = ["id", "name", "email", "gender", "age", "address1", "postcode"]
    people, _ = read_in_people(head, people_data, features, settings)
    return people


def create_people_with_complex_households(
    features: FeatureCollection,
    settings: Settings,
) -> People:
    """Create test people with complex household patterns for advanced testing.

    Creates 5 people with specific household arrangements:
    - John, Jane, Carol all at same address
    - Bob and Alice at different addresses

    Args:
        features: FeatureCollection to validate against
        settings: Settings object for configuration

    Returns:
        People object with complex household data
    """
    people_data = [
        {
            "id": "0",
            "name": "John",
            "email": "john@example.com",
            "gender": "male",
            "age": "young",
            "address1": "123 Main St",
            "address2": "12345",
        },
        {
            "id": "1",
            "name": "Jane",
            "email": "jane@example.com",
            "gender": "female",
            "age": "young",
            "address1": "123 Main St",
            "address2": "12345",
        },
        {
            "id": "2",
            "name": "Bob",
            "email": "bob@example.com",
            "gender": "male",
            "age": "old",
            "address1": "456 Oak Ave",
            "address2": "67890",
        },
        {
            "id": "3",
            "name": "Alice",
            "email": "alice@example.com",
            "gender": "female",
            "age": "old",
            "address1": "789 Pine Rd",
            "address2": "11111",
        },
        {
            "id": "4",
            "name": "Carol",
            "email": "carol@example.com",
            "gender": "female",
            "age": "old",
            "address1": "123 Main St",  # Same as John/Jane
            "address2": "12345",
        },
    ]

    head = ["id", "name", "email", "gender", "age", "address1", "address2"]
    people, _ = read_in_people(head, people_data, features, settings)
    return people


def create_people_with_legacy_addresses(
    features: FeatureCollection,
    settings: Settings,
) -> People:
    """Create test people for legacy address testing with specific requirements.

    Creates 5 people matching the legacy test pattern:
    - John, Jane, Carol all at 123 Main St / 12345
    - Bob at 456 Oak Ave / 67890
    - Alice at 789 Pine Rd / 11111

    Args:
        features: FeatureCollection to validate against
        settings: Settings object for configuration

    Returns:
        People object with address data for legacy testing
    """
    people_data = [
        {
            "id": "0",
            "name": "John",
            "gender": "male",
            "address1": "123 Main St",
            "address2": "12345",
        },
        {
            "id": "1",
            "name": "Jane",
            "gender": "female",
            "address1": "123 Main St",
            "address2": "12345",
        },
        {
            "id": "2",
            "name": "Bob",
            "gender": "male",
            "address1": "456 Oak Ave",
            "address2": "67890",
        },
        {
            "id": "3",
            "name": "Alice",
            "gender": "female",
            "address1": "789 Pine Rd",
            "address2": "11111",
        },
        {
            "id": "4",
            "name": "Carol",
            "gender": "female",
            "address1": "123 Main St",
            "address2": "12345",
        },
    ]

    head = ["id", "name", "gender", "address1", "address2"]
    people, _ = read_in_people(head, people_data, features, settings)
    return people


def create_test_scenario(
    include_addresses: bool = False,
    people_count: int = 6,
    check_same_address: bool = False,
    **settings_kwargs,
) -> tuple[FeatureCollection, People, Settings]:
    """One-stop function to create coordinated test objects.

    Args:
        include_addresses: Whether to include address data in people
        people_count: Number of people to create (ignored if include_addresses=True)
        check_same_address: Whether settings should check for same addresses
        **settings_kwargs: Additional arguments for Settings creation

    Returns:
        Tuple of (features, people, settings) ready for testing
    """
    # Create features
    features = create_simple_features()

    # Create settings with appropriate columns
    if include_addresses:
        columns_to_keep = ["name", "email", "address1", "postcode"]
        check_same_address_columns = ["address1", "postcode"] if check_same_address else []
    else:
        columns_to_keep = ["name", "email"]
        check_same_address_columns = []

    settings = create_test_settings(
        columns_to_keep=columns_to_keep,
        check_same_address=check_same_address,
        check_same_address_columns=check_same_address_columns,
        **settings_kwargs,
    )

    # Create people
    if include_addresses:
        people = create_people_with_addresses(features, settings, include_households=check_same_address)
    else:
        people = create_simple_people(features, settings, count=people_count)

    return features, people, settings


def create_minimal_test_objects() -> tuple[FeatureCollection, People, Settings]:
    """Create minimal test objects for basic functionality testing.

    Returns:
        Tuple of (features, people, settings) with minimal viable configuration
    """
    features = create_gender_only_features(min_val=1, max_val=3)
    settings = create_test_settings(columns_to_keep=["name"])
    people = create_simple_people(features, settings, count=4)
    return features, people, settings
