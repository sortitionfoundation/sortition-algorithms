from pathlib import Path

import pytest

from sortition_algorithms.core import run_stratification
from sortition_algorithms.features import FeatureCollection, FeatureValueCounts
from sortition_algorithms.people import People
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import StrippedDict


def test_run_stratification_basic_success():
    """Test basic successful run of stratification algorithm."""
    # Create People object
    columns_to_keep = ["gender", "age"]
    people = People(columns_to_keep)

    # Create features first
    features = FeatureCollection()
    features.add_feature("gender", "male", FeatureValueCounts(min=1, max=4))
    features.add_feature("gender", "female", FeatureValueCounts(min=1, max=4))
    features.add_feature("age", "young", FeatureValueCounts(min=1, max=4))
    features.add_feature("age", "old", FeatureValueCounts(min=1, max=4))

    # Add people to the collection
    people_data = [
        {"id": "1", "gender": "male", "age": "young"},
        {"id": "2", "gender": "female", "age": "young"},
        {"id": "3", "gender": "male", "age": "old"},
        {"id": "4", "gender": "female", "age": "old"},
        {"id": "5", "gender": "male", "age": "young"},
        {"id": "6", "gender": "female", "age": "old"},
    ]

    for person_data in people_data:
        people.add(person_data["id"], StrippedDict(person_data), features)

    # Create settings
    settings = Settings(
        id_column="id",
        columns_to_keep=columns_to_keep,
        check_same_address=False,
        check_same_address_columns=[],
        max_attempts=10,
        selection_algorithm="maximin",
        random_number_seed=42,
        json_file_path=Path("/dev/null"),
    )

    # Run stratification
    success, committees, output_lines = run_stratification(
        features=features,
        people=people,
        number_people_wanted=4,
        settings=settings,
        test_selection=True,  # Use deterministic selection for testing
        number_selections=1,
    )

    # Check results
    assert success is True
    assert len(committees) == 1
    assert len(committees[0]) == 4
    assert isinstance(committees[0], frozenset)
    assert len(output_lines) > 0
    assert any("SUCCESS" in line for line in output_lines)


def test_run_stratification_infeasible_quotas():
    """Test run_stratification with infeasible quotas."""
    # Create People object
    columns_to_keep = ["gender"]
    people = People(columns_to_keep)

    # Create features where it's impossible to select the desired number
    features = FeatureCollection()
    features.add_feature("gender", "male", FeatureValueCounts(min=1, max=1))
    features.add_feature("gender", "female", FeatureValueCounts(min=1, max=1))

    # Add test data - only 2 people total
    people_data = [
        {"id": "1", "gender": "male"},
        {"id": "2", "gender": "female"},
    ]

    for person_data in people_data:
        people.add(person_data["id"], StrippedDict(person_data), features)

    # Create settings
    settings = Settings(
        id_column="id",
        columns_to_keep=columns_to_keep,
        check_same_address=False,
        check_same_address_columns=[],
        max_attempts=2,
        selection_algorithm="maximin",
        random_number_seed=42,
        json_file_path=Path("/dev/null"),
    )

    # Should raise exception for invalid desired number (can't select 4 from 2 total)
    with pytest.raises(Exception, match="out of the range"):
        run_stratification(
            features=features,
            people=people,
            number_people_wanted=4,  # Impossible: need 1 male + 1 female = 2 max
            settings=settings,
        )


def test_run_stratification_multiple_attempts():
    """Test run_stratification with retry logic."""
    # Create People object
    columns_to_keep = ["gender"]
    people = People(columns_to_keep)

    # Create features with loose constraints that should allow selection
    features = FeatureCollection()
    features.add_feature("gender", "male", FeatureValueCounts(min=1, max=3))
    features.add_feature("gender", "female", FeatureValueCounts(min=1, max=3))

    # Add test data
    people_data = [
        {"id": "1", "gender": "male"},
        {"id": "2", "gender": "female"},
        {"id": "3", "gender": "male"},
        {"id": "4", "gender": "female"},
        {"id": "5", "gender": "male"},
        {"id": "6", "gender": "female"},
    ]

    for person_data in people_data:
        people.add(person_data["id"], StrippedDict(person_data), features)

    # Create settings with limited attempts
    settings = Settings(
        id_column="id",
        columns_to_keep=columns_to_keep,
        check_same_address=False,
        check_same_address_columns=[],
        max_attempts=3,
        selection_algorithm="maximin",
        random_number_seed=42,
        json_file_path=Path("/dev/null"),
    )

    # Run stratification with a feasible request
    success, committees, output_lines = run_stratification(
        features=features,
        people=people,
        number_people_wanted=4,
        settings=settings,
    )

    # Should succeed
    assert success is True
    assert len(committees) == 1
    assert len(committees[0]) == 4

    # Check that it attempted at least one trial
    trial_lines = [line for line in output_lines if "Trial number" in line]
    assert len(trial_lines) >= 1
