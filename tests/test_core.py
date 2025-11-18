import pytest

from sortition_algorithms.core import run_stratification
from tests.helpers import create_gender_only_features, create_simple_people, create_test_scenario, create_test_settings


@pytest.mark.slow
def test_run_stratification_basic_success():
    """Test basic successful run of stratification algorithm."""
    # Create test scenario with coordinated objects
    features, people, settings = create_test_scenario(
        people_count=6,
        max_attempts=10,
        selection_algorithm="maximin",
    )

    # Run stratification
    success, committees, report = run_stratification(
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
    assert "SUCCESS" in report.as_text()


def test_run_stratification_infeasible_quotas():
    """Test run_stratification with infeasible quotas."""
    # Create features where it's impossible to select the desired number
    features = create_gender_only_features(min_val=1, max_val=1)
    settings = create_test_settings(columns_to_keep=["name"])
    people = create_simple_people(features, settings, count=2)

    # Should raise exception for invalid desired number (can't select 4 from 2 total)
    with pytest.raises(Exception, match="out of the range"):
        run_stratification(
            features=features,
            people=people,
            number_people_wanted=4,  # Impossible: need 1 male + 1 female = 2 max
            settings=settings,
        )


@pytest.mark.slow
def test_run_stratification_multiple_attempts():
    """Test run_stratification with retry logic."""
    # Create test scenario that should succeed
    features, people, settings = create_test_scenario(
        people_count=6,
        max_attempts=3,
        selection_algorithm="maximin",
    )

    # Run stratification with a feasible request
    success, committees, report = run_stratification(
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
    assert "Trial number" in report.as_text()
