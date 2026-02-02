# ABOUTME: Generate scaled test fixtures for benchmarking.
# ABOUTME: Creates people pools of various sizes while maintaining demographic proportions.
"""
Generate scaled test fixtures for profiling.

The base fixture has 150 people. This module can generate larger pools
by scaling up while maintaining similar demographic distributions.

Usage:
    uv run python -m benchmarks.generate_fixtures
"""

import copy
import csv
import random
from pathlib import Path

from sortition_algorithms.features import FeatureCollection, read_in_features
from sortition_algorithms.people import People, read_in_people
from sortition_algorithms.settings import Settings

# Paths to base fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
BASE_FEATURES_PATH = FIXTURES_DIR / "features.csv"
BASE_CANDIDATES_PATH = FIXTURES_DIR / "candidates.csv"


def get_base_settings() -> Settings:
    """Get settings configured for the base fixtures."""
    return Settings(
        id_column="nationbuilder_id",
        columns_to_keep=[
            "first_name",
            "last_name",
            "email",
            "mobile_number",
            "primary_address1",
            "primary_address2",
            "primary_city",
            "primary_zip",
        ],
        check_same_address=False,
        check_same_address_columns=[],
    )


def load_base_fixtures() -> tuple[FeatureCollection, People]:
    """Load the base 150-person fixtures.

    Returns:
        tuple of (features, people)
    """
    settings = get_base_settings()

    with open(BASE_FEATURES_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        head = reader.fieldnames or []
        rows = list(reader)
    features, _, _ = read_in_features(head, rows)

    with open(BASE_CANDIDATES_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        head = reader.fieldnames or []
        rows = list(reader)
    people, _ = read_in_people(head, rows, features, settings)

    return features, people


def generate_scaled_fixtures(
    target_size: int,
    seed: int = 42,
    panel_size: int | None = None,
    tight_constraints: bool = False,
) -> tuple[FeatureCollection, People]:
    """Generate a scaled fixture with the target number of people.

    Maintains similar demographic proportions to the base fixture by
    sampling from each feature-value combination proportionally.

    Args:
        target_size: Target number of people
        seed: Random seed for reproducibility
        panel_size: Target panel size (defaults to ~15% of pool)
        tight_constraints: If True, use narrow min/max gaps to stress solvers

    Returns:
        tuple of (features, people)
    """
    random.seed(seed)

    # Load base fixtures
    features, base_people = load_base_fixtures()
    settings = get_base_settings()

    # Generate new people data
    new_people_data: list[dict[str, str]] = []
    header = [
        "nationbuilder_id",
        "first_name",
        "last_name",
        "email",
        "mobile_number",
        "primary_address1",
        "primary_address2",
        "primary_city",
        "primary_zip",
    ]

    # Add feature columns
    for feature_name in features:
        header.append(feature_name)

    # Build a mapping of base person_id -> their data
    base_data = {pid: base_people.get_person_dict(pid) for pid in base_people}

    # Generate scaled number of people for each combination
    for i in range(target_size):
        # Pick a random base person to use as template
        template_id = random.choice(list(base_people))
        template_data = base_data[template_id]

        # Create new person with unique ID and slightly varied data
        new_id = f"gen_{i}"
        new_person = {
            "nationbuilder_id": new_id,
            "first_name": f"First{i}",
            "last_name": f"Last{i}",
            "email": f"person{i}@example.com",
            "mobile_number": f"+1-555-{i:04d}",
            "primary_address1": f"{i} Generated Street",
            "primary_address2": "",
            "primary_city": f"City{i % 10}",
            "primary_zip": f"{10000 + i}",
        }

        # Copy feature values from template
        for feature_name in features:
            new_person[feature_name] = template_data[feature_name]

        new_people_data.append(new_person)

    # Create People object from generated data
    people, _ = read_in_people(header, new_people_data, features, settings)

    # Determine panel size (default to ~15% of pool)
    if panel_size is None:
        panel_size = max(22, int(target_size * 0.15))

    # Scale the feature quotas for the panel size
    scaled_features = scale_features_for_panel(features, base_people, people, panel_size, tight_constraints)

    return scaled_features, people


def scale_features_for_panel(
    base_features: FeatureCollection,
    base_people: People,
    new_people: People,
    panel_size: int,
    tight_constraints: bool = False,
) -> FeatureCollection:
    """Scale feature quotas for a specific panel size based on population distribution.

    Args:
        base_features: Original FeatureCollection
        base_people: Original People pool
        new_people: New scaled People pool
        panel_size: Target panel size
        tight_constraints: If True, use narrow min/max gaps (harder for solvers)

    Returns:
        New FeatureCollection with scaled quotas
    """
    scaled = copy.deepcopy(base_features)
    pool_size = new_people.count

    for feature_name in scaled:
        for value_name in scaled[feature_name]:
            fv = scaled[feature_name][value_name]

            # Count how many people have this feature value
            count = sum(
                1 for pid in new_people if new_people.get_person_dict(pid)[feature_name].lower() == value_name.lower()
            )

            # Calculate proportion in the pool
            proportion = count / pool_size if pool_size > 0 else 0

            # Target number in the panel based on proportion
            target = round(proportion * panel_size)

            if tight_constraints:
                # Tight constraints: min and max are very close (or equal)
                # This forces the solver to find exact solutions
                fv.min = max(0, target)
                fv.max = max(fv.min, target + 1)  # Allow just 1 flexibility
                fv.min_flex = 0
                fv.max_flex = fv.max + 1
            else:
                # Looser constraints with some flexibility
                fv.min = max(0, target - 2)
                fv.max = target + 2
                fv.min_flex = 0
                fv.max_flex = fv.max + 2

    return scaled


def scale_features(features: FeatureCollection, scale_factor: float) -> FeatureCollection:
    """Scale feature min/max quotas by a factor (legacy function).

    Args:
        features: Original FeatureCollection
        scale_factor: Factor to scale quotas by

    Returns:
        New FeatureCollection with scaled quotas
    """
    scaled = copy.deepcopy(features)

    for feature_name in scaled:
        for value_name in scaled[feature_name]:
            fv = scaled[feature_name][value_name]
            # Scale min/max, rounding to integers
            fv.min = round(fv.min * scale_factor)
            fv.max = round(fv.max * scale_factor)
            fv.min_flex = round(fv.min_flex * scale_factor)
            fv.max_flex = round(fv.max_flex * scale_factor)

    return scaled


def main() -> None:
    """Test fixture generation."""
    print("Testing fixture generation...")

    for size in [150, 500, 1000]:
        print(f"\nGenerating {size} people fixture...")
        features, people = generate_scaled_fixtures(size) if size != 150 else load_base_fixtures()
        print(f"  People count: {people.count}")
        print(f"  Features: {len(features)}")

        # Print feature value counts
        for feature_name in list(features.keys())[:2]:  # Just first 2 features
            print(f"  {feature_name}:")
            for value_name in features[feature_name]:
                fv = features[feature_name][value_name]
                count = sum(
                    1 for pid in people if people.get_person_dict(pid)[feature_name].lower() == value_name.lower()
                )
                print(f"    {value_name}: {count} people, target [{fv.min}, {fv.max}]")


if __name__ == "__main__":
    main()
