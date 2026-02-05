# ABOUTME: Anonymize CSV data files for sharing while preserving benchmark difficulty.
# ABOUTME: Removes sensitive data but maintains feature distributions and address duplicates.
"""
Anonymize sortition data files for sharing.

This script removes sensitive data from people and features CSV files while
preserving the characteristics that affect solver difficulty:
- Feature distributions remain identical
- Duplicate addresses remain duplicates
- All other identifying information is removed

Usage:
    uv run python -m benchmarks.anonymize_data \\
        --people input_people.csv \\
        --features input_features.csv \\
        --settings settings.toml \\
        --output-dir anonymized/
"""

import argparse
import csv
import tomllib
from pathlib import Path
from typing import Any

import tomli_w


def load_settings(settings_path: Path) -> dict:
    """Load settings from a TOML file."""
    with open(settings_path, "rb") as f:
        return tomllib.load(f)


def load_features_csv(features_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load features CSV and return header and rows."""
    with open(features_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def load_people_csv(people_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load people CSV and return header and rows."""
    with open(people_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def get_from_one_key(somedict: dict[str, Any], keys_to_try: list[str]) -> Any:
    """Try a few different keys to get a value, to allow for different naming"""
    for key in keys_to_try:
        if key in somedict:
            return somedict[key]
    raise KeyError(
        f"Could not find any of {', '.join(keys_to_try)} in the dict - full keys are {', '.join(somedict.keys())}"
    )


def extract_feature_info(features_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Extract feature names and their possible values from features CSV.

    Returns:
        dict mapping feature_name -> list of value names
    """
    features: dict[str, list[str]] = {}
    for row in features_rows:
        feature_name = get_from_one_key(row, ["feature", "category"])
        value_name = get_from_one_key(row, ["value", "name"])
        if feature_name not in features:
            features[feature_name] = []
        if value_name not in features[feature_name]:
            features[feature_name].append(value_name)
    return features


def create_feature_mappings(
    feature_info: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Create mappings for anonymizing feature names and values.

    Returns:
        tuple of (feature_name_map, feature_value_maps)
        - feature_name_map: old_name -> new_name (e.g., "Gender" -> "feature1")
        - feature_value_maps: feature_name -> {old_value -> new_value}
    """
    feature_name_map: dict[str, str] = {}
    feature_value_maps: dict[str, dict[str, str]] = {}

    for idx, feature_name in enumerate(feature_info.keys(), start=1):
        new_feature_name = f"feature{idx}"
        feature_name_map[feature_name] = new_feature_name

        feature_value_maps[feature_name] = {}
        for val_idx, value_name in enumerate(feature_info[feature_name], start=1):
            new_value_name = f"f{idx}value{val_idx}"
            feature_value_maps[feature_name][value_name] = new_value_name

    return feature_name_map, feature_value_maps


def create_address_mappings(
    people_rows: list[dict[str, str]],
    address_columns: list[str],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Create mappings for anonymizing address columns.

    Preserves duplicate detection by mapping identical addresses to identical values.

    Returns:
        tuple of (address_col_map, address_value_maps)
        - address_col_map: old_col -> new_col (e.g., "primary_address1" -> "address1")
        - address_value_maps: col_name -> {old_value -> new_value}
    """
    address_col_map: dict[str, str] = {}
    address_value_maps: dict[str, dict[str, str]] = {}

    for idx, col_name in enumerate(address_columns, start=1):
        new_col_name = f"address{idx}"
        address_col_map[col_name] = new_col_name

        # Collect unique values for this column
        unique_values: dict[str, str] = {}
        value_counter = 1

        for row in people_rows:
            old_value = row.get(col_name, "")
            if old_value not in unique_values:
                unique_values[old_value] = f"a{idx}value{value_counter}"
                value_counter += 1

        address_value_maps[col_name] = unique_values

    return address_col_map, address_value_maps


def anonymize_people(
    people_rows: list[dict[str, str]],
    id_column: str,
    feature_info: dict[str, list[str]],
    feature_name_map: dict[str, str],
    feature_value_maps: dict[str, dict[str, str]],
    address_columns: list[str],
    address_col_map: dict[str, str],
    address_value_maps: dict[str, dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """Anonymize people data.

    Returns:
        tuple of (new_header, new_rows)
    """
    # Build new header: id, features, addresses
    new_header = ["id"]
    for feature_name in feature_info:
        new_header.append(feature_name_map[feature_name])
    for addr_col in address_columns:
        new_header.append(address_col_map[addr_col])

    # Build new rows
    new_rows: list[dict[str, str]] = []
    for row_idx, row in enumerate(people_rows, start=1):
        new_row: dict[str, str] = {"id": str(row_idx)}

        # Map feature values
        for feature_name in feature_info:
            new_col = feature_name_map[feature_name]
            old_value = row.get(feature_name, "")
            # Handle case-insensitive matching
            value_map = feature_value_maps[feature_name]
            new_value = None
            for old_val, mapped_val in value_map.items():
                if old_val.lower() == old_value.lower():
                    new_value = mapped_val
                    break
            if new_value is None:
                # Value not in features CSV, create a placeholder
                print(
                    f"unknown! Row: {row_idx} Feature: {feature_name}, value: {old_val}, possible values: {', '.join(value_map.keys())}"
                )
                new_value = f"f{list(feature_info.keys()).index(feature_name) + 1}unknown"
            new_row[new_col] = new_value

        # Map address values
        for addr_col in address_columns:
            new_col = address_col_map[addr_col]
            old_value = row.get(addr_col, "")
            new_value = address_value_maps[addr_col].get(old_value, "")
            new_row[new_col] = new_value

        new_rows.append(new_row)

    return new_header, new_rows


def anonymize_features(
    features_rows: list[dict[str, str]],
    feature_name_map: dict[str, str],
    feature_value_maps: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Anonymize features data.

    Returns:
        list of new feature rows
    """
    new_rows: list[dict[str, str]] = []

    for row in features_rows:
        old_feature = get_from_one_key(row, ["feature", "category"])
        old_value = get_from_one_key(row, ["value", "name"])

        new_feature = feature_name_map[old_feature]
        new_value = feature_value_maps[old_feature][old_value]

        new_row = {
            "feature": new_feature,
            "value": new_value,
            "min": row["min"],
            "max": row["max"],
            "min_flex": row.get("min_flex", "0"),
            "max_flex": row.get("max_flex", row["max"]),
        }
        new_rows.append(new_row)

    return new_rows


def create_anonymized_settings(
    address_columns: list[str],
    address_col_map: dict[str, str],
    check_same_address: bool,
) -> dict:
    """Create anonymized settings dictionary."""
    new_address_columns = [address_col_map[col] for col in address_columns]

    return {
        "id_column": "id",
        "check_same_address": check_same_address,
        "check_same_address_columns": new_address_columns if check_same_address else [],
        "columns_to_keep": new_address_columns,  # Only keep address columns
        "max_attempts": 100,
        "selection_algorithm": "maximin",
        "solver_backend": "highspy",
        "random_number_seed": 0,
    }


def write_people_csv(output_path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    """Write anonymized people CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def write_features_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    """Write anonymized features CSV."""
    header = ["feature", "value", "min", "max", "min_flex", "max_flex"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def write_settings_toml(output_path: Path, settings: dict) -> None:
    """Write anonymized settings TOML."""
    with open(output_path, "wb") as f:
        tomli_w.dump(settings, f)


def parse_args() -> argparse.Namespace:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Anonymize sortition data files for sharing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    uv run python -m benchmarks.anonymize_data \\
        --people data/candidates.csv \\
        --features data/features.csv \\
        --settings data/settings.toml \\
        --output-dir anonymized/

    # The output directory will contain:
    #   - people.csv (anonymized)
    #   - features.csv (anonymized)
    #   - settings.toml (with new column names)
""",
    )
    parser.add_argument(
        "--people",
        type=Path,
        required=True,
        help="Path to input people/candidates CSV file",
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to input features/targets CSV file",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        required=True,
        help="Path to settings TOML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write anonymized files",
    )

    args = parser.parse_args()
    # Validate inputs
    if not args.people.exists():
        raise Exception(f"Error: People file not found: {args.people}")
    if not args.features.exists():
        raise Exception(f"Error: Features file not found: {args.features}")
    if not args.settings.exists():
        raise Exception(f"Error: Settings file not found: {args.settings}")
    return args


def main() -> None:
    args = parse_args()
    # Load inputs
    print(f"Loading settings from {args.settings}...")
    settings = load_settings(args.settings)

    print(f"Loading features from {args.features}...")
    _, features_rows = load_features_csv(args.features)

    print(f"Loading people from {args.people}...")
    _, people_rows = load_people_csv(args.people)

    # Extract configuration
    id_column = settings.get("id_column", "id")
    check_same_address = settings.get("check_same_address", False)
    address_columns = settings.get("check_same_address_columns", []) if check_same_address else []

    print(f"  ID column: {id_column}")
    print(f"  Check same address: {check_same_address}")
    if address_columns:
        print(f"  Address columns: {address_columns}")

    # Extract feature information
    feature_info = extract_feature_info(features_rows)
    print(f"  Features: {list(feature_info.keys())}")

    # Create mappings
    print("\nCreating anonymization mappings...")
    feature_name_map, feature_value_maps = create_feature_mappings(feature_info)
    address_col_map, address_value_maps = create_address_mappings(people_rows, address_columns)

    # Print mapping summary
    print("  Feature mappings:")
    for old_name, new_name in feature_name_map.items():
        num_values = len(feature_value_maps[old_name])
        print(f"    {old_name} -> {new_name} ({num_values} values)")

    if address_columns:
        print("  Address mappings:")
        for old_col, new_col in address_col_map.items():
            num_values = len(address_value_maps[old_col])
            print(f"    {old_col} -> {new_col} ({num_values} unique values)")

    # Anonymize data
    print("\nAnonymizing data...")
    new_people_header, new_people_rows = anonymize_people(
        people_rows,
        id_column,
        feature_info,
        feature_name_map,
        feature_value_maps,
        address_columns,
        address_col_map,
        address_value_maps,
    )
    new_features_rows = anonymize_features(features_rows, feature_name_map, feature_value_maps)
    new_settings = create_anonymized_settings(address_columns, address_col_map, check_same_address)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write outputs
    print(f"\nWriting anonymized files to {args.output_dir}/...")

    people_out = args.output_dir / "people.csv"
    write_people_csv(people_out, new_people_header, new_people_rows)
    print(f"  Wrote {len(new_people_rows)} people to {people_out}")

    features_out = args.output_dir / "features.csv"
    write_features_csv(features_out, new_features_rows)
    print(f"  Wrote {len(new_features_rows)} feature rows to {features_out}")

    settings_out = args.output_dir / "settings.toml"
    write_settings_toml(settings_out, new_settings)
    print(f"  Wrote settings to {settings_out}")

    print("\nDone! Anonymized files are ready for sharing.")


if __name__ == "__main__":
    main()
