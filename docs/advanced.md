# Advanced Usage

This guide covers complex scenarios, optimization techniques, troubleshooting strategies, and advanced usage patterns for the sortition algorithms library.

## Algorithm Deep Dive

Read [more about the algorithms](concepts.md#selection-algorithms).

## Complex Scenarios

### Weighted Selection

For scenarios where some demographic groups need stronger representation:

```python
def create_weighted_features():
    """Create features with weighted quotas for underrepresented groups."""

    # Standard proportional representation
    base_features = [
        ("Gender", "Male", 45, 55),
        ("Gender", "Female", 45, 55),
        ("Age", "18-30", 20, 30),
        ("Age", "31-50", 35, 45),
        ("Age", "51+", 25, 35),
    ]

    # Weighted to ensure representation of underrepresented groups
    weighted_features = [
        ("Gender", "Male", 40, 50),       # Slightly reduce majority
        ("Gender", "Female", 45, 55),     # Maintain strong representation
        ("Gender", "Non-binary", 5, 10),  # Ensure inclusion
        ("Age", "18-30", 25, 35),         # Boost young representation
        ("Age", "31-50", 35, 45),
        ("Age", "51+", 20, 30),
    ]

    return create_features_from_list(weighted_features)

def create_features_from_list(feature_list):
    """Helper to create FeatureCollection from tuples."""
    import csv
    from io import StringIO

    # Convert to CSV format
    csv_content = "feature,value,min,max\n"
    for feature, value, min_val, max_val in feature_list:
        csv_content += f"{feature},{value},{min_val},{max_val}\n"

    # Use CSV adapter to create FeatureCollection
    data_source = CSVStringDataSource(csv_content, "")
    select_data = SelectionData(data_source)
    features, msgs = data_source.load_features()
    return features
```

## Troubleshooting Guide

### Common Error Patterns

#### Infeasible Quotas

**Symptoms**: `InfeasibleQuotasError` exception

**Diagnosis**:

```python
def diagnose_quota_feasibility(features: FeatureCollection, panel_size: int):
    """Analyze why quotas might be infeasible."""

    issues = []

    max_value_of_minimums = minimum_selection(features)
    if max_value_of_minimums > panel_size:
        issues.append(f"Max value of minimums ({max_value_of_minimums}) exceeds panel size ({panel_size})")

    min_value_of_maximums = maximum_selection(features)
    if min_value_of_maximums < panel_size:
        issues.append(f"Min value of maximums ({min_value_of_maximums}) is less than panel size ({panel_size})")

    # Check for impossible individual quotas
    for feature_name in features:
        sum_of_min = sum(c.min for c in features[feature_name].values())
        sum_of_max = sum(c.max for c in features[feature_name].values())

        if sum_of_min > panel_size:
            issues.append(f"{feature_name} sum of minimum ({sum_of_min}) exceeds panel size")

        if sum_of_max < panel_size:
            issues.append(f"{feature_name} sum of maximum ({sum_of_max}) is less than panel size")

    for feature_name, value_name, fv_minmax in iterate_feature_collection(features):
        if fv_minmax.max < fv_minmax.min:
            issues.append(f"{feature_name}:{value_name} max ({fv_minmax.max}) < min ({fv_minmax.min})")

    return issues

def suggest_quota_fixes(features: FeatureCollection, people: People, panel_size: int):
    """Suggest quota adjustments to make selection feasible."""

    suggestions = []

    # Count available people per category
    availability = {}
    for person_id in people:
        person_data = people.get_person_dict(person_id)
        for feature_name in features:
            value = person_data.get(feature_name, "Unknown")
            key = (feature_name, value)
            availability[key] = availability.get(key, 0) + 1

    # Suggest adjustments
    for feature_name, value_name, fv_minmax in iterate_feature_collection(features):
        available = availability.get((feature_name, value_name), 0)

        if fv_minmax.min > available:
            suggestions.append(
                f"Reduce {feature_name}:{value_name} minimum from {fv_minmax.min} to {available} "
                f"(only {available} candidates available)"
            )

    return suggestions
```

**Solutions**:

1. **Reduce minimum quotas**: Lower the minimum requirements
2. **Increase maximum quotas**: Allow more flexibility
3. **Expand candidate pool**: Recruit more candidates in underrepresented categories
4. **Adjust panel size**: Sometimes a smaller or larger panel works better

#### Data Quality Issues

**Symptoms**: Unexpected selection results, warnings about data inconsistencies

**Diagnosis**:

```python
from collection import Counter, defaultdict

def audit_data_quality(people: People, features: FeatureCollection):
    """Comprehensive data quality audit."""

    issues = []

    # Check for missing demographic data
    for person_id in people:
        person_data = people.get_person_dict(person_id)

        for feature in features:
            if feature not in person_data or not person_data[feature].strip():
                issues.append(f"Person {person_id} missing {feature}")

    # Check for unexpected feature values
    expected_values = {name: set(features[name].keys()]) for name in features}

    for person_id in people:
        person_data = people.get_person_dict(person_id)

        for feature_name, values in expected_values.items():
            actual_val = person_data.get(feature_name, "")
            if actual_val and actual_val not in values:
                issues.append(
                    f"Person {person_id} has unexpected {feature_name} value: '{actual_val}'"
                )

    # Check for duplicate IDs
    count_ids = Counter(people)
    for person_id, count in count_ids.items():
        if count > 1:
            issues.append(f"Duplicate person ID: {person_id}")

    return issues

def clean_data_automatically(people_data: list[dict], features: FeatureCollection):
    """Automatically clean common data issues."""

    cleaned_data = []

    for person in people_data:
        cleaned_person = {}

        for key, value in person.items():
            # Strip whitespace
            if isinstance(value, str):
                value = value.strip()

            # Standardize case for categorical variables
            if key in features:
                # Convert to title case for consistency
                value = value.title() if value else ""

            cleaned_person[key] = value

        # Skip records with missing required data
        required_fields = ["id"] + list(features.keys())
        if all(cleaned_person.get(field) for field in required_fields):
            cleaned_data.append(cleaned_person)

    return cleaned_data
```

## Best Practices Summary

### Development Best Practices

1. **Always validate inputs**: Check data quality before running selections
2. **Use appropriate random seeds**: Fixed seeds for testing, None for production
3. **Handle errors gracefully**: Provide meaningful error messages and recovery options
4. **Test with edge cases**: Small pools, extreme quotas, missing data
5. **Monitor performance**: Track memory usage and runtime for large datasets

### Production Best Practices

1. **Implement comprehensive logging**: Track all selection attempts and results
2. **Set up monitoring and alerting**: Detect failures and performance issues
3. **Use version control for configurations**: Track changes to quotas and settings
4. **Backup candidate data**: Ensure data persistence and recoverability
5. **Document selection criteria**: Maintain audit trails for transparency

## Next Steps

- **[Core Concepts](concepts.md)** - Understand sortition fundamentals
- **[API Reference](api-reference.md)** - Complete function documentation
- **[Data Adapters](adapters.md)** - Working with different data sources
