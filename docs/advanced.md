# Advanced Usage

This guide covers complex scenarios, optimization techniques, troubleshooting strategies, and advanced usage patterns for the sortition algorithms library.

## Algorithm Deep Dive

### Understanding Selection Algorithms

Each algorithm optimizes for different fairness criteria:

#### Maximin Algorithm (Default)
**Objective**: Maximize the minimum selection probability across all groups.

```python
settings = Settings(selection_algorithm="maximin")
```

**When to use**:
- Default choice for most applications
- Ensures no group is severely underrepresented
- Good for citizen assemblies and deliberative panels

**Trade-offs**:
- May not optimize overall fairness
- Can be conservative in selection choices

**Example scenario**: A panel where ensuring minimum representation for small minorities is crucial.

#### Nash Algorithm
**Objective**: Maximize the product of all selection probabilities.

```python
settings = Settings(selection_algorithm="nash")
```

**When to use**:
- Large, diverse candidate pools
- When you want balanced representation across all groups
- Academic research requiring mathematical optimality

**Trade-offs**:
- More complex optimization
- May be harder to explain to stakeholders

**Example scenario**: Research study requiring theoretically optimal fairness across all demographic groups.

#### Leximin Algorithm
**Objective**: Lexicographic maximin optimization (requires Gurobi license).

```python
settings = Settings(selection_algorithm="leximin")
```

**When to use**:
- Academic research requiring strongest fairness guarantees
- When you have access to Gurobi (commercial/academic license)
- High-stakes selections where maximum fairness is essential

**Trade-offs**:
- Requires commercial solver (Gurobi)
- More computationally intensive
- May be overkill for routine selections

**Example scenario**: Government-sponsored citizen assembly where mathematical proof of fairness is required.

#### Legacy Algorithm
**Objective**: Backwards compatibility with earlier implementations.

```python
settings = Settings(selection_algorithm="legacy")
```

**When to use**:
- Reproducing historical selections
- Comparison studies
- Specific compatibility requirements

**Trade-offs**:
- Less sophisticated than modern algorithms
- May not provide optimal fairness

### Algorithm Performance Comparison

```python
def compare_algorithms():
    algorithms = ["maximin", "nash", "leximin"]
    results = {}

    for algorithm in algorithms:
        settings = Settings(
            selection_algorithm=algorithm,
            random_number_seed=42  # For fair comparison
        )

        start_time = time.time()
        success, panels, msgs = run_stratification(
            features, people, 100, settings
        )
        end_time = time.time()

        results[algorithm] = {
            "success": success,
            "runtime": end_time - start_time,
            "panel_size": len(panels[0]) if success else 0,
            "messages": len(msgs)
        }

    return results
```

## Complex Scenarios

### Multiple Selection Rounds

For applications requiring multiple panels:

```python
def multiple_panel_selection():
    settings = Settings(random_number_seed=None)  # Different each time
    all_panels = []
    remaining_people = deepcopy(original_people)

    for round_num in range(5):  # 5 panels of 50 each
        success, panels, msgs = run_stratification(
            features, remaining_people, 50, settings
        )

        if success:
            selected_panel = panels[0]
            all_panels.append(selected_panel)

            # Remove selected people from pool
            for person_id in selected_panel:
                remaining_people.remove(person_id)

            print(f"Round {round_num + 1}: Selected {len(selected_panel)} people")
            print(f"Remaining pool: {len(remaining_people)} people")
        else:
            print(f"Round {round_num + 1} failed: {msgs}")
            break

    return all_panels
```

### Weighted Selection

For scenarios where some demographic groups need stronger representation:

```python
def create_weighted_features():
    \"\"\"Create features with weighted quotas for underrepresented groups.\"\"\"

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
    \"\"\"Helper to create FeatureCollection from tuples.\"\"\"
    import csv
    from io import StringIO

    # Convert to CSV format
    csv_content = "feature,value,min,max\\n"
    for feature, value, min_val, max_val in feature_list:
        csv_content += f"{feature},{value},{min_val},{max_val}\\n"

    # Use CSV adapter to create FeatureCollection
    adapter = CSVAdapter()
    features, msgs = adapter.load_features_from_str(csv_content)
    return features
```

### Dynamic Quota Adjustment

Automatically adjust quotas based on available candidates:

```python
def adjust_quotas_for_availability(features: FeatureCollection, people: People) -> FeatureCollection:
    \"\"\"Adjust quotas based on actual candidate availability.\"\"\"

    # Count available people in each category
    category_counts = {}
    for person_id in people:
        person_data = people.get_person_dict(person_id)

        for feature_name in features.feature_names:
            feature_value = person_data.get(feature_name, "Unknown")
            key = (feature_name, feature_value)
            category_counts[key] = category_counts.get(key, 0) + 1

    # Calculate proportional quotas
    total_people = len(people)
    adjusted_features = []

    for feature_name, value_name, value_counts in features.feature_values_counts():
        available = category_counts.get((feature_name, value_name), 0)

        if available == 0:
            # No candidates available - set quotas to 0
            min_quota = max_quota = 0
        else:
            # Calculate proportional representation with some flexibility
            proportion = available / total_people
            min_quota = max(0, int(proportion * 100) - 5)  # Allow 5% flexibility
            max_quota = min(available, int(proportion * 100) + 5)

        adjusted_features.append((feature_name, value_name, min_quota, max_quota))

    return create_features_from_list(adjusted_features)
```

### Hierarchical Quotas

For complex quota relationships:

```python
def hierarchical_quota_validation(features: FeatureCollection, panel_size: int) -> list[str]:
    \"\"\"Validate hierarchical quota constraints.\"\"\"
    warnings = []

    # Example: Age + Education constraints
    # Ensure university graduates are distributed across age groups

    age_university_constraints = {
        ("Age", "18-30", "Education", "University"): (5, 15),  # 5-15 young graduates
        ("Age", "31-50", "Education", "University"): (10, 20), # 10-20 middle-age graduates
        ("Age", "51+", "Education", "University"): (5, 15),    # 5-15 older graduates
    }

    for (age_feature, age_value, edu_feature, edu_value), (min_q, max_q) in age_university_constraints.items():
        # This is conceptual - actual implementation would need
        # custom constraint checking logic
        warnings.append(f"Hierarchical constraint: {age_value} {edu_value} should be {min_q}-{max_q}")

    return warnings
```

## Performance Optimization

### Large Dataset Handling

For pools with hundreds of thousands of candidates:

```python
def optimize_for_large_datasets():
    settings = Settings(
        # Reduce retry attempts for speed
        max_attempts=3,

        # Use faster algorithm
        selection_algorithm="maximin",

        # Minimize address checking overhead if not needed
        check_same_address=False
    )

    return settings

def batch_process_candidates(people_file: Path, batch_size: int = 10000):
    \"\"\"Process large candidate files in batches.\"\"\"

    # Read file in chunks
    import pandas as pd

    chunk_iter = pd.read_csv(people_file, chunksize=batch_size)

    all_people = []
    for chunk in chunk_iter:
        # Process each chunk
        chunk_dict = chunk.to_dict('records')
        all_people.extend(chunk_dict)

        # Optional: provide progress feedback
        print(f"Processed {len(all_people)} candidates...")

    return all_people
```

### Memory Management

For memory-constrained environments:

```python
def memory_efficient_selection():
    \"\"\"Demonstrate memory-efficient patterns.\"\"\"

    # Use generators instead of loading all data at once
    def load_people_generator(file_path: Path):
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row

    # Process in smaller batches
    def process_in_batches(data_generator, batch_size: int = 1000):
        batch = []
        for item in data_generator:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # Clean up intermediate objects
    import gc

    def cleanup_after_selection():
        gc.collect()  # Force garbage collection
```

### Parallel Processing

For multiple independent selections:

```python
import concurrent.futures
from multiprocessing import Pool

def parallel_selections(features, people, panel_sizes: list[int]):
    \"\"\"Run multiple selections in parallel.\"\"\"

    def run_single_selection(panel_size):
        settings = Settings(random_number_seed=None)  # Different seed each time
        return run_stratification(features, people, panel_size, settings)

    # Use process pool for CPU-bound work
    with Pool() as pool:
        results = pool.map(run_single_selection, panel_sizes)

    return results

def concurrent_algorithm_comparison(features, people, panel_size: int):
    \"\"\"Compare algorithms concurrently.\"\"\"

    algorithms = ["maximin", "nash"]

    def test_algorithm(algorithm):
        settings = Settings(
            selection_algorithm=algorithm,
            random_number_seed=42  # Same seed for fair comparison
        )
        return run_stratification(features, people, panel_size, settings)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(test_algorithm, alg): alg for alg in algorithms}
        results = {}

        for future in concurrent.futures.as_completed(futures):
            algorithm = futures[future]
            results[algorithm] = future.result()

    return results
```

## Troubleshooting Guide

### Common Error Patterns

#### Infeasible Quotas

**Symptoms**: `InfeasibleQuotasError` exception

**Diagnosis**:
```python
def diagnose_quota_feasibility(features: FeatureCollection, panel_size: int):
    \"\"\"Analyze why quotas might be infeasible.\"\"\"

    issues = []

    # Check if minimum quotas exceed panel size
    total_minimums = sum(
        value_counts.min
        for _, _, value_counts in features.feature_values_counts()
    )

    if total_minimums > panel_size:
        issues.append(f"Sum of minimums ({total_minimums}) exceeds panel size ({panel_size})")

    # Check for impossible individual quotas
    for feature_name, value_name, value_counts in features.feature_values_counts():
        if value_counts.min > panel_size:
            issues.append(f"{feature_name}:{value_name} minimum ({value_counts.min}) exceeds panel size")

        if value_counts.max < value_counts.min:
            issues.append(f"{feature_name}:{value_name} max ({value_counts.max}) < min ({value_counts.min})")

    return issues

def suggest_quota_fixes(features: FeatureCollection, people: People, panel_size: int):
    \"\"\"Suggest quota adjustments to make selection feasible.\"\"\"

    suggestions = []

    # Count available people per category
    availability = {}
    for person_id in people:
        person_data = people.get_person_dict(person_id)
        for feature_name in features.feature_names:
            value = person_data.get(feature_name, "Unknown")
            key = (feature_name, value)
            availability[key] = availability.get(key, 0) + 1

    # Suggest adjustments
    for feature_name, value_name, value_counts in features.feature_values_counts():
        available = availability.get((feature_name, value_name), 0)

        if value_counts.min > available:
            suggestions.append(
                f"Reduce {feature_name}:{value_name} minimum from {value_counts.min} to {available} "
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
def audit_data_quality(people: People, features: FeatureCollection):
    \"\"\"Comprehensive data quality audit.\"\"\"

    issues = []

    # Check for missing demographic data
    required_features = features.feature_names
    for person_id in people:
        person_data = people.get_person_dict(person_id)

        for feature in required_features:
            if feature not in person_data or not person_data[feature].strip():
                issues.append(f"Person {person_id} missing {feature}")

    # Check for unexpected feature values
    expected_values = {}
    for feature_name, value_name, _ in features.feature_values_counts():
        if feature_name not in expected_values:
            expected_values[feature_name] = set()
        expected_values[feature_name].add(value_name)

    for person_id in people:
        person_data = people.get_person_dict(person_id)

        for feature_name, expected_vals in expected_values.items():
            actual_val = person_data.get(feature_name, "")
            if actual_val and actual_val not in expected_vals:
                issues.append(
                    f"Person {person_id} has unexpected {feature_name} value: '{actual_val}'"
                )

    # Check for duplicate IDs
    seen_ids = set()
    for person_id in people:
        if person_id in seen_ids:
            issues.append(f"Duplicate person ID: {person_id}")
        seen_ids.add(person_id)

    return issues

def clean_data_automatically(people_data: list[dict], features: FeatureCollection):
    \"\"\"Automatically clean common data issues.\"\"\"

    cleaned_data = []

    for person in people_data:
        cleaned_person = {}

        for key, value in person.items():
            # Strip whitespace
            if isinstance(value, str):
                value = value.strip()

            # Standardize case for categorical variables
            if key in features.feature_names:
                # Convert to title case for consistency
                value = value.title() if value else ""

            cleaned_person[key] = value

        # Skip records with missing required data
        required_fields = ["id"] + features.feature_names
        if all(cleaned_person.get(field) for field in required_fields):
            cleaned_data.append(cleaned_person)

    return cleaned_data
```

#### Performance Issues

**Symptoms**: Long runtime, memory errors, timeouts

**Diagnosis**:
```python
import time
import psutil
import tracemalloc

def profile_selection_performance():
    \"\"\"Profile memory and CPU usage during selection.\"\"\"

    # Start memory tracing
    tracemalloc.start()

    # Monitor CPU and memory
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()

    try:
        # Run your selection
        success, panels, msgs = run_stratification(features, people, 100, settings)

        # Measure resource usage
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Runtime: {end_time - start_time:.2f} seconds")
        print(f"Memory used: {end_memory - start_memory:.2f} MB")
        print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

        return success, panels, msgs

    except Exception as e:
        tracemalloc.stop()
        raise e
```

### Debug Mode

Enable detailed debugging:

```python
def debug_selection_process():
    \"\"\"Run selection with comprehensive debugging.\"\"\"

    import logging

    # Set up detailed logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Validate inputs
    logger.info("Starting selection process validation...")

    # Check data quality
    data_issues = audit_data_quality(people, features)
    if data_issues:
        logger.warning(f"Found {len(data_issues)} data quality issues:")
        for issue in data_issues[:10]:  # Show first 10
            logger.warning(f"  - {issue}")

    # Check quota feasibility
    quota_issues = diagnose_quota_feasibility(features, 100)
    if quota_issues:
        logger.error("Quota feasibility issues:")
        for issue in quota_issues:
            logger.error(f"  - {issue}")
        return False, [], ["Quota issues prevent selection"]

    # Run selection with profiling
    logger.info("Starting selection algorithm...")
    return profile_selection_performance()
```

## Integration Patterns

### Web Application Integration

For Flask/Django applications:

```python
from flask import Flask, request, jsonify, send_file
import tempfile
import os

app = Flask(__name__)

@app.route('/api/selection', methods=['POST'])
def run_selection_api():
    try:
        # Parse request
        data = request.get_json()
        panel_size = data['panel_size']
        features_data = data['features']
        people_data = data['people']

        # Convert to library objects
        features = create_features_from_data(features_data)
        people = create_people_from_data(people_data)

        # Run selection
        settings = Settings()
        success, panels, msgs = run_stratification(features, people, panel_size, settings)

        if success:
            # Format results
            selected_table, remaining_table, _ = selected_remaining_tables(
                people, panels[0], features, settings
            )

            return jsonify({
                'success': True,
                'selected_count': len(panels[0]),
                'selected_ids': list(panels[0]),
                'messages': msgs
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Selection failed',
                'messages': msgs
            }), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/selection/export/<format>')
def export_results(format):
    # Implementation for exporting results in various formats
    pass
```

### Batch Processing Pipeline

For automated processing:

```python
import argparse
import sys
from pathlib import Path
from datetime import datetime

def batch_processing_pipeline():
    \"\"\"Complete pipeline for batch processing multiple selections.\"\"\"

    parser = argparse.ArgumentParser(description='Batch sortition processing')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--panel-sizes', nargs='+', type=int, default=[100])
    parser.add_argument('--algorithms', nargs='+', default=['maximin'])

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = args.output_dir / f"selection_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Process each configuration
    results_summary = []

    for algorithm in args.algorithms:
        for panel_size in args.panel_sizes:
            try:
                result = process_single_configuration(
                    args.input_dir, run_dir, algorithm, panel_size
                )
                results_summary.append(result)

            except Exception as e:
                print(f"Error processing {algorithm}/{panel_size}: {e}")
                results_summary.append({
                    'algorithm': algorithm,
                    'panel_size': panel_size,
                    'success': False,
                    'error': str(e)
                })

    # Generate summary report
    generate_summary_report(run_dir, results_summary)

    return results_summary

def process_single_configuration(input_dir: Path, output_dir: Path, algorithm: str, panel_size: int):
    \"\"\"Process a single algorithm/panel size combination.\"\"\"

    # Load data
    adapter = CSVAdapter()
    features, _ = adapter.load_features_from_file(input_dir / "features.csv")
    people, _ = adapter.load_people_from_file(input_dir / "people.csv", Settings(), features)

    # Configure settings
    settings = Settings(
        selection_algorithm=algorithm,
        random_number_seed=42  # Reproducible for comparison
    )

    # Run selection
    success, panels, msgs = run_stratification(features, people, panel_size, settings)

    if success:
        # Export results
        selected_table, remaining_table, _ = selected_remaining_tables(
            people, panels[0], features, settings
        )

        output_prefix = f"{algorithm}_{panel_size}"

        with open(output_dir / f"{output_prefix}_selected.csv", "w", newline="") as f:
            adapter.selected_file = f
            adapter._write_rows(f, selected_table)

        with open(output_dir / f"{output_prefix}_remaining.csv", "w", newline="") as f:
            adapter.remaining_file = f
            adapter._write_rows(f, remaining_table)

    return {
        'algorithm': algorithm,
        'panel_size': panel_size,
        'success': success,
        'selected_count': len(panels[0]) if success else 0,
        'messages': len(msgs)
    }
```

### Monitoring and Alerting

For production deployments:

```python
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

class SelectionMonitor:
    \"\"\"Monitor selection processes and send alerts.\"\"\"

    def __init__(self, email_config: dict):
        self.email_config = email_config

    def monitor_selection(self, features, people, panel_size, settings):
        \"\"\"Run selection with monitoring and alerting.\"\"\"

        start_time = datetime.now()

        try:
            # Pre-flight checks
            issues = self._pre_flight_checks(features, people, panel_size)
            if issues:
                self._send_alert("Pre-flight check failed", "\\n".join(issues))
                return False, [], issues

            # Run selection
            success, panels, msgs = run_stratification(features, people, panel_size, settings)

            # Post-selection analysis
            if success:
                analysis = self._analyze_results(people, panels[0], features)
                self._send_success_notification(analysis)
            else:
                self._send_alert("Selection failed", "\\n".join(msgs))

            return success, panels, msgs

        except Exception as e:
            self._send_alert("Selection error", str(e))
            raise

        finally:
            duration = datetime.now() - start_time
            print(f"Selection completed in {duration}")

    def _pre_flight_checks(self, features, people, panel_size):
        \"\"\"Run pre-flight checks before selection.\"\"\"
        issues = []

        # Check pool size
        if len(people) < panel_size * 2:
            issues.append(f"Small candidate pool: {len(people)} for panel of {panel_size}")

        # Check quota feasibility
        quota_issues = diagnose_quota_feasibility(features, panel_size)
        issues.extend(quota_issues)

        return issues

    def _analyze_results(self, people, selected_panel, features):
        \"\"\"Analyze selection results for quality metrics.\"\"\"

        analysis = {
            'panel_size': len(selected_panel),
            'pool_size': len(people),
            'selection_rate': len(selected_panel) / len(people),
            'demographic_breakdown': {}
        }

        # Calculate demographic breakdown
        for person_id in selected_panel:
            person_data = people.get_person_dict(person_id)
            for feature_name in features.feature_names:
                feature_value = person_data.get(feature_name, "Unknown")
                key = f"{feature_name}:{feature_value}"
                analysis['demographic_breakdown'][key] = analysis['demographic_breakdown'].get(key, 0) + 1

        return analysis

    def _send_alert(self, subject: str, message: str):
        \"\"\"Send email alert.\"\"\"
        # Implementation depends on your email setup
        print(f"ALERT - {subject}: {message}")

    def _send_success_notification(self, analysis: dict):
        \"\"\"Send success notification with analysis.\"\"\"
        message = f"Selection completed successfully. Panel size: {analysis['panel_size']}"
        print(f"SUCCESS - {message}")
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

### Scaling Best Practices

1. **Optimize for your use case**: Choose appropriate algorithms and settings
2. **Consider parallel processing**: For multiple independent selections
3. **Implement caching**: For expensive data loading operations
4. **Monitor resource usage**: Plan capacity for peak loads
5. **Use appropriate hardware**: SSDs for I/O intensive operations

## Next Steps

- **[Core Concepts](concepts.md)** - Understand sortition fundamentals
- **[Quick Start](quickstart.md)** - Get started quickly
- **[API Reference](api-reference.md)** - Complete function documentation
- **[CLI Usage](cli.md)** - Command line interface
- **[Data Adapters](adapters.md)** - Working with different data sources
