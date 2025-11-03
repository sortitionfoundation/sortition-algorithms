# Quick Start Guide

This guide will get you up and running with sortition algorithms in just a few minutes.

## Installation

```bash
pip install sortition-algorithms

# Optional: Install CLI support
pip install 'sortition-algorithms[cli]'

# Optional: Install leximin algorithm support (requires commercial/academic license)
pip install 'sortition-algorithms[gurobi]'
```

## Basic Concepts

Before diving in, understand these key concepts:

- **Sortition**: Random selection that maintains demographic representativeness
- **Features**: Demographic characteristics (e.g., Gender, Age, Location)
- **Quotas**: Min/max targets for each demographic group
- **Stratified Selection**: Random selection that respects quotas

## Your First Selection

### 1. Prepare Your Data

You'll need two CSV files:

**demographics.csv** (features with quotas):

```csv
feature,value,min,max
Gender,Male,45,55
Gender,Female,45,55
Age,18-30,20,30
Age,31-50,30,40
Age,51+,30,50
```

**candidates.csv** (people to select from):

```csv
id,Gender,Age,Location
person1,Male,18-30,Urban
person2,Female,31-50,Rural
person3,Male,51+,Urban
...
```

### 2. Run Your First Selection

```python
from sortition_algorithms import (
    run_stratification,
    read_in_features,
    read_in_people,
    Settings
)

# Load your data
settings = Settings()
features = read_in_features("demographics.csv")
people = read_in_people("candidates.csv", settings, features)

# Select a panel of 50 people
success, selected_panels, report = run_stratification(
    features=features,
    people=people,
    number_people_wanted=50,
    settings=settings
)

if success:
    selected_people = selected_panels[0]  # frozenset of person IDs
    print(f"✅ Successfully selected {len(selected_people)} people")
    print("Selected IDs:", list(selected_people)[:5], "...")
else:
    print("❌ Selection failed")
    if report.last_error():
        print(str(report.last_error()))

# Display the detailed report
print(report.as_text())
```

### 3. Export Results

```python
from sortition_algorithms import selected_remaining_tables

# Get formatted tables for export
selected_table, remaining_table, info = selected_remaining_tables(
    people, selected_panels[0], features, settings
)

# Save to CSV
import csv

with open("selected.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(selected_table)

with open("remaining.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(remaining_table)
```

## Using the Command Line

For quick operations, use the CLI:

```bash
# CSV workflow
python -m sortition_algorithms csv \
  --settings settings.toml \
  --features-csv demographics.csv \
  --people-csv candidates.csv \
  --selected-csv selected.csv \
  --remaining-csv remaining.csv \
  --number-wanted 50
```

## Configuration with Settings

Customize behavior with a settings file:

**settings.toml**:

```toml
id_column = "my_id"

# Random seed for reproducible results (optional)
# Set to zero to be properly random
random_number_seed = 0

# Ensure household diversity
check_same_address = true
check_same_address_columns = ["Address", "Postcode"]

# Selection algorithm: "maximin", "leximin", "nash", or "legacy"
selection_algorithm = "maximin"

# Maximum selection attempts
max_attempts = 10

# Output columns to include
columns_to_keep = ["Name", "Email", "Phone"]
```

```python
settings, report = Settings.load_from_file("settings.toml")
```

## Common Patterns

### Working with Google Sheets

```python
from sortition_algorithms import GSheetDataSource, SelectionData
from pathlib import Path

data_source = GSheetDataSource(
    feature_tab_name="Demographics",
    people_tab_name="Candidates",
    auth_json_path=Path("credentials.json"),
    gen_rem_tab=True,
)
data_source.set_g_sheet_name("My Spreadsheet")
select_data = SelectionData(data_source)
features, report = select_data.load_features()
people, report = select_data.load_people(settings, features)
```

### Address Checking for Household Diversity

```python
# Ensure only one person per household is selected
settings = Settings(
    check_same_address=True,
    check_same_address_columns=["Address", "Postcode"]
)
```

### Multiple Selection Algorithms

```python
# Maximin: Maximize the minimum probability
settings.selection_algorithm = "maximin"

# Nash: Maximize the product of probabilities
settings.selection_algorithm = "nash"

# Leximin: Lexicographic maximin (requires Gurobi)
settings.selection_algorithm = "leximin"
```

Read [more about the algorithms](concepts.md#selection-algorithms).

## Working with Reports and Logging

Most library functions return a `RunReport` object containing detailed status information:

```python
# Reports contain formatted messages and tables
features, report = adapter.load_features_from_file(Path("features.csv"))
print("Loading report:")
print(report.as_text())

# Get HTML for web display
html_report = report.as_html()

# Control whether to show messages that were already logged
summary = report.as_text(include_logged=False)

# Extract the last error added to the report (or None if there was no error)
error = report.last_error()
```

### Custom Logging Integration

Redirect log messages for integration with your application:

```python
from sortition_algorithms.utils import override_logging_handlers
import logging

# Send logs to a file
file_handler = logging.FileHandler('sortition.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

override_logging_handlers([file_handler], [file_handler])

# Now all library operations will log to the file
success, panels, report = run_stratification(features, people, 50, settings)
```

See the [API Reference](api-reference.md#custom-logging) for complete logging documentation.

## What's Next?

- **[Core Concepts](concepts.md)** - Deep dive into sortition theory
- **[API Reference](api-reference.md)** - Complete function documentation
- **[CLI Usage](cli.md)** - Advanced command line examples
- **[Data Adapters](adapters.md)** - CSV, Google Sheets, and custom adapters
- **[Advanced Usage](advanced.md)** - Complex scenarios and troubleshooting

## Troubleshooting

**"Selection failed" errors**: Check that your quotas are achievable given your candidate pool. The sum of minimum quotas shouldn't exceed your target panel size.

**Import errors**: Ensure you've installed the package correctly. For Gurobi features, you need a valid license.

**Empty results**: Verify your CSV files have the correct format and column names match between demographics and candidates files.
