# Command Line Interface

The CLI provides a convenient way to run sortition algorithms without writing Python code. It's ideal for:

- **One-off selections**: Quick panel selections for events or research
- **Sample code**: The code in the command line functions can be the basis for writing your own implementation.
- **Batch processing**: Running multiple selections with scripts
- **Non-programmers**: Teams who prefer command-line tools
- **Integration**: Incorporating sortition into existing workflows

## Installation

Install the CLI with optional dependencies:

```bash
# Basic installation
pip install 'sortition-algorithms[cli]'

# With Gurobi support for leximin algorithm
pip install 'sortition-algorithms[cli,gurobi]'
```

## Quick Start

```bash
# Check installation
python -m sortition_algorithms --help

# Basic CSV selection
python -m sortition_algorithms csv \
  --settings config.toml \
  --features-csv demographics.csv \
  --people-csv candidates.csv \
  --selected-csv selected.csv \
  --remaining-csv remaining.csv \
  --number-wanted 100
```

## Commands Overview

The CLI provides three main commands:

```bash
$ python -m sortition_algorithms --help
Usage: python -m sortition_algorithms [OPTIONS] COMMAND [ARGS]...

  A command line tool to exercise the sortition algorithms.

Options:
  --help  Show this message and exit.

Commands:
  csv         Do sortition with CSV files
  gen-sample  Generate sample CSV file compatible with features
  gsheet      Do sortition with Google Spreadsheets
```

## CSV Workflow

The most common usage pattern for working with local CSV files.

### Command Reference

```bash
$ python -m sortition_algorithms csv --help
Usage: python -m sortition_algorithms csv [OPTIONS]

  Do sortition with CSV files.

Options:
  -S, --settings FILE             Settings file (TOML format) [required]
  -f, --features-csv FILE         CSV with demographic features [required]
  -p, --people-csv FILE           CSV with candidate pool [required]
  -s, --selected-csv FILE         Output: selected people [required]
  -r, --remaining-csv FILE        Output: remaining people [required]
  -n, --number-wanted INTEGER     Number of people to select [required]
  --help                          Show this message and exit.
```

### Example Files

**demographics.csv** (feature definitions):

```csv
feature,value,min,max
Gender,Male,45,55
Gender,Female,45,55
Age,18-30,20,30
Age,31-50,35,45
Age,51+,25,35
Location,Urban,40,60
Location,Rural,40,60
```

**candidates.csv** (candidate pool):

```csv
id,Name,Email,Gender,Age,Location,Address,Postcode
p001,Alice Smith,alice@email.com,Female,18-30,Urban,123 Main St,12345
p002,Bob Jones,bob@email.com,Male,31-50,Rural,456 Oak Ave,67890
p003,Carol Davis,carol@email.com,Female,51+,Urban,789 Pine Rd,12345
...
```

**config.toml** (settings):

```toml
# Set to zero for secure random results
random_number_seed = 0

# Household diversity
check_same_address = true
check_same_address_columns = ["Address", "Postcode"]

# Algorithm choice
selection_algorithm = "maximin"
max_attempts = 10

# Output customization
columns_to_keep = ["Name", "Email", "Phone"]
id_column = "id"
```

### Basic Selection

```bash
python -m sortition_algorithms csv \
  --settings config.toml \
  --features-csv demographics.csv \
  --people-csv candidates.csv \
  --selected-csv selected.csv \
  --remaining-csv remaining.csv \
  --number-wanted 100
```

### Using Environment Variables

Set commonly used paths as environment variables:

```bash
export SORTITION_SETTINGS="config.toml"

python -m sortition_algorithms csv \
  --features-csv demographics.csv \
  --people-csv candidates.csv \
  -s selected.csv \
  -r remaining.csv \
  -n 100
```

### Batch Processing

Create a script for multiple selections:

```bash
#!/bin/bash
# batch_selection.sh

SETTINGS="config.toml"
FEATURES="demographics.csv"
PEOPLE="candidates.csv"
AREAS=(north south east west)
SIZE=50

for area in "${AREAS[@]}"; do
    echo "Selecting $size people..."
    python -m sortition_algorithms csv \
        --settings "$SETTINGS" \
        --features-csv "$FEATURES" \
        --people-csv "candidates_${area}.csv" \
        --selected-csv "selected_${area}.csv" \
        --remaining-csv "remaining_${area}.csv" \
        --number-wanted "$SIZE"
done
```

## Google Sheets Workflow

For organizations using Google Sheets for data management.

### Setup Requirements

1. **Google Cloud Project**: Create a project in Google Cloud Console
2. **Enable APIs**: Enable Google Sheets API and Google Drive API
3. **Service Account**: Create service account credentials
4. **Share Sheet**: Share your spreadsheet with the service account email

### Command Reference

```bash
$ python -m sortition_algorithms gsheet --help
Usage: python -m sortition_algorithms gsheet [OPTIONS]

  Do sortition with Google Spreadsheets.

Options:
  -S, --settings FILE             Settings file (TOML format) [required]
  --auth-json-file FILE           Google API credentials JSON [required]
  --gen-rem-tab / --no-gen-rem-tab Generate 'Remaining' tab [default: true]
  -g, --gsheet-name TEXT          Spreadsheet name [required]
  -f, --feature-tab-name TEXT     Features tab name [default: Categories]
  -p, --people-tab-name TEXT      People tab name [default: Categories]
  -s, --selected-tab-name TEXT    Selected output tab [default: Selected]
  -r, --remaining-tab-name TEXT   Remaining output tab [default: Remaining]
  -n, --number-wanted INTEGER     Number of people to select [required]
  --help                          Show this message and exit.
```

### Authentication Setup

1. Download service account credentials JSON file
2. **Never commit this file to version control**
3. Store securely and reference by path

### Example Usage

```bash
python -m sortition_algorithms gsheet \
  --settings config.toml \
  --auth-json-file /secure/path/credentials.json \
  --gsheet-name "Citizen Panel 2024" \
  --feature-tab-name "Demographics" \
  --people-tab-name "Candidates" \
  --selected-tab-name "Selected Panel" \
  --remaining-tab-name "Reserve Pool" \
  --number-wanted 120
```

### Spreadsheet Structure

Your Google Sheet should have tabs structured like this:

**Demographics tab**:

| feature | value  | min | max |
| ------- | ------ | --- | --- |
| Gender  | Male   | 45  | 55  |
| Gender  | Female | 45  | 55  |
| Age     | 18-30  | 20  | 30  |

**Candidates tab**:

| id   | Name  | Email             | Gender | Age   | Location |
| ---- | ----- | ----------------- | ------ | ----- | -------- |
| p001 | Alice | <alice@email.com> | Female | 18-30 | Urban    |
| p002 | Bob   | <bob@email.com>   | Male   | 31-50 | Rural    |

## Sample Generation

Generate test data compatible with your feature definitions.

### Command Reference

```bash
$ python -m sortition_algorithms gen-sample --help
Usage: python -m sortition_algorithms gen-sample [OPTIONS]

  Generate sample CSV file compatible with features and settings.

Options:
  -S, --settings FILE             Settings file [required]
  -f, --features-csv FILE         Features CSV file [required]
  -p, --people-csv FILE           Output: generated people CSV [required]
  -n, --number-wanted INTEGER     Number of people to generate [required]
  --help                          Show this message and exit.
```

### Example Usage

```bash
# Generate 500 sample people
python -m sortition_algorithms gen-sample \
  --settings config.toml \
  --features-csv demographics.csv \
  --people-csv sample_candidates.csv \
  --number-wanted 500
```

This creates a CSV with realistic synthetic data that matches your feature definitions - useful for testing quotas and algorithms.

## Configuration Files

### Settings File Format

All settings are optional and have sensible defaults:

```toml
# config.toml

# Randomization
random_number_seed = 0  # Set non-zero for reproducible results, omit for random

# Address checking for household diversity
check_same_address = true
check_same_address_columns = ["Address", "Postcode", "City"]

# Algorithm selection
selection_algorithm = "maximin"  # "maximin", "nash", "leximin", "legacy"
max_attempts = 10

# Output customization
columns_to_keep = ["Name", "Email", "Phone", "Notes"]
id_column = "id"  # Column name containing unique IDs
```

### Algorithm Comparison

| Algorithm | Pros                 | Cons                     | Use Case               |
| --------- | -------------------- | ------------------------ | ---------------------- |
| `maximin` | Fair to minorities   | May not optimize overall | Default choice         |
| `nash`    | Balanced overall     | Complex optimization     | Large diverse pools    |
| `leximin` | Strongest fairness   | Requires Gurobi license  | Academic/research      |
| `legacy`  | Backwards compatible | Less sophisticated       | Historical consistency |

## Common Workflows

### Standard Selection Process

```bash
# 1. Prepare your data files
# 2. Configure settings
# 3. Run selection
python -m sortition_algorithms csv \
  --settings config.toml \
  --features-csv demographics.csv \
  --people-csv candidates.csv \
  --selected-csv selected.csv \
  --remaining-csv remaining.csv \
  --number-wanted 100

# 4. Review results
head selected.csv
wc -l remaining.csv
```

### With Address Checking

Ensure household diversity by preventing multiple selections from the same address:

```toml
# config.toml
check_same_address = true
check_same_address_columns = ["Address", "Postcode"]
```

### Reproducible Selections

For auditable results, use a fixed random seed:

```toml
# config.toml
random_number_seed = 20241214  # Use today's date or similar
```

### Testing Quotas

Use sample generation to test if your quotas are achievable:

```bash
# Generate large sample
python -m sortition_algorithms gen-sample \
  --settings config.toml \
  --features-csv demographics.csv \
  --people-csv test_pool.csv \
  --number-wanted 1000

# Test selection
python -m sortition_algorithms csv \
  --settings config.toml \
  --features-csv demographics.csv \
  --people-csv test_pool.csv \
  --selected-csv test_selected.csv \
  --remaining-csv test_remaining.csv \
  --number-wanted 100
```

## Troubleshooting

### Common Errors

**"Selection failed"**

- Check that the sum of quota minimums for any given features don't exceed panel size (or that maximums are smaller than the panel size).
- Verify feature values match between files.
- Review constraint feasibility.

**"File not found"**

- Use absolute paths or verify working directory.
- Check file permissions.
- Ensure files exist before running.

**"Invalid feature values"**

- Verify exact string matching between demographics.csv and candidates.csv
- Check for typos, case sensitivity, extra spaces
- Review non-ASCII characters

**"Authentication failed" (Google Sheets)**

- Verify `credentials.json` is correct and accessible
- Check that service account has access to the spreadsheet
- Ensure APIs are enabled in Google Cloud Console

## Next Steps

- **[Core Concepts](concepts.md)** - Understand the theory behind sortition
- **[API Reference](api-reference.md)** - For programmatic usage
- **[Data Adapters](adapters.md)** - Custom data sources and formats
- **[Advanced Usage](advanced.md)** - Complex scenarios and optimization
