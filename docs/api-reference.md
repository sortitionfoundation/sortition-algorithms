# API Reference

Complete documentation for all public functions and classes in the sortition-algorithms library.

## Core Functions

### run_stratification()

Main function for running stratified random selection with retry logic.

```python
def run_stratification(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    settings: Settings,
    test_selection: bool = False,
    number_selections: int = 1,
) -> tuple[bool, list[frozenset[str]], list[str]]:
```

**Parameters:**

- `features`: FeatureCollection with min/max quotas for each feature value
- `people`: People object containing the pool of candidates
- `number_people_wanted`: Desired size of the panel
- `settings`: Settings object containing configuration
- `test_selection`: If True, don't randomize (for testing only)
- `number_selections`: Number of panels to return (usually 1)

**Returns:**

- `success`: Whether selection succeeded within max attempts
- `selected_committees`: List of committees (frozensets of person IDs)
- `output_lines`: Debug and status messages

**Raises:**

- `InfeasibleQuotasError`: If quotas cannot be satisfied
- `SelectionError`: For various failure cases
- `ValueError`: For invalid parameters
- `RuntimeError`: If required solver is not available

**Example:**

```python
success, panels, messages = run_stratification(
    features, people, 100, Settings()
)
if success:
    selected_people = panels[0]  # frozenset of IDs
```

### find_random_sample()

Lower-level algorithm function for finding random committees.

```python
def find_random_sample(
    features: FeatureCollection,
    people: People,
    number_people_wanted: int,
    settings: Settings,
    selection_algorithm: str = "maximin",
    test_selection: bool = False,
    number_selections: int = 1,
) -> tuple[list[frozenset[str]], list[str]]:
```

**Parameters:**

- `selection_algorithm`: One of "maximin", "leximin", "nash", or "legacy"
- Other parameters same as `run_stratification()`

**Returns:**

- `committee_lottery`: List of committees (may contain duplicates)
- `output_lines`: Debug strings

**Example:**

```python
committees, messages = find_random_sample(
    features, people, 50, settings, "nash"
)
```

### selected_remaining_tables()

Format selection results for export to CSV or other formats.

```python
def selected_remaining_tables(
    full_people: People,
    people_selected: frozenset[str],
    features: FeatureCollection,
    settings: Settings,
) -> tuple[list[list[str]], list[list[str]], list[str]]:
```

**Parameters:**

- `full_people`: Original People object
- `people_selected`: Single frozenset of selected person IDs
- `features`: FeatureCollection used for selection
- `settings`: Settings object

**Returns:**

- `selected_rows`: Table with selected people data
- `remaining_rows`: Table with remaining people data
- `output_lines`: Additional information messages

**Example:**

```python
selected_table, remaining_table, info = selected_remaining_tables(
    people, selected_panel, features, settings
)

# Write to CSV
import csv
with open("selected.csv", "w", newline="") as f:
    csv.writer(f).writerows(selected_table)
```

## Data Loading Functions

### read_in_features()

Load feature definitions from a CSV file.

```python
def read_in_features(features_file: str | Path) -> FeatureCollection:
```

**Parameters:**

- `features_file`: Path to CSV file with feature definitions

**Expected CSV format:**

```csv
feature,value,min,max
Gender,Male,45,55
Gender,Female,45,55
Age,18-30,20,30
```

**Returns:**

- `FeatureCollection`: Nested dict containing all features and quotas

**Example:**

```python
features = read_in_features("demographics.csv")
```

### read_in_people()

Load candidate pool from a CSV file.

```python
def read_in_people(
    people_file: str | Path,
    settings: Settings,
    features: FeatureCollection
) -> People:
```

**Parameters:**

- `people_file`: Path to CSV file with candidate data
- `settings`: Settings object for configuration
- `features`: FeatureCollection for validation

**Expected CSV format:**

```csv
id,Name,Gender,Age,Email
p001,Alice,Female,18-30,alice@example.com
p002,Bob,Male,31-50,bob@example.com
```

**Returns:**

- `People`: Object containing candidate pool

**Example:**

```python
people = read_in_people("candidates.csv", settings, features)
```

## Settings Class

Configuration object for customizing selection behavior.

```python
class Settings:
    def __init__(
        self,
        random_number_seed: int | None = None,
        check_same_address: bool = False,
        check_same_address_columns: list[str] | None = None,
        selection_algorithm: str = "maximin",
        max_attempts: int = 10,
        columns_to_keep: list[str] | None = None,
        id_column: str = "id",
    ):
```

**Parameters:**

- `random_number_seed`: Fixed seed for reproducible results (None or 0 = random)
- `check_same_address`: Enable household diversity checking
- `check_same_address_columns`: Columns that define an address
- `selection_algorithm`: "maximin", "leximin", "nash", or "legacy"
- `max_attempts`: Maximum selection retry attempts
- `columns_to_keep`: Additional columns to include in output
- `id_column`: Name of the ID column in people data

**Class Methods:**

#### Settings.load_from_file()

```python
@classmethod
def load_from_file(
    cls,
    settings_file_path: Path
) -> tuple[Settings, str]:
```

Load settings from a TOML file.

**Example settings.toml:**

```toml
id_column = "my_id"
random_number_seed = 0
check_same_address = true
check_same_address_columns = ["Address", "Postcode"]
selection_algorithm = "maximin"
max_attempts = 10
columns_to_keep = ["Name", "Email", "Phone"]
```

**Returns:**

- `Settings`: Configured settings object
- `str`: Status message

**Example:**

```python
settings, msg = Settings.load_from_file(Path("config.toml"))
print(msg)  # "Settings loaded from config.toml"
```

## Data Adapters

### CSVAdapter

Handles CSV file input and output operations.

```python
class CSVAdapter:
    def load_features_from_file(
        self, features_file: Path
    ) -> tuple[FeatureCollection, list[str]]:

    def load_people_from_file(
        self, people_file: Path, settings: Settings, features: FeatureCollection
    ) -> tuple[People, list[str]]:

    def output_selected_remaining(
        self, selected_rows: list[list[str]], remaining_rows: list[list[str]]
    ) -> None:
```

**Example:**

```python
adapter = CSVAdapter()
features, msgs = adapter.load_features_from_file(Path("features.csv"))
people, msgs = adapter.load_people_from_file(Path("people.csv"), settings, features)

# Set output files
adapter.selected_file = open("selected.csv", "w", newline="")
adapter.remaining_file = open("remaining.csv", "w", newline="")
adapter.output_selected_remaining(selected_table, remaining_table)
```

### GSheetAdapter

Handles Google Sheets input and output operations.

```python
class GSheetAdapter:
    def __init__(
        self,
        credentials_file: Path,
        gen_rem_tab: str = "on"
    ):

    def load_features(
        self, gsheet_name: str, tab_name: str
    ) -> tuple[FeatureCollection | None, list[str]]:

    def load_people(
        self, tab_name: str, settings: Settings, features: FeatureCollection
    ) -> tuple[People | None, list[str]]:

    def output_selected_remaining(
        self, selected_rows: list[list[str]], remaining_rows: list[list[str]], settings: Settings
    ) -> None:
```

**Parameters:**

- `credentials_file`: Path to Google API credentials JSON
- `gen_rem_tab`: "on" or "off" to control remaining tab generation

**Example:**

```python
adapter = GSheetAdapter(Path("credentials.json"))
features, msgs = adapter.load_features("My Spreadsheet", "Demographics")
people, msgs = adapter.load_people("Candidates", settings, features)

adapter.selected_tab_name = "Selected"
adapter.remaining_tab_name = "Remaining"
adapter.output_selected_remaining(selected_table, remaining_table, settings)
```

## Core Data Classes

### FeatureCollection

Container for demographic features and their quotas. It is a nested dict of `FeatureValueMinMax`
objects. The outer dict keys are the feature names, and the inner dict keys are the value names.

**Key Helper Functions:**

```python
def check_desired(fc: FeatureCollection, desired_number: int) -> None:
    # Validates that quotas are achievable for the desired panel size
    # Raises exception if infeasible

def iterate_feature_collection(features: FeatureCollection) -> Generator[tuple[str, str, FeatureValueMinMax]]:
    # Iterate over all feature values and their count objects
```

### People

Container for the candidate pool.

**Key Methods:**

```python
def __len__(self) -> int:
    # Number of people in the pool

def __iter__(self) -> Iterator[str]:
    # Iterate over person IDs

def get_person_dict(self, person_id: str) -> dict[str, str]:
    # Get all data for a specific person

def matching_address(
    self, person_id: str, address_columns: list[str]
) -> list[str]:
    # Find people with matching address to given person

def remove(self, person_id: str) -> None:
    # Remove person from pool

def remove_many(self, person_ids: list[str]) -> None:
    # Remove multiple people from pool
```

## Error Classes

### InfeasibleQuotasError

Raised when quotas cannot be satisfied with the available candidate pool.

```python
class InfeasibleQuotasError(Exception):
    def __init__(self, output: list[str])
```

**Attributes:**

- `output`: List of diagnostic messages explaining the infeasibility

### SelectionError

General error for selection process failures.

```python
class SelectionError(Exception):
    pass
```

## Utility Functions

### set_random_provider()

Configure the random number generator for reproducible results.

```python
def set_random_provider(seed: int | None) -> None
```

**Parameters:**

- `seed`: Random seed (None for secure random)

**Example:**

```python
set_random_provider(42)  # Reproducible results
set_random_provider(None)  # Secure random
```

## Type Hints

Common type aliases used throughout the API:

```python
# A committee is a set of person IDs
Committee = frozenset[str]

# Selection results are lists of committees
SelectionResult = list[Committee]

# Tables are lists of rows (lists of strings)
Table = list[list[str]]
```
