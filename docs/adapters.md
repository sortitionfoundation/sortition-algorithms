# Data Adapters

Data adapters handle loading demographic data and candidate pools from various sources, and exporting selection results back to those sources. The library includes adapters for CSV files and Google Sheets, and you can write custom adapters for other data sources.

## Built-in Adapters

### CSVAdapter

The most commonly used adapter for working with local CSV files.

#### Basic Usage

```python
from sortition_algorithms import CSVAdapter, Settings
from pathlib import Path

adapter = CSVAdapter()

# Load data
features, msgs = adapter.load_features_from_file(Path("demographics.csv"))
people, msgs = adapter.load_people_from_file(Path("candidates.csv"), Settings(), features)

# Configure output files
adapter.selected_file = open("selected.csv", "w", newline="")
adapter.remaining_file = open("remaining.csv", "w", newline="")

# Export results (after running selection)
adapter.output_selected_remaining(selected_rows, remaining_rows)

# Clean up
adapter.selected_file.close()
adapter.remaining_file.close()
```

#### Working with String Data

For data already in memory:

```python
# Load from string content
features_csv = """feature,value,min,max
Gender,Male,45,55
Gender,Female,45,55"""

people_csv = """id,Name,Gender
p001,Alice,Female
p002,Bob,Male"""

features, msgs = adapter.load_features_from_str(features_csv)
people, msgs = adapter.load_people_from_str(people_csv, settings, features)
```

#### Full CSV Workflow Example

```python
from sortition_algorithms import CSVAdapter, run_stratification, selected_remaining_tables, Settings
from pathlib import Path
import csv

def csv_selection_workflow():
    # Initialize
    adapter = CSVAdapter()
    settings = Settings()

    # Load data
    features, msgs = adapter.load_features_from_file(Path("demographics.csv"))
    print("\n".join(msgs))

    people, msgs = adapter.load_people_from_file(Path("candidates.csv"), settings, features)
    print("\n".join(msgs))

    # Run selection
    success, panels, msgs = run_stratification(features, people, 100, settings)

    if success:
        # Format results
        selected_table, remaining_table, _ = selected_remaining_tables(
            people, panels[0], features, settings
        )

        # Export results
        with open("selected.csv", "w", newline="") as selected_f, \\
             open("remaining.csv", "w", newline="") as remaining_f:

            adapter.selected_file = selected_f
            adapter.remaining_file = remaining_f
            adapter.output_selected_remaining(selected_table, remaining_table)

        print(f"Selected {len(panels[0])} people successfully")
    else:
        print("Selection failed")
        print("\n".join(msgs))
```

### GSheetAdapter

For organizations using Google Sheets for data management.

#### Setup Requirements

1. **Google Cloud Project**: Create a project in [Google Cloud Console](https://console.cloud.google.com/)
2. **Enable APIs**: Enable Google Sheets API and Google Drive API
3. **Service Account**: Create service account credentials and download JSON key
4. **Share Spreadsheet**: Share your spreadsheet with the service account email address

#### Basic Usage

```python
from sortition_algorithms import GSheetAdapter, Settings
from pathlib import Path

# Initialize with credentials
adapter = GSheetAdapter(
    auth_json_path=Path("/secure/path/credentials.json"),
    gen_rem_tab="on"  # Generate remaining tab
)

# Load data from Google Sheet
features, msgs = adapter.load_features("My Spreadsheet", "Demographics")
print("\n".join(msgs))

people, msgs = adapter.load_people("Candidates", settings, features)
print("\n".join(msgs))

# Configure output tabs
adapter.selected_tab_name = "Selected Panel"
adapter.remaining_tab_name = "Reserve Pool"

# Export results (after running selection)
adapter.output_selected_remaining(selected_rows, remaining_rows, settings)
```

#### Full Google Sheets Workflow

```python
from sortition_algorithms import GSheetAdapter, run_stratification, selected_remaining_tables, Settings
from pathlib import Path

def gsheet_selection_workflow():
    # Initialize
    adapter = GSheetAdapter(
        auth_json_path=Path("credentials.json"),
        gen_rem_tab="on"
    )
    settings = Settings()

    # Load data
    features, msgs = adapter.load_features("Citizen Panel 2024", "Demographics")
    if features is None:
        print("Failed to load features:", "\n".join(msgs))
        return

    people, msgs = adapter.load_people("Candidates", settings, features)
    if people is None:
        print("Failed to load people:", "\n".join(msgs))
        return

    # Run selection
    success, panels, msgs = run_stratification(features, people, 120, settings)

    if success:
        # Format results
        selected_table, remaining_table, _ = selected_remaining_tables(
            people, panels[0], features, settings
        )

        # Configure output
        adapter.selected_tab_name = "Selected Panel"
        adapter.remaining_tab_name = "Reserve Pool"

        # Export to Google Sheets
        dupes = adapter.output_selected_remaining(selected_table, remaining_table, settings)

        print(f"Selected {len(panels[0])} people successfully")
        if dupes:
            print(f"Warning: {len(dupes)} people in remaining pool share addresses")
    else:
        print("Selection failed:", "\n".join(msgs))
```

#### Google Sheets Data Format

Your spreadsheet should be structured as follows:

**Demographics Tab:**

| feature | value  | min | max |
| ------- | ------ | --- | --- |
| Gender  | Male   | 45  | 55  |
| Gender  | Female | 45  | 55  |
| Age     | 18-30  | 20  | 30  |

**Candidates Tab:**

| id   | Name        | Email             | Gender | Age   | Location | Address     | Postcode |
| ---- | ----------- | ----------------- | ------ | ----- | -------- | ----------- | -------- |
| p001 | Alice Smith | <alice@email.com> | Female | 18-30 | Urban    | 123 Main St | 12345    |
| p002 | Bob Jones   | <bob@email.com>   | Male   | 31-50 | Rural    | 456 Oak Ave | 67890    |

## Writing Custom Adapters

You can create custom adapters for other data sources like Excel files, SQL databases, or APIs.

### Adapter Interface Pattern

All adapters should implement these core methods:

```python
from sortition_algorithms import FeatureCollection, People, Settings
from typing import Protocol

class SortitionAdapter(Protocol):
    """Protocol defining the adapter interface."""

    def load_features(self, source_info: str, **kwargs) -> tuple[FeatureCollection, list[str]]:
        """Load feature definitions from data source.

        Returns:
            (features, messages) - features object and status messages
        """
        ...

    def load_people(
        self,
        source_info: str,
        settings: Settings,
        features: FeatureCollection,
        **kwargs
    ) -> tuple[People, list[str]]:
        """Load candidate pool from data source.

        Returns:
            (people, messages) - people object and status messages
        """
        ...

    def output_selected_remaining(
        self,
        selected_rows: list[list[str]],
        remaining_rows: list[list[str]],
        **kwargs
    ) -> None:
        """Export selection results to data source."""
        ...
```

### Example: Excel Adapter

Here's a complete example of an Excel adapter using the `openpyxl` library:

```python
from pathlib import Path
from typing import Any
import openpyxl
from openpyxl.worksheet.worksheet import Worksheet

from sortition_algorithms import FeatureCollection, People, Settings
from sortition_algorithms.features import read_in_features
from sortition_algorithms.people import read_in_people

class ExcelAdapter:
    """Adapter for Excel files using openpyxl."""

    def __init__(self) -> None:
        self.workbook: openpyxl.Workbook | None = None
        self.output_file: Path | None = None

    def load_features_from_file(
        self,
        excel_file: Path,
        sheet_name: str = "Demographics"
    ) -> tuple[FeatureCollection, list[str]]:
        """Load features from Excel file."""
        workbook = openpyxl.load_workbook(excel_file)

        if sheet_name not in workbook.sheetnames:
            return None, [f"Sheet '{sheet_name}' not found in {excel_file}"]

        sheet = workbook[sheet_name]

        # Read header row
        headers = [cell.value for cell in sheet[1]]

        # Read data rows
        data = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if any(cell is not None for cell in row):  # Skip empty rows
                row_dict = {headers[i]: str(row[i]) if row[i] is not None else ""
                           for i in range(len(headers))}
                data.append(row_dict)

        features, msgs = read_in_features(headers, data)
        return features, msgs

    def load_people_from_file(
        self,
        excel_file: Path,
        settings: Settings,
        features: FeatureCollection,
        sheet_name: str = "Candidates"
    ) -> tuple[People, list[str]]:
        """Load people from Excel file."""
        workbook = openpyxl.load_workbook(excel_file)

        if sheet_name not in workbook.sheetnames:
            return None, [f"Sheet '{sheet_name}' not found in {excel_file}"]

        sheet = workbook[sheet_name]

        # Read header row
        headers = [cell.value for cell in sheet[1]]

        # Read data rows
        data = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if any(cell is not None for cell in row):  # Skip empty rows
                row_dict = {headers[i]: str(row[i]) if row[i] is not None else ""
                           for i in range(len(headers))}
                data.append(row_dict)

        people, msgs = read_in_people(headers, data, features, settings)
        return people, msgs

    def output_selected_remaining(
        self,
        selected_rows: list[list[str]],
        remaining_rows: list[list[str]],
        output_file: Path,
        selected_sheet: str = "Selected",
        remaining_sheet: str = "Remaining"
    ) -> None:
        """Export results to Excel file."""
        workbook = openpyxl.Workbook()

        # Remove default sheet
        workbook.remove(workbook.active)

        # Create selected sheet
        selected_ws = workbook.create_sheet(selected_sheet)
        self._write_data_to_sheet(selected_ws, selected_rows)

        # Create remaining sheet
        remaining_ws = workbook.create_sheet(remaining_sheet)
        self._write_data_to_sheet(remaining_ws, remaining_rows)

        # Save workbook
        workbook.save(output_file)

    def _write_data_to_sheet(self, sheet: Worksheet, data: list[list[str]]) -> None:
        """Write data rows to worksheet."""
        for row_idx, row_data in enumerate(data, 1):
            for col_idx, cell_value in enumerate(row_data, 1):
                sheet.cell(row=row_idx, column=col_idx, value=cell_value)

        # Style header row
        if data:
            for cell in sheet[1]:
                cell.font = openpyxl.styles.Font(bold=True)
                cell.fill = openpyxl.styles.PatternFill("solid", fgColor="CCCCCC")

# Usage example
def excel_workflow():
    adapter = ExcelAdapter()
    settings = Settings()

    # Load data
    features, msgs = adapter.load_features_from_file(
        Path("selection_data.xlsx"), "Demographics"
    )
    people, msgs = adapter.load_people_from_file(
        Path("selection_data.xlsx"), settings, features, "Candidates"
    )

    # Run selection (assuming you have the selection logic)
    # success, panels, msgs = run_stratification(...)

    # Export results
    # adapter.output_selected_remaining(
    #     selected_table, remaining_table, Path("results.xlsx")
    # )
```

### Example: SQL Database Adapter

For larger datasets stored in databases:

```python
import sqlite3
from pathlib import Path
from typing import Any

from sortition_algorithms import FeatureCollection, People, Settings
from sortition_algorithms.features import read_in_features
from sortition_algorithms.people import read_in_people

class SQLiteAdapter:
    """Adapter for SQLite databases."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def load_features_from_db(
        self,
        table_name: str = "features"
    ) -> tuple[FeatureCollection, list[str]]:
        """Load features from database table."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()

            if not rows:
                return None, [f"No data found in table {table_name}"]

            # Convert to format expected by read_in_features
            headers = list(rows[0].keys())
            data = [{col: str(row[col]) for col in headers} for row in rows]

            features, msgs = read_in_features(headers, data)
            return features, msgs

        finally:
            conn.close()

    def load_people_from_db(
        self,
        settings: Settings,
        features: FeatureCollection,
        table_name: str = "candidates",
        where_clause: str = ""
    ) -> tuple[People, list[str]]:
        """Load people from database table."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            query = f"SELECT * FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"

            cursor = conn.execute(query)
            rows = cursor.fetchall()

            if not rows:
                return None, [f"No candidates found in table {table_name}"]

            # Convert to format expected by read_in_people
            headers = list(rows[0].keys())
            data = [{col: str(row[col]) for col in headers} for row in rows]

            people, msgs = read_in_people(headers, data, features, settings)
            return people, msgs

        finally:
            conn.close()

    def output_selected_remaining(
        self,
        selected_rows: list[list[str]],
        remaining_rows: list[list[str]],
        selected_table: str = "selected_panel",
        remaining_table: str = "remaining_pool"
    ) -> None:
        """Export results to database tables."""
        conn = sqlite3.connect(self.db_path)

        try:
            # Create tables if they don't exist
            if selected_rows:
                headers = selected_rows[0]
                self._create_table_from_headers(conn, selected_table, headers)
                self._insert_data(conn, selected_table, headers, selected_rows[1:])

            if remaining_rows:
                headers = remaining_rows[0]
                self._create_table_from_headers(conn, remaining_table, headers)
                self._insert_data(conn, remaining_table, headers, remaining_rows[1:])

            conn.commit()

        finally:
            conn.close()

    def _create_table_from_headers(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        headers: list[str]
    ) -> None:
        """Create table with columns based on headers."""
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        columns = ", ".join(f"{header} TEXT" for header in headers)
        conn.execute(f"CREATE TABLE {table_name} ({columns})")

    def _insert_data(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        headers: list[str],
        data_rows: list[list[str]]
    ) -> None:
        """Insert data rows into table."""
        placeholders = ", ".join("?" * len(headers))
        query = f"INSERT INTO {table_name} VALUES ({placeholders})"

        conn.executemany(query, data_rows)

# Usage example
def sqlite_workflow():
    adapter = SQLiteAdapter(Path("selection.db"))
    settings = Settings()

    # Load data
    features, msgs = adapter.load_features_from_db("demographics")
    people, msgs = adapter.load_people_from_db(
        settings, features, "candidates", "status = 'eligible'"
    )

    # Export results
    # adapter.output_selected_remaining(
    #     selected_table, remaining_table, "panel_2024", "reserves_2024"
    # )
```

## Advanced Adapter Patterns

### Caching Adapter

For expensive data sources, implement caching:

```python
import pickle
from pathlib import Path
from datetime import datetime, timedelta

class CachingAdapter:
    """Wrapper that adds caching to any adapter."""

    def __init__(self, base_adapter: Any, cache_dir: Path, cache_ttl_hours: int = 24):
        self.base_adapter = base_adapter
        self.cache_dir = cache_dir
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.cache_dir.mkdir(exist_ok=True)

    def load_features(self, *args, **kwargs) -> tuple[FeatureCollection, list[str]]:
        cache_key = f"features_{hash(str(args) + str(kwargs))}.pickle"
        cache_file = self.cache_dir / cache_key

        # Check cache
        if cache_file.exists():
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time < self.cache_ttl:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        # Load fresh data
        result = self.base_adapter.load_features(*args, **kwargs)

        # Cache result
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        return result
```

### Validation Adapter

Add data validation to any adapter:

```python
class ValidatingAdapter:
    """Wrapper that adds validation to any adapter."""

    def __init__(self, base_adapter: Any):
        self.base_adapter = base_adapter

    def load_people(self, *args, **kwargs) -> tuple[People, list[str]]:
        people, msgs = self.base_adapter.load_people(*args, **kwargs)

        # Add validation
        validation_msgs = self._validate_people(people)
        msgs.extend(validation_msgs)

        return people, msgs

    def _validate_people(self, people: People) -> list[str]:
        """Validate people data quality."""
        msgs = []

        # Check for duplicate IDs
        seen_ids = set()
        for person_id in people:
            if person_id in seen_ids:
                msgs.append(f"Duplicate ID found: {person_id}")
            seen_ids.add(person_id)

        # Check for missing required fields
        required_fields = ["Name", "Email"]
        for person_id in people:
            person_data = people.get_person_dict(person_id)
            for field in required_fields:
                if not person_data.get(field):
                    msgs.append(f"Missing {field} for person {person_id}")

        return msgs
```

## Best Practices

### Error Handling

Always provide detailed error messages:

```python
def load_features_from_api(self, api_url: str) -> tuple[FeatureCollection | None, list[str]]:
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()

        data = response.json()
        features, msgs = read_in_features(data["headers"], data["rows"])
        return features, msgs

    except requests.RequestException as e:
        return None, [f"Failed to load from API: {e}"]
    except KeyError as e:
        return None, [f"Invalid API response format: missing {e}"]
    except Exception as e:
        return None, [f"Unexpected error: {e}"]
```

### Configuration

Make adapters configurable:

```python
class ConfigurableAdapter:
    def __init__(self, config: dict):
        self.config = config
        self.timeout = config.get("timeout", 30)
        self.retry_count = config.get("retries", 3)
        self.batch_size = config.get("batch_size", 1000)
```

### Testing Custom Adapters

Test your adapters with known data:

```python
def test_excel_adapter():
    adapter = ExcelAdapter()

    # Test with known data
    features, msgs = adapter.load_features_from_file(
        Path("test_data.xlsx"), "TestFeatures"
    )

    assert features is not None
    assert len(features.feature_names) > 0
    assert "Gender" in features.feature_names
```

### Resource Cleanup

Ensure proper resource cleanup:

```python
class DatabaseAdapter:
    def __enter__(self):
        self.connection = connect_to_database()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'connection'):
            self.connection.close()

# Usage
with DatabaseAdapter() as adapter:
    features, msgs = adapter.load_features(...)
    # Connection automatically closed
```

## Integration Examples

### Pandas Integration

```python
import pandas as pd

class PandasAdapter:
    """Adapter for pandas DataFrames."""

    def load_features_from_dataframe(
        self, df: pd.DataFrame
    ) -> tuple[FeatureCollection, list[str]]:
        headers = df.columns.tolist()
        data = df.to_dict('records')
        # Convert all values to strings
        data = [{k: str(v) for k, v in row.items()} for row in data]

        features, msgs = read_in_features(headers, data)
        return features, msgs
```

### AWS S3 Integration

```python
import boto3
import csv
from io import StringIO

class S3Adapter:
    """Adapter for files stored in AWS S3."""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')

    def load_features_from_s3(
        self, key: str
    ) -> tuple[FeatureCollection, list[str]]:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            content = response['Body'].read().decode('utf-8')

            # Parse CSV content
            csv_file = StringIO(content)
            reader = csv.DictReader(csv_file)

            headers = reader.fieldnames
            data = list(reader)

            features, msgs = read_in_features(headers, data)
            return features, msgs

        except Exception as e:
            return None, [f"Failed to load from S3: {e}"]
```

## Next Steps

- **[Core Concepts](concepts.md)** - Understand sortition fundamentals
- **[API Reference](api-reference.md)** - Complete function documentation
- **[CLI Usage](cli.md)** - Command line interface
- **[Advanced Usage](advanced.md)** - Complex scenarios and optimization
