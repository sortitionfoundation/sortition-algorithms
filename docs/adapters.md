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
features, report = adapter.load_features_from_file(Path("demographics.csv"))
people, report = adapter.load_people_from_file(Path("candidates.csv"), Settings(), features)

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

features, report = adapter.load_features_from_str(features_csv)
people, report = adapter.load_people_from_str(people_csv, settings, features)
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
    features, report = adapter.load_features_from_file(Path("demographics.csv"))
    print(report.as_text())

    people, report = adapter.load_people_from_file(Path("candidates.csv"), settings, features)
    print(report.as_text())

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
adapter.set_g_sheet_name("My Spreadsheet")
features, report = adapter.load_features("Demographics")
print(report.as_text())

people, report = adapter.load_people("Candidates", settings, features)
print(report.as_text())

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
    adapter.set_g_sheet_name("Citizen Panel 2024")
    features, report = adapter.load_features("Demographics")
    if features is None:
        print("Failed to load features:", "\n".join(msgs))
        return

    people, report = adapter.load_people("Candidates", settings, features)
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

Note that you can have other columns on the tab - the features import code will ignore them.

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
from sortition_algorithms import FeatureCollection, People, RunReport, Settings
from typing import Protocol

class SortitionAdapter(Protocol):
    """Protocol defining the adapter interface."""

    def load_features(self, source_info: str, **kwargs) -> tuple[FeatureCollection, RunReport]:
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
    ) -> tuple[People, RunReport]:
        """Load candidate pool from data source.

        Returns:
            (people, report) - people object and report with messages
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

from sortition_algorithms import FeatureCollection, People, RunReport, Settings
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
    ) -> FeatureCollection:
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

        features = read_in_features(headers, data)
        return features

    def load_people_from_file(
        self,
        excel_file: Path,
        settings: Settings,
        features: FeatureCollection,
        sheet_name: str = "Candidates"
    ) -> tuple[People, RunReport]:
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

        people, report = read_in_people(headers, data, features, settings)
        return people, report

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
    features = adapter.load_features_from_file(
        Path("selection_data.xlsx"), "Demographics"
    )
    people, report = adapter.load_people_from_file(
        Path("selection_data.xlsx"), settings, features, "Candidates"
    )

    # Run selection (assuming you have the selection logic)
    # success, panels, msgs = run_stratification(...)

    # Export results
    # adapter.output_selected_remaining(
    #     selected_table, remaining_table, Path("results.xlsx")
    # )
```

## Next Steps

- **[Core Concepts](concepts.md)** - Understand sortition fundamentals
- **[API Reference](api-reference.md)** - Complete function documentation
- **[CLI Usage](cli.md)** - Command line interface
- **[Advanced Usage](advanced.md)** - Complex scenarios and optimization
