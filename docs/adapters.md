# Data Adapters

Data adapters handle loading demographic data and candidate pools from various sources, and exporting selection results back to those sources. The library includes adapters for CSV files and Google Sheets, and you can write custom adapters for other data sources.

## Built-in Data Sources

### CSVFileDataSource

The most commonly used adapter for working with local CSV files.

#### Basic Usage

```python
from sortition_algorithms import CSVFileDataSource, SelectionData, Settings
from pathlib import Path

data_source = CSVFileDataSource(
    Path("demographics.csv"),
    Path("candidates.csv"),
    Path("selected.csv"),
    Path("remaining.csv"),
)
select_data = SelectionData(data_source)

# Load data
features, report = select_data.load_features()
people, report = select_data.load_people(Settings(), features)

# Do Selection
# ...

# Export results (after running selection)
data_source.output_selected_remaining(selected_rows, remaining_rows)
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

data_source = CSVStringDataSource(features_csv, people_csv)
select_data = SelectionData(data_source)

features, report = select_data.load_features()
people, report = select_data.load_people(Settings(), features)
```

#### Full CSV Workflow Example

```python
from sortition_algorithms import CSVFileDataSource, run_stratification, selected_remaining_tables, SelectionData, Settings
from pathlib import Path
import csv

def csv_selection_workflow():
    # Initialize
    data_source = CSVFileDataSource(
        Path("demographics.csv"),
        Path("candidates.csv"),
        Path("selected.csv"),
        Path("remaining.csv"),
    )
    select_data = SelectionData(data_source)
    settings = Settings()

    # Load data
    features, report = select_data.load_features()
    print(report.as_text())
    people, report = select_data.load_people(Settings(), features)
    print(report.as_text())

    # Run selection
    success, panels, msgs = run_stratification(features, people, 100, settings)

    if success:
        # Format results
        selected_table, remaining_table, _ = selected_remaining_tables(
            people, panels[0], features, settings
        )

        # Export results
        data_source.output_selected_remaining(selected_rows, remaining_rows)
        print(f"Selected {len(panels[0])} people successfully")
    else:
        print("Selection failed")
        print("\n".join(msgs))
```

### GSheetDataSource

For organizations using Google Sheets for data management.

#### Setup Requirements

1. **Google Cloud Project**: Create a project in [Google Cloud Console](https://console.cloud.google.com/)
2. **Enable APIs**: Enable Google Sheets API and Google Drive API
3. **Service Account**: Create service account credentials and download JSON key
4. **Share Spreadsheet**: Share your spreadsheet with the service account email address

#### Basic Usage

```python
from sortition_algorithms import GSheetDataSource, SelectionData, Settings
from pathlib import Path

# Initialize with credentials
data_source = GSheetDataSource(
    feature_tab_name="Demographics",
    people_tab_name="Candidates",
    auth_json_path=Path("/secure/path/credentials.json"),
    gen_rem_tab=True,  # Generate remaining tab
)
data_source.set_g_sheet_name("My Spreadsheet")
select_data = SelectionData(data_source)

# Load data from Google Sheet
features, report = select_data.load_features()
print(report.as_text())

people, report = select_data.load_people(settings, features)
print(report.as_text())

# Configure output tabs
data_source.selected_tab_name_stub = "Selected Panel"
data_source.remaining_tab_name_stub = "Reserve Pool"

# Export results (after running selection)
select_data.output_selected_remaining(selected_rows, remaining_rows, settings)
```

#### Full Google Sheets Workflow

```python
from sortition_algorithms import GSheetAdapter, run_stratification, selected_remaining_tables, Settings
from pathlib import Path

def gsheet_selection_workflow():
    # Initialize
    adapter = GSheetAdapter(
        auth_json_path=Path("credentials.json"),
        gen_rem_tab=True,
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
    success, panels, report = run_stratification(features, people, 120, settings)

    if success:
        # Format results
        selected_table, remaining_table, _ = selected_remaining_tables(
            people, panels[0], features, settings
        )

        # Configure output
        adapter.selected_tab_name = "Selected Panel"
        adapter.remaining_tab_name = "Reserve Pool"

        # Export to Google Sheets
        dupes, _ = adapter.output_selected_remaining(selected_table, remaining_table, settings)

        print(f"Selected {len(panels[0])} people successfully")
        if dupes:
            print(f"Warning: {len(dupes)} people in remaining pool share addresses")
    else:
        print("Selection failed:", report.as_text())
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

## Writing custom Data Source classes

You can create custom data source classes for other data sources like Excel files, SQL databases, or APIs.

### AbstractDataSource

All data source classes should inherit from `AbstractDataSource` - and implement the methods it defines:

```python
from sortition_algorithms import RunReport

class AbstractDataSource(abc.ABC):
    @abc.abstractmethod
    @contextmanager
    def read_feature_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]: ...

    @abc.abstractmethod
    @contextmanager
    def read_people_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]: ...

    @abc.abstractmethod
    def write_selected(self, selected: list[list[str]]) -> None: ...

    @abc.abstractmethod
    def write_remaining(self, remaining: list[list[str]]) -> None: ...

    @abc.abstractmethod
    def highlight_dupes(self, dupes: list[int]) -> None: ...

```

### Example: Excel Data Source

Here's a complete example of an Excel adapter using the `openpyxl` library:

```python
from pathlib import Path
from typing import Any
import openpyxl
from openpyxl.worksheet.worksheet import Worksheet

from sortition_algorithms import AbstractDataSource, FeatureCollection, People, RunReport, SelectionError, Settings
from sortition_algorithms.features import read_in_features
from sortition_algorithms.people import read_in_people

class ExcelDataSource(AbstractDataSource):
    """DataSource for Excel files using openpyxl."""

    def __init__(self, excel_file: Path, feature_tab_name: str, people_tab_name: str) -> None:
        self.excel_file = excel_file
        self.feature_tab_name = feature_tab_name
        self.people_tab_name = people_tab_name
        self.selected_tab_name = "selected"
        self.remaining_tab_name = "remaining"

    @contextmanager
    def read_feature_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        """Load features data from Excel file."""
        workbook = openpyxl.load_workbook(self.excel_file)

        if self.feature_tab_name not in workbook.sheetnames:
            msg = f"Sheet '{self.feature_tab_name}' not found in {excel_file}"
            report.add_line(msg)
            raise SelectionError(msg, report)

        sheet = workbook[self.feature_tab_name]

        # Read header row
        headers = [cell.value for cell in sheet[1]]

        # Read data rows
        data = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if any(cell is not None for cell in row):  # Skip empty rows
                row_dict = {headers[i]: str(row[i]) if row[i] is not None else ""
                           for i in range(len(headers))}
                data.append(row_dict)
        yield headers, data
        # close the workbook

    @contextmanager
    def read_people_data(
        self, report: RunReport
    ) -> Generator[tuple[Iterable[str], Iterable[dict[str, str]]], None, None]:
        """Load people from Excel file."""
        workbook = openpyxl.load_workbook(excel_file)

        if self.people_tab_name not in workbook.sheetnames:
            msg = f"Sheet '{self.people_tab_name}' not found in {excel_file}"
            report.add_line(msg)
            raise SelectionError(msg, report)

        sheet = workbook[self.people_tab_name]

        # Read header row
        headers = [cell.value for cell in sheet[1]]

        # Read data rows
        data = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if any(cell is not None for cell in row):  # Skip empty rows
                row_dict = {headers[i]: str(row[i]) if row[i] is not None else ""
                           for i in range(len(headers))}
                data.append(row_dict)

        yield headers, data

    def write_selected(self, selected: list[list[str]]) -> None:
        selected_ws = workbook.create_sheet(self.selected_tab_name)
        self._write_data_to_sheet(selected_ws, selected_rows)
        workbook.save(self.excel_file)

    def write_remaining(self, remaining: list[list[str]]) -> None:
        remaining_ws = workbook.create_sheet(self.remaining_tab_name)
        self._write_data_to_sheet(remaining_ws, remaining_rows)
        workbook.save(self.excel_file)

    def highlight_dupes(self, dupes: list[int]) -> None:
        # not implemented
        pass

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
    data_source = ExcelDataSource(
        Path("selection_data.xlsx"),
        "Demographics",
        "Candidates",
    )
    select_data = SelectionData(data_source)
    settings = Settings()

    # Load data
    features, _ = select_data.load_features()
    people, report = select_data.load_people(settings, features)

    # Run selection (assuming you have the selection logic)
    success, panels, report = run_stratification(...)

    # Export results
    select_data.output_selected_remaining(selected_table, remaining_table, settings)
```

## Next Steps

- **[Core Concepts](concepts.md)** - Understand sortition fundamentals
- **[API Reference](api-reference.md)** - Complete function documentation
- **[CLI Usage](cli.md)** - Command line interface
- **[Advanced Usage](advanced.md)** - Complex scenarios and optimization
