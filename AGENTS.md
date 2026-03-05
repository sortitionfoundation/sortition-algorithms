# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses uv for package management and just for task running.

### Setup

```bash
just install           # Install dependencies and pre-commit hooks
```

### Testing

```bash
just test              # Run pytest with coverage
uv run python -m pytest tests/test_specific.py  # Run single test file
uv run python -m pytest -k "test_name"          # Run specific test
```

### Code Quality

```bash
just check             # Run all quality checks (pre-commit using prek, mypy, deptry)
uv tool run prek run -a  # Run formatting and linting
uv run mypy            # Type checking
uv run deptry src      # Check for unused dependencies
```

### Build & Documentation

```bash
just build             # Build wheel package
just docs              # Serve documentation locally
just docs-test         # Test documentation builds
```

## Architecture

This is a Python package for sortition algorithms (democratic lotteries). The core architecture consists of:

### Core Classes

- **`People`** (`src/sortition_algorithms/people.py`): Manages a collection of people with their data, including demographic features and metadata. Provides methods for adding/removing people and finding households based on address matching.

- **`FeatureValueCounts`** (`src/sortition_algorithms/features.py`): Tracks selection counts for individual feature values (e.g., "male" within "gender" feature), including min/max targets and current selected/remaining counts.

- **`FeatureCollection`** (`src/sortition_algorithms/features.py`): Manages stratification features (e.g., gender, age) and their allowed values with min/max selection targets. Each feature contains multiple values with associated counts and constraints. This is actually a nested dict with `FeatureValueCounts` objects as the leaf elements.

- **`PeopleFeatures`** (`src/sortition_algorithms/people_features.py`): Bridges People and Features, maintaining running totals of how many people are available/selected for each feature value.

- **`Settings`** (`src/sortition_algorithms/settings.py`): The core settings for running the selection, whether to check for addresses, what columns to keep etc.

### Data Flow

1. Features are loaded from CSV with min/max targets for each value
2. People data is loaded and validated against feature constraints
3. PeopleFeatures tracks running counts as people are selected/removed
4. Selection algorithms can query constraints and update counts

### Key Patterns

- Uses `attrs` for data classes with type hints
- Extensive input validation with custom error classes in `errors.py`
- Separation of concerns: People handles data, Features handles constraints, PeopleFeatures handles the intersection
- Helper utilities in `utils.py` for data cleaning (StrippedDict)

### Testing

Tests are organized by module with comprehensive coverage including error conditions and edge cases.

### Details Docs

Detailed docs are in the docs/ folder - read files from there as appropriate for the task you are doing. And don't forget to keep those docs up-to-date.
