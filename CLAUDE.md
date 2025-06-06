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
just check             # Run all quality checks (pre-commit, mypy, deptry)
uv run pre-commit run -a  # Run formatting and linting
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

- **`FeatureCollection`** (`src/sortition_algorithms/features.py`): Manages stratification features (e.g., gender, age) and their allowed values with min/max selection targets. Each feature contains multiple values with associated counts and constraints.

- **`FeatureValueCounts`** (`src/sortition_algorithms/features.py`): Tracks selection counts for individual feature values (e.g., "male" within "gender" feature), including min/max targets and current selected/remaining counts.

- **`PeopleFeatures`** (`src/sortition_algorithms/people_features.py`): Bridges People and Features, maintaining running totals of how many people are available/selected for each feature value.

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

## Strategy for migrating code

This project is producing a standalone python package from the single python file in `old_code/stratification.py`. The old code has no tests. Looking under `src/` you can see the progress I
have made so far, splitting out functionality bit-by-bit. I want you to continue the work.

My strategy so far has been to find bits of code that don't depend on anything I have yet to migrate, taking small bites as I go. I am structuring it into small classes and **not** using
inheritance, instead relying on composition. In particular the `FeaturesCollection` and `People` classes that are initially created should **not** change during the life of the process. But
the `PeopleFeatures` class makes a deep copy of both those classes, and it is allowed to change the state of the copies that it owns.

Identify the next set of functions/methods that would be good to move - just a few, don't try to do it all in one go. Tell me what you plan to migrate next and how you plan to do it. Once you
and I have agreed a plan you should write the new code.
