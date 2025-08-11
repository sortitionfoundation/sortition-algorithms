# sortition-algorithms

[![Release](https://img.shields.io/github/v/release/sortitionfoundation/sortition-algorithms)](https://img.shields.io/github/v/release/sortitionfoundation/sortition-algorithms)
[![Build status](https://img.shields.io/github/actions/workflow/status/sortitionfoundation/sortition-algorithms/main.yml?branch=main)](https://github.com/sortitionfoundation/sortition-algorithms/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sortitionfoundation/sortition-algorithms/graph/badge.svg?token=8M0KLNCMIA)](https://codecov.io/gh/sortitionfoundation/sortition-algorithms)
[![Commit activity](https://img.shields.io/github/commit-activity/m/sortitionfoundation/sortition-algorithms)](https://img.shields.io/github/commit-activity/m/sortitionfoundation/sortition-algorithms)
[![License](https://img.shields.io/github/license/sortitionfoundation/sortition-algorithms)](https://img.shields.io/github/license/sortitionfoundation/sortition-algorithms)

A package containing algorithms for sortition - democratic lotteries.

- **Github repository**: <https://github.com/sortitionfoundation/sortition-algorithms/>
- **Documentation** <https://sortitionfoundation.github.io/sortition-algorithms/>

## About

This library implements algorithms for **sortition** - the random selection of representative citizen panels (also known as citizens' assemblies, juries, or deliberative panels). Unlike simple random sampling, these algorithms use **stratified selection** to ensure the chosen panel reflects the demographic composition of the broader population.

### What is Sortition?

Sortition creates representative groups by randomly selecting people while respecting demographic quotas. For example, if your population is 52% women and 48% men, sortition ensures your panel maintains similar proportions rather than risking an all-male or all-female selection through pure chance.

### Key Features

- **Stratified Random Selection**: Respects demographic quotas while maintaining randomness
- **Household Diversity**: Optional address checking to ensure geographic and household spread
- **Multiple Algorithms**: Choose from maximin, leximin, nash, or legacy selection methods
- **Flexible Data Sources**: Works with CSV files, Google Sheets, or direct Python data structures
- **Transparency**: Detailed reporting of selection process and quota fulfillment

### Quick Example

```python
from sortition_algorithms import run_stratification, read_in_features, read_in_people, Settings

# Load your data
features = read_in_features("demographics.csv")  # Age, Gender, Location quotas
people = read_in_people("candidates.csv", Settings(), features)

# Select a representative panel of 100 people
success, selected_panels, messages = run_stratification(
    features, people, number_people_wanted=100, settings=Settings()
)

if success:
    panel = selected_panels[0]  # Set of selected person IDs
    print(f"Selected {len(panel)} people for the panel")
```

### Research Background

The algorithms are described in [this paper (open access)](https://www.nature.com/articles/s41586-021-03788-6). Other relevant papers are linked to [from the docs](https://sortitionfoundation.github.io/sortition-algorithms/concepts/#research-background)

## Installing the library

```sh
pip install sortition-algorithms
```

(Or `uv add ...` or ...)

### Optional dependencies

There are two sets of optional dependencies:

```sh
# Install the library to use the leximin algorithm
# This requires a commercial/academic license to use
pip install 'sortition-algorithms[gurobi]'

# Install the basic Command Line Interface
pip install 'sortition-algorithms[cli]'
```

## The Command Line Interface

The library includes a CLI for common operations:

```bash
# CSV workflow
python -m sortition_algorithms csv \
  --settings config.toml \
  --features-csv demographics.csv \
  --people-csv candidates.csv \
  --selected-csv selected.csv \
  --remaining-csv remaining.csv \
  --number-wanted 100
```

## Documentation

For detailed usage instructions, API reference, and advanced examples:

- **[Quick Start Guide](https://sortitionfoundation.github.io/sortition-algorithms/quickstart/)** - Get up and running quickly
- **[Core Concepts](https://sortitionfoundation.github.io/sortition-algorithms/concepts/)** - Understand sortition and stratified selection
- **[API Reference](https://sortitionfoundation.github.io/sortition-algorithms/api-reference/)** - Complete function documentation
- **[CLI Usage](https://sortitionfoundation.github.io/sortition-algorithms/cli/)** - Command line interface examples
- **[Data Adapters](https://sortitionfoundation.github.io/sortition-algorithms/adapters/)** - Working with CSV, Google Sheets, and custom data sources

## Starting Development

### Prerequisites

The recommended prerequisites are:

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [justfile](https://github.com/casey/just?tab=readme-ov-file#installation)
- [pre-commit](https://pre-commit.com/)

### Set Up

To install a virtualenv with the required dependencies and set up pre-commit hooks:

```sh
just install
```

### Get going

```sh
# run all the tests
just test

# run all the tests that aren't slow
just test

# run all the code quality checks
just check
```

The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version

- Create an API Token on [PyPI](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/sortitionfoundation/sortition-algorithms/settings/secrets/actions/new).
- Create a [new release](https://github.com/sortitionfoundation/sortition-algorithms/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
