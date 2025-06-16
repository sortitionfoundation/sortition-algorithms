# Sortition Algorithms Documentation

[![Release](https://img.shields.io/github/v/release/sortitionfoundation/sortition-algorithms)](https://img.shields.io/github/v/release/sortitionfoundation/sortition-algorithms)
[![Build status](https://img.shields.io/github/actions/workflow/status/sortitionfoundation/sortition-algorithms/main.yml?branch=main)](https://github.com/sortitionfoundation/sortition-algorithms/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/sortitionfoundation/sortition-algorithms)](https://img.shields.io/github/commit-activity/m/sortitionfoundation/sortition-algorithms)
[![License](https://img.shields.io/github/license/sortitionfoundation/sortition-algorithms)](https://img.shields.io/github/license/sortitionfoundation/sortition-algorithms)

Welcome to the documentation for **sortition-algorithms** - a Python library for democratic lotteries and stratified random selection.

## What is Sortition?

**Sortition** is the random selection of representatives from a larger population, designed to create panels that reflect the demographic composition of the whole group. Unlike simple random sampling, sortition uses **stratified random selection** to ensure demographic balance while maintaining the randomness essential for fairness.

This library provides algorithms for:

- **Citizens' Assemblies**: Representative groups for policy deliberation
- **Deliberative Polls**: Research panels reflecting population diversity
- **Jury Selection**: Fair selection respecting demographic quotas
- **Participatory Democracy**: Community engagement with guaranteed representation

## Quick Start

```bash
# Install the library
pip install sortition-algorithms

# Basic selection
python -m sortition_algorithms csv \
  --settings config.toml \
  --features-csv demographics.csv \
  --people-csv candidates.csv \
  --selected-csv selected.csv \
  --remaining-csv remaining.csv \
  --number-wanted 100
```

## Documentation Guide

### Getting Started

- **[Quick Start Guide](quickstart.md)** - Get up and running in minutes with practical examples
- **[Core Concepts](concepts.md)** - Understand sortition, features, quotas, and address checking
- **[Installation & Setup](quickstart.md#installation)** - Install the library and optional dependencies

### Using the Library

- **[CLI Usage](cli.md)** - Command line interface for common operations
- **[Data Adapters](adapters.md)** - Working with CSV, Google Sheets, and custom data sources
- **[API Reference](api-reference.md)** - Extended documentation of key functions and classes
- **[Modules](modules.md)** - Complete documentation of all functions and classes

### Advanced Topics

- **[Advanced Usage](advanced.md)** - Performance optimization, complex scenarios, and troubleshooting
- **[Algorithm Deep Dive](advanced.md#algorithm-deep-dive)** - Understanding maximin, nash, and leximin algorithms

## Key Features

### 🎯 **Stratified Selection**

Ensures demographic representativeness while maintaining randomness - no more accidentally all-male or all-young panels.

### 🏠 **Household Diversity**

Optional address checking prevents multiple selections from the same household, ensuring geographic and social diversity.

### ⚖️ **Multiple Algorithms**

Choose from maximin (default), nash, leximin, or legacy algorithms based on your fairness requirements.

### 📊 **Flexible Data Sources**

Works seamlessly with CSV files, Google Sheets, or custom data adapters for databases and APIs.

### 🔍 **Full Transparency**

Detailed reporting shows exactly how quotas were met and provides audit trails for democratic accountability.

## Common Use Cases

### Academic Research

```python
from sortition_algorithms import run_stratification, Settings

# Reproducible results for research
settings = Settings(
    random_number_seed=42,
    selection_algorithm="leximin"  # Strongest fairness guarantees
)
success, panels, msgs = run_stratification(features, people, 150, settings)
```

### Citizen Assemblies

```python
# Ensure household diversity for community representation
settings = Settings(
    check_same_address=True,
    check_same_address_columns=["Address", "Postcode"],
    selection_algorithm="maximin"
)
```

### Large-Scale Surveys

```bash
# Batch processing with CLI
python -m sortition_algorithms csv \
  --features-csv national_demographics.csv \
  --people-csv voter_registry.csv \
  --number-wanted 2000 \
  --settings survey_config.toml
```

## Algorithm Comparison

| Algorithm   | Best For                        | Strengths                       | Requirements   |
| ----------- | ------------------------------- | ------------------------------- | -------------- |
| **Maximin** | General use, citizen assemblies | Fair to minorities, intuitive   | None           |
| **Nash**    | Large diverse pools             | Balanced overall representation | None           |
| **Leximin** | Academic research               | Strongest fairness guarantees   | Gurobi license |
| **Legacy**  | Historical compatibility        | Backwards compatible            | None           |

Read [more about the algorithms](concepts.md#selection-algorithms).

## Real-World Applications

### Government & Democracy

- **Ireland's Citizens' Assembly**: Used sortition for constitutional reform discussions
- **French Citizens' Convention**: 150 citizens selected to address climate change
- **UK Citizens' Assemblies**: Local and national policy deliberation

### Research & Academia

- **Deliberative Polling**: Stanford's Center for Deliberative Democracy
- **Policy Research**: Representative samples for social science studies
- **Market Research**: Demographically balanced focus groups

### Community Engagement

- **Participatory Budgeting**: Community members deciding local spending
- **Planning Consultations**: Representative input on development projects
- **Local Government**: Advisory panels for municipal decisions

## Support & Community

### Getting Help

- **[Troubleshooting Guide](advanced.md#troubleshooting-guide)** - Solutions to common problems
- **[GitHub Issues](https://github.com/sortitionfoundation/sortition-algorithms/issues)** - Report bugs or request features
- **[Discussion Forum](https://github.com/sortitionfoundation/sortition-algorithms/discussions)** - Community support and questions

### Contributing

- **[Contributing Guide](https://github.com/sortitionfoundation/sortition-algorithms/blob/main/CONTRIBUTING.md)** - How to contribute to the project
- **[Development Setup](https://github.com/sortitionfoundation/sortition-algorithms#starting-development)** - Set up your development environment

### Research & Citations

- **[Core Paper](https://www.nature.com/articles/s41586-021-03788-6)** - Academic foundation for the algorithms
- **[Related Research](concepts.md#research-background)** - Additional academic resources

## License & Usage

This library is open source under the GPL License. You're free to use it for:

- ✅ Academic research and education
- ✅ Government and civic applications
- ✅ Commercial projects and consulting
- ✅ Community organizing and activism

**Note**: The leximin algorithm requires Gurobi, which has commercial licensing requirements. All other algorithms use open-source solvers.
