# sortition-algorithms

[![Release](https://img.shields.io/github/v/release/sortitionfoundation/sortition-algorithms)](https://img.shields.io/github/v/release/sortitionfoundation/sortition-algorithms)
[![Build status](https://img.shields.io/github/actions/workflow/status/sortitionfoundation/sortition-algorithms/main.yml?branch=main)](https://github.com/sortitionfoundation/sortition-algorithms/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sortitionfoundation/sortition-algorithms/branch/main/graph/badge.svg)](https://codecov.io/gh/sortitionfoundation/sortition-algorithms)
[![Commit activity](https://img.shields.io/github/commit-activity/m/sortitionfoundation/sortition-algorithms)](https://img.shields.io/github/commit-activity/m/sortitionfoundation/sortition-algorithms)
[![License](https://img.shields.io/github/license/sortitionfoundation/sortition-algorithms)](https://img.shields.io/github/license/sortitionfoundation/sortition-algorithms)

A package containing algorithms for sortition - democratic lotteries.

- **Github repository**: <https://github.com/sortitionfoundation/sortition-algorithms/>
- **Documentation** <https://sortitionfoundation.github.io/sortition-algorithms/>

## About

Random stratified selection software. The algorithms are described in [this paper (open access)](https://www.nature.com/articles/s41586-021-03788-6).

Other relevant papers:

- Procaccia et al. [Is Sortition Both Representative and Fair?](https://procaccia.info/wp-content/uploads/2022/06/repfair.pdf)
- Tiago c Peixoto
  - [Reflections on the representativeness of citizens’ assemblies and similar innovations](https://democracyspot.net/2023/02/22/reflections-on-the-representativeness-of-citizens-assemblies-and-similar-innovations/) and
  - [How representative is it really? A correspondence on sortition](https://www.publicdeliberation.net/how-representative-is-it-really-a-correspondence-on-sortition/)

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

There is a basic command line interface as part of this package. As much as anything it is to show
you example code that will exercise the library. See the docs for more details.

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

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version

- Create an API Token on [PyPI](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/sortitionfoundation/sortition-algorithms/settings/secrets/actions/new).
- Create a [new release](https://github.com/sortitionfoundation/sortition-algorithms/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
