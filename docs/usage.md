# Usage

## Typical stages

The typical stages are to load settings and features, then people. Once you have them loaded
you can run a selection (or many) and then save the selection out again.

If you want to run directly with python data structures going in and out, you
could look at the code in [test_committee_generation](https://github.com/sortitionfoundation/sortition-algorithms/blob/main/tests/test_committee_generation.py)

If you want to read and write from CSV or Google Sheets, you can use the adapters for each.
Reading the [cli code](https://github.com/sortitionfoundation/sortition-algorithms/blob/main/src/sortition_algorithms/__main__.py) would be
a good place to start.

## Terminology

- features
- feature values (sometimes just "values")

For example, Gender would be a feature, Male and Female would be values.
