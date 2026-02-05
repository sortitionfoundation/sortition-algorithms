# Benchmarks

Performance profiling tools for sortition algorithm solver backends.

For full documentation, see [docs/benchmarking.md](../docs/benchmarking.md).

## Quick Start

```bash
# Run basic profiling
uv run python -m benchmarks.profile_solvers

# Compare backends
uv run python -m benchmarks.profile_solvers --backends highspy mip

# Profile with existing dataset
uv run python -m benchmarks.profile_solvers \
    --people-csv data/candidates.csv \
    --features-csv data/features.csv \
    --settings data/settings.toml \
    --panel-size 30

# Anonymize data for sharing
uv run python -m benchmarks.anonymize_data \
    --people data/candidates.csv \
    --features data/features.csv \
    --settings data/settings.toml \
    --output-dir anonymized/
```
