# Benchmarking

The sortition-algorithms package includes profiling tools for measuring solver performance across different backends and algorithms.

## Solver Backends

The library supports two solver backends:

| Backend   | Package  | Description                                     |
| --------- | -------- | ----------------------------------------------- |
| `highspy` | Built-in | HiGHS solver via highspy (default, recommended) |
| `mip`     | Optional | python-mip package (`pip install mip`)          |

Configure the backend in your settings:

```toml
# settings.toml
solver_backend = "highspy"  # or "mip"
```

Or programmatically:

```python
from sortition_algorithms.settings import Settings

settings = Settings(
    id_column="id",
    columns_to_keep=["name"],
    solver_backend="highspy",  # or "mip"
)
```

## Running Benchmarks

The benchmarking tools are located in the `benchmarks/` directory of the repository.

### Quick Start

```bash
# Clone the repository
git clone https://github.com/sortitionfoundation/sortition-algorithms
cd sortition-algorithms

# Run basic profiling
uv run python -m benchmarks.profile_solvers

# Compare backends (requires mip package)
uv run python -m benchmarks.profile_solvers --backends highspy mip

# Profile specific algorithms
uv run python -m benchmarks.profile_solvers --algorithms maximin nash diversimax
```

### Profiling Options

| Option           | Default                   | Description                                       |
| ---------------- | ------------------------- | ------------------------------------------------- |
| `--backends`     | `highspy`                 | Solver backends to test                           |
| `--algorithms`   | `maximin nash diversimax` | Algorithms to profile                             |
| `--sizes`        | `150`                     | Pool sizes to test (ignored with `--people-csv`)  |
| `--panel-size`   | auto (~15%)               | Panel size to select                              |
| `--runs`         | `3`                       | Runs per configuration                            |
| `--seed`         | `42`                      | Random seed for reproducibility                   |
| `--people-csv`   | -                         | Path to existing people CSV                       |
| `--features-csv` | -                         | Path to existing features CSV                     |
| `--settings`     | -                         | Path to settings TOML file                        |

### Using Existing Datasets

You can benchmark with your own datasets:

```bash
uv run python -m benchmarks.profile_solvers \
    --people-csv data/candidates.csv \
    --features-csv data/features.csv \
    --settings data/settings.toml \
    --panel-size 30 \
    --backends highspy mip
```

All three options (`--people-csv`, `--features-csv`, `--settings`) must be provided together.

### Example Output

```
Sortition Algorithm Solver Profiling
========================================
Backends: ['highspy']
Algorithms: ['maximin', 'nash', 'diversimax']
Pool sizes: [150]
Panel size: 22
Runs per config: 3

============================================================
Loading fixtures for 150 people...
Loaded 150 people with 4 features

  Profiling maximin with highspy backend...
    Run 1: 2.716s, 0.5MB - OK
    Run 2: 2.698s, 0.5MB - OK
    Run 3: 2.721s, 0.5MB - OK

================================================================================
PROFILING SUMMARY
================================================================================

Backend    Algorithm    People   Runs   Mean Time    Max Memory
----------------------------------------------------------------------
highspy    maximin      150      3      2.712s       0.5MB
highspy    nash         150      3      1.234s       0.4MB
highspy    diversimax   150      3      0.567s       0.3MB
```

## Memory Profiling with Memray

For detailed memory analysis, use memray:

```bash
# Run memray profiling (generates .bin file)
uv run memray run -m benchmarks.memray_profile

# Generate flamegraph
uv run memray flamegraph memray-*.bin -o benchmarks/results/flamegraph.html

# View summary
uv run memray summary memray-*.bin
```

Configure via environment variables:

```bash
PROFILE_BACKEND=mip uv run memray run -m benchmarks.memray_profile
PROFILE_ALGORITHM=nash uv run memray run -m benchmarks.memray_profile
PROFILE_SIZE=500 uv run memray run -m benchmarks.memray_profile
```

## Scaling Tests

Generate larger test fixtures for stress testing:

```bash
# Profile with larger pool sizes
uv run python -m benchmarks.profile_solvers --sizes 150 500 1000
```

The fixture generator scales the demographic distributions proportionally while maintaining similar constraint ratios.

## Anonymizing Data for Sharing

If you want to share benchmark data without exposing sensitive information, use the anonymize script:

```bash
uv run python -m benchmarks.anonymize_data \
    --people data/candidates.csv \
    --features data/features.csv \
    --settings data/settings.toml \
    --output-dir anonymized/
```

This creates anonymized versions of your files that:

- Remove all personal information (names, emails, addresses)
- Rename columns to generic names (`feature1`, `feature2`, etc.)
- Rename values to generic names (`f1value1`, `f1value2`, etc.)
- Preserve duplicate address detection (same addresses stay same)
- Maintain identical feature distributions for benchmarking

The output includes:
- `people.csv` - Anonymized people data
- `features.csv` - Features with renamed values
- `settings.toml` - Updated settings with new column names

## Metrics Collected

| Metric               | Tool                  | Description                         |
| -------------------- | --------------------- | ----------------------------------- |
| Elapsed time         | `time.perf_counter()` | Wall-clock time for the algorithm   |
| Peak memory          | `tracemalloc`         | Maximum heap usage during execution |
| Detailed allocations | `memray`              | Memory allocation flamegraph        |

## Output Files

Results are saved to `benchmarks/results/`:

- `profile_results_TIMESTAMP.csv` - Raw timing/memory data for each run
- `profile_summary_TIMESTAMP.json` - Aggregated statistics

## Choosing a Backend

**Use `highspy` (default) when:**

- You want the best out-of-box experience
- You need consistent cross-platform behaviour
- You're deploying to production

**Consider `mip` when:**

- You have an existing python-mip installation
- You need specific CBC solver features
- You're comparing solver implementations

Both backends produce identical selection results for the same random seed.
