# Benchmarks

Performance profiling tools for sortition algorithm solver backends.

## Quick Start

```bash
# Run basic profiling with highspy backend
uv run python benchmarks/profile_solvers.py

# Compare highspy and mip backends (requires mip package)
uv run python benchmarks/profile_solvers.py --backends highspy mip

# Profile specific algorithms
uv run python benchmarks/profile_solvers.py --algorithms maximin nash

# Profile with larger pool sizes
uv run python benchmarks/profile_solvers.py --sizes 150 500 1000
```

## Scripts

### profile_solvers.py

Main profiling script that measures elapsed time and peak memory usage for solver operations.

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--backends` | `highspy` | Solver backends to test |
| `--algorithms` | `maximin nash diversimax` | Algorithms to test |
| `--sizes` | `150` | Pool sizes to test |
| `--panel-size` | `22` | Panel size to select |
| `--runs` | `3` | Number of runs per configuration |
| `--seed` | `42` | Base random seed |
| `--output-dir` | `benchmarks/results` | Output directory |

**Output:**

- `profile_results_TIMESTAMP.csv` - Raw results for each run
- `profile_summary_TIMESTAMP.json` - Summary statistics

### generate_fixtures.py

Generates scaled test fixtures for benchmarking larger pool sizes.

```bash
# Test fixture generation
uv run python benchmarks/generate_fixtures.py
```

### memray_profile.py

Wrapper for detailed memory profiling with memray.

```bash
# Run memray profiling
uv run memray run benchmarks/memray_profile.py

# Generate flamegraph
uv run memray flamegraph memray-*.bin -o benchmarks/results/flamegraph.html

# View summary
uv run memray summary memray-*.bin

# Environment variable configuration
PROFILE_BACKEND=mip uv run memray run benchmarks/memray_profile.py
PROFILE_ALGORITHM=nash uv run memray run benchmarks/memray_profile.py
PROFILE_SIZE=500 uv run memray run benchmarks/memray_profile.py
```

## Metrics

| Metric | Tool | Description |
|--------|------|-------------|
| Elapsed time | `time.perf_counter()` | Wall-clock time for algorithm |
| Peak memory | `tracemalloc` | Maximum heap usage during run |
| Detailed memory | `memray` | Memory allocation flamegraph |

## Example Output

```
Sortition Algorithm Solver Profiling
========================================
Backends: ['highspy', 'mip']
Algorithms: ['maximin', 'nash', 'diversimax']
Pool sizes: [150]
Panel size: 24
Runs per config: 3

============================================================
Loading fixtures for 150 people...
Loaded 150 people with 4 features

  Profiling maximin with highspy backend...
    Run 1: 0.234s, 45.2MB - OK
    Run 2: 0.228s, 44.8MB - OK
    Run 3: 0.231s, 45.0MB - OK

================================================================================
PROFILING SUMMARY
================================================================================

Backend    Algorithm    People   Runs   Mean Time    Max Memory
----------------------------------------------------------------------
highspy    maximin      150      3      0.231s       45.2MB
highspy    nash         150      3      0.156s       38.4MB
highspy    diversimax   150      3      0.089s       32.1MB
```

## Results Directory

The `results/` directory stores profiling output files. These are gitignored except for `.gitkeep`.
