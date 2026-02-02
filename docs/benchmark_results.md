# Benchmark Results

This page documents performance comparisons between solver backends for the sortition algorithms.

## Test Environment

- **Date**: January 2025
- **Platform**: Linux 6.8.0-90-generic
- **Python**: 3.11
- **Packages**: highspy (HiGHS solver), mip 1.15.0 (CBC solver)

## Maximin Algorithm: highspy vs mip

### Small Pool (150 people, panel size 22)

```bash
uv run python benchmarks/profile_solvers.py \
    --backends highspy mip \
    --algorithms maximin \
    --runs 5
```

| Backend | Mean Time | Max Memory |
|---------|-----------|------------|
| highspy | 3.25s | 0.5MB |
| mip | 1.40s | 4.6MB |

**Result**: MIP is **2.3x faster**, but uses more memory on first run (JIT warmup).

### Medium Pool (300 people, panel size 45)

```bash
uv run python benchmarks/profile_solvers.py \
    --backends highspy mip \
    --algorithms maximin \
    --sizes 300 \
    --runs 3
```

| Backend | Mean Time | Max Memory |
|---------|-----------|------------|
| highspy | 30.3s | 1.1MB |
| mip | 9.6s | 1.8MB |

**Result**: MIP is **3.2x faster**.

### Large Pool (500 people, panel size 75)

```bash
uv run python benchmarks/profile_solvers.py \
    --backends highspy mip \
    --algorithms maximin \
    --sizes 500 \
    --runs 3
```

| Backend | Mean Time | Max Memory |
|---------|-----------|------------|
| highspy | 148.1s | 1.8MB |
| mip | 41.8s | 1.9MB |

**Result**: MIP is **3.5x faster**.

### Extra Large Pool (750 people, panel size 112)

```bash
uv run python benchmarks/profile_solvers.py \
    --backends highspy mip \
    --algorithms maximin \
    --sizes 750 \
    --runs 3
```

| Backend | Mean Time | Max Memory |
|---------|-----------|------------|
| highspy | 261.1s | 7.2MB |
| mip | 119.1s | 7.4MB |

**Result**: MIP is **2.2x faster**.

### Tight Constraints (300 people, panel size 45)

Testing with narrow min/max gaps to stress the solvers:

```bash
uv run python benchmarks/profile_solvers.py \
    --backends highspy mip \
    --algorithms maximin \
    --sizes 300 \
    --runs 3 \
    --tight-constraints
```

| Backend | Mean Time | Max Memory |
|---------|-----------|------------|
| highspy | 17.3s | 0.9MB |
| mip | 4.5s | 1.7MB |

**Result**: MIP is **3.9x faster** with tight constraints.

## Summary

### Performance Comparison

| Pool Size | Panel Size | highspy | mip | MIP Speedup |
|-----------|------------|---------|-----|-------------|
| 150 | 22 | 3.25s | 1.40s | 2.3x |
| 300 | 45 | 30.3s | 9.6s | 3.2x |
| 500 | 75 | 148.1s | 41.8s | 3.5x |
| 750 | 112 | 261.1s | 119.1s | 2.2x |

### Key Findings

1. **MIP (CBC) is consistently faster than highspy (HiGHS)** for the maximin algorithm, by a factor of 2-4x depending on problem size.

2. **Memory usage is comparable** between both backends after initial warmup.

3. **Scaling behaviour** is similar for both backends (roughly O(n²) with pool size), but MIP maintains its performance advantage across all sizes tested.

4. **Tight constraints** make problems harder for both solvers, but MIP's advantage increases slightly.

### Recommendations

- For **production use** with maximin algorithm, consider using `solver_backend = "mip"` if execution time is critical and the mip package is available.

- For **ease of deployment**, `solver_backend = "highspy"` (the default) requires no additional dependencies.

- These results are specific to the **maximin algorithm**. Other algorithms (nash, diversimax, leximin) may show different characteristics.

### Possible Explanations for Performance Difference

The performance gap is unexpected since HiGHS is generally considered a fast solver. Possible factors:

1. **Column generation pattern**: The maximin algorithm iteratively adds constraints. CBC may handle this pattern more efficiently than HiGHS.

2. **Python binding overhead**: The highspy bindings may have different overhead characteristics than python-mip.

3. **Default parameters**: HiGHS and CBC have different default solver parameters that may favour different problem types.

4. **Problem structure**: The committee selection ILP may have structure that CBC exploits better.

## Further Work

Areas for additional investigation:

### Algorithm Comparison

- **Nash algorithm**: Does mip maintain its advantage for Nash welfare optimization?
- **Diversimax algorithm**: How do the backends compare for the diversity-maximizing approach?
- **Leximin algorithm**: Currently requires Gurobi for the dual LP; could benefit from pure HiGHS/mip implementation.

### Performance Tuning

- **HiGHS parameters**: Test different solver parameters (presolve settings, cutting planes, etc.) to see if HiGHS can be tuned for better performance on this problem type.
- **Warm starting**: Investigate whether warm-starting the solver between iterations could improve performance.
- **Model formulation**: The column generation pattern may benefit from different constraint formulations.

### Scaling Analysis

- **Larger pools**: Test with 1000+ people to understand asymptotic scaling behaviour.
- **More features**: Test with additional demographic features to see how constraint complexity affects performance.
- **Panel size ratios**: Test different panel-to-pool ratios (e.g., 10%, 20%, 30%).

### Memory Profiling

- **Detailed memory analysis**: Use memray to generate flamegraphs and identify memory hotspots.
- **Peak vs sustained memory**: Understand whether memory spikes are transient or sustained.

### Real-World Validation

- **Production datasets**: Test with actual sortition datasets (with appropriate anonymisation).
- **Constraint patterns**: Understand which constraint patterns are common in practice and optimise for those.

## Raw Data

Full benchmark results are saved in `benchmarks/results/` as CSV and JSON files with timestamps.
