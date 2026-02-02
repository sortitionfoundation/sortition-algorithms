# ABOUTME: Profile solver performance (time and memory) across backends and algorithms.
# ABOUTME: Generates comparison data for highspy vs mip solvers.
"""
Profiling script for comparing solver backend performance.

Usage:
    uv run python benchmarks/profile_solvers.py
    uv run python benchmarks/profile_solvers.py --backends highspy mip
    uv run python benchmarks/profile_solvers.py --algorithms maximin nash
    uv run python benchmarks/profile_solvers.py --sizes 150 500
"""

import argparse
import csv
import json
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from benchmarks.generate_fixtures import generate_scaled_fixtures, load_base_fixtures
from sortition_algorithms.committee_generation.solver import MIP_AVAILABLE
from sortition_algorithms.core import find_random_sample
from sortition_algorithms.features import FeatureCollection
from sortition_algorithms.people import People
from sortition_algorithms.utils import set_random_provider


@dataclass
class ProfileResult:
    """Results from a single profiling run."""

    backend: str
    algorithm: str
    num_people: int
    panel_size: int
    elapsed_seconds: float
    peak_memory_mb: float
    success: bool
    error: str | None = None


@dataclass
class ProfileConfig:
    """Configuration for a profiling session."""

    backends: list[str] = field(default_factory=lambda: ["highspy"])
    algorithms: list[str] = field(default_factory=lambda: ["maximin", "nash", "diversimax"])
    sizes: list[int] = field(default_factory=lambda: [150])
    panel_size: int | None = None  # None = auto-calculate based on pool size
    num_runs: int = 3
    seed: int = 42
    tight_constraints: bool = False  # If True, use narrow min/max gaps


def load_fixtures(
    size: int = 150,
    panel_size: int | None = None,
    tight_constraints: bool = False,
) -> tuple[FeatureCollection, People, int]:
    """Load test fixtures of the specified size.

    Args:
        size: Number of people (150 uses existing fixture, others use generated)
        panel_size: Target panel size (None = auto-calculate)
        tight_constraints: If True, use narrow min/max gaps

    Returns:
        tuple of (features, people, actual_panel_size)
    """
    if size == 150 and not tight_constraints and panel_size is None:
        features, people = load_base_fixtures()
        return features, people, 22  # Default panel size for base fixture
    else:
        # Calculate default panel size if not specified
        if panel_size is None:
            panel_size = max(22, int(size * 0.15))

        features, people = generate_scaled_fixtures(
            size,
            panel_size=panel_size,
            tight_constraints=tight_constraints,
        )
        return features, people, panel_size


def profile_single_run(
    features: FeatureCollection,
    people: People,
    panel_size: int,
    algorithm: str,
    backend: str,
    seed: int,
) -> ProfileResult:
    """Run a single profiling measurement.

    Args:
        features: FeatureCollection with quotas
        people: People pool
        panel_size: Number of people to select
        algorithm: Selection algorithm name
        backend: Solver backend name
        seed: Random seed for reproducibility

    Returns:
        ProfileResult with timing and memory data
    """
    set_random_provider(seed)

    # Start memory tracking
    tracemalloc.start()

    error_msg = None
    success = True
    start_time = time.perf_counter()

    try:
        _committees, _report = find_random_sample(
            features,
            people,
            panel_size,
            check_same_address_columns=[],
            selection_algorithm=algorithm,
            solver_backend=backend,
            max_seconds=60,
        )
    except Exception as e:
        success = False
        error_msg = str(e)

    elapsed = time.perf_counter() - start_time

    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)

    return ProfileResult(
        backend=backend,
        algorithm=algorithm,
        num_people=people.count,
        panel_size=panel_size,
        elapsed_seconds=elapsed,
        peak_memory_mb=peak_mb,
        success=success,
        error=error_msg,
    )


def run_profiling(config: ProfileConfig) -> list[ProfileResult]:
    """Run full profiling suite based on configuration.

    Args:
        config: ProfileConfig with settings

    Returns:
        list of ProfileResult objects
    """
    results: list[ProfileResult] = []

    # Check which backends are available
    available_backends = ["highspy"]
    if MIP_AVAILABLE:
        available_backends.append("mip")

    backends_to_test = [b for b in config.backends if b in available_backends]
    skipped_backends = [b for b in config.backends if b not in available_backends]

    if skipped_backends:
        print(f"Skipping unavailable backends: {skipped_backends}")

    for size in config.sizes:
        print(f"\n{'=' * 60}")
        constraint_type = "tight" if config.tight_constraints else "normal"
        print(f"Loading fixtures for {size} people ({constraint_type} constraints)...")
        features, people, actual_panel_size = load_fixtures(
            size,
            panel_size=config.panel_size,
            tight_constraints=config.tight_constraints,
        )
        print(f"Loaded {people.count} people with {len(features)} features")
        print(f"Panel size: {actual_panel_size}")

        for algorithm in config.algorithms:
            for backend in backends_to_test:
                print(f"\n  Profiling {algorithm} with {backend} backend...")

                run_results = []
                for run_num in range(config.num_runs):
                    seed = config.seed + run_num
                    result = profile_single_run(
                        features,
                        people,
                        actual_panel_size,
                        algorithm,
                        backend,
                        seed,
                    )
                    run_results.append(result)

                    status = "OK" if result.success else f"FAILED: {result.error}"
                    print(
                        f"    Run {run_num + 1}: {result.elapsed_seconds:.3f}s, "
                        f"{result.peak_memory_mb:.1f}MB - {status}"
                    )

                results.extend(run_results)

    return results


def summarize_results(results: list[ProfileResult]) -> dict:
    """Generate summary statistics from profiling results.

    Args:
        results: list of ProfileResult objects

    Returns:
        dict with summary statistics grouped by configuration
    """
    groups: dict[tuple, list[ProfileResult]] = defaultdict(list)

    for r in results:
        key = (r.backend, r.algorithm, r.num_people, r.panel_size)
        groups[key].append(r)

    summary = {}
    for key, group in groups.items():
        backend, algorithm, num_people, panel_size = key
        successful = [r for r in group if r.success]

        if successful:
            times = [r.elapsed_seconds for r in successful]
            memories = [r.peak_memory_mb for r in successful]

            summary[f"{backend}_{algorithm}_{num_people}"] = {
                "backend": backend,
                "algorithm": algorithm,
                "num_people": num_people,
                "panel_size": panel_size,
                "runs": len(group),
                "successful_runs": len(successful),
                "mean_time_s": sum(times) / len(times),
                "min_time_s": min(times),
                "max_time_s": max(times),
                "mean_memory_mb": sum(memories) / len(memories),
                "max_memory_mb": max(memories),
            }
        else:
            summary[f"{backend}_{algorithm}_{num_people}"] = {
                "backend": backend,
                "algorithm": algorithm,
                "num_people": num_people,
                "panel_size": panel_size,
                "runs": len(group),
                "successful_runs": 0,
                "error": group[0].error if group else "Unknown",
            }

    return summary


def save_results(results: list[ProfileResult], summary: dict, output_dir: Path) -> None:
    """Save profiling results to files.

    Args:
        results: list of ProfileResult objects
        summary: Summary statistics dict
        output_dir: Directory to save results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results as CSV
    csv_path = output_dir / f"profile_results_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "backend",
            "algorithm",
            "num_people",
            "panel_size",
            "elapsed_seconds",
            "peak_memory_mb",
            "success",
            "error",
        ])
        for r in results:
            writer.writerow([
                r.backend,
                r.algorithm,
                r.num_people,
                r.panel_size,
                f"{r.elapsed_seconds:.4f}",
                f"{r.peak_memory_mb:.2f}",
                r.success,
                r.error or "",
            ])
    print(f"\nRaw results saved to: {csv_path}")

    # Save summary as JSON
    json_path = output_dir / f"profile_summary_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")


def print_summary_table(summary: dict) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 80)
    print("PROFILING SUMMARY")
    print("=" * 80)

    print(f"\n{'Backend':<10} {'Algorithm':<12} {'People':<8} {'Runs':<6} {'Mean Time':<12} {'Max Memory':<12}")
    print("-" * 70)

    for key in sorted(summary.keys()):
        s = summary[key]
        if s["successful_runs"] > 0:
            print(
                f"{s['backend']:<10} {s['algorithm']:<12} {s['num_people']:<8} "
                f"{s['successful_runs']:<6} {s['mean_time_s']:.3f}s       {s['max_memory_mb']:.1f}MB"
            )
        else:
            print(
                f"{s['backend']:<10} {s['algorithm']:<12} {s['num_people']:<8} "
                f"{'FAILED':<6} {s.get('error', 'Unknown')[:30]}"
            )


def main() -> None:
    """Main entry point for the profiling script."""
    parser = argparse.ArgumentParser(description="Profile sortition algorithm solver performance")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["highspy"],
        help="Solver backends to test (default: highspy)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["maximin", "nash", "diversimax"],
        help="Algorithms to test (default: maximin nash diversimax)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[150],
        help="Pool sizes to test (default: 150)",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=None,
        help="Panel size to select (default: auto ~15%% of pool)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per configuration (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--tight-constraints",
        action="store_true",
        help="Use tight min/max constraints (harder for solvers)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    config = ProfileConfig(
        backends=args.backends,
        algorithms=args.algorithms,
        sizes=args.sizes,
        panel_size=args.panel_size,
        num_runs=args.runs,
        seed=args.seed,
        tight_constraints=args.tight_constraints,
    )

    print("Sortition Algorithm Solver Profiling")
    print("=" * 40)
    print(f"Backends: {config.backends}")
    print(f"Algorithms: {config.algorithms}")
    print(f"Pool sizes: {config.sizes}")
    print(f"Panel size: {config.panel_size or 'auto (~15% of pool)'}")
    print(f"Tight constraints: {config.tight_constraints}")
    print(f"Runs per config: {config.num_runs}")

    results = run_profiling(config)
    summary = summarize_results(results)

    print_summary_table(summary)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, summary, args.output_dir)


if __name__ == "__main__":
    main()
