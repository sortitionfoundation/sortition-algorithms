# ABOUTME: Memray profiling wrapper for detailed memory analysis.
# ABOUTME: Run with: uv run memray run benchmarks/memray_profile.py
"""
Memray profiling script for detailed memory analysis.

Usage:
    # Run memray profiling (generates .bin file)
    uv run memray run benchmarks/memray_profile.py

    # Generate flamegraph from the output
    uv run memray flamegraph memray-*.bin -o benchmarks/results/flamegraph.html

    # Generate summary report
    uv run memray summary memray-*.bin

    # Generate tree view
    uv run memray tree memray-*.bin

Options can be passed via environment variables:
    PROFILE_BACKEND=mip uv run memray run benchmarks/memray_profile.py
    PROFILE_ALGORITHM=nash uv run memray run benchmarks/memray_profile.py
    PROFILE_SIZE=500 uv run memray run benchmarks/memray_profile.py
"""

import os

from benchmarks.generate_fixtures import generate_scaled_fixtures, load_base_fixtures
from sortition_algorithms.core import find_random_sample
from sortition_algorithms.utils import set_random_provider


def run_profiled_selection() -> None:
    """Run a single selection that will be profiled by memray."""
    # Configuration from environment variables
    backend = os.environ.get("PROFILE_BACKEND", "highspy")
    algorithm = os.environ.get("PROFILE_ALGORITHM", "maximin")
    size = int(os.environ.get("PROFILE_SIZE", "150"))
    panel_size = int(os.environ.get("PROFILE_PANEL_SIZE", "22"))
    seed = int(os.environ.get("PROFILE_SEED", "42"))

    print("Memray Profiling Configuration:")
    print(f"  Backend: {backend}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Pool size: {size}")
    print(f"  Panel size: {panel_size}")
    print(f"  Seed: {seed}")
    print()

    # Load fixtures
    print("Loading fixtures...")
    if size == 150:
        features, people = load_base_fixtures()
    else:
        features, people = generate_scaled_fixtures(size)
    print(f"Loaded {people.count} people")

    # Set random seed
    set_random_provider(seed)

    # Run the selection
    print(f"\nRunning {algorithm} with {backend} backend...")
    committees, report = find_random_sample(
        features,
        people,
        panel_size,
        check_same_address_columns=[],
        selection_algorithm=algorithm,
        solver_backend=backend,
        max_seconds=120,
    )

    print("\nSelection complete!")
    print(f"  Committees found: {len(committees)}")
    if committees:
        print(f"  Committee size: {len(committees[0])}")


if __name__ == "__main__":
    run_profiled_selection()
