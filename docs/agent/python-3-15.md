# Python 3.15 support: findings and options

Written 2026-10-14, on the `python-3.15-support` branch. See `README.md` in this directory - this is a snapshot, so check PyPI and the code before acting on it.

## What the branch changes

- `pyproject.toml`: adds the `Programming Language :: Python :: 3.15` classifier. `requires-python` stays at `>=3.11`.
- `.github/workflows/main.yml`: adds `"3.15"` to the `tests-and-type-check` matrix.
- `.github/actions/setup-python-env/action.yml`: adds `allow-prereleases: true` to `actions/setup-python`. That lets `3.15` resolve to a release candidate while no stable 3.15 build is available. It has no effect on versions with a stable release.

The 3.15 CI job is expected to fail at the `uv sync` step until the blockers below are resolved.

## Findings

At the time of writing, uv (0.12.23) only offered CPython 3.15.0rc3.

### The lockfile has no 3.15 wheels for the core numeric stack

These locked packages have compiled wheels but none for cp315:

| Package      | Locked version | cp315 wheels on PyPI?                               |
| ------------ | -------------- | --------------------------------------------------- |
| cvxpy        | 1.9.3          | **None in any release**                             |
| highspy      | 1.15.1         | **None in any release**                             |
| osqp         | 1.1.3          | **None in any release** (cvxpy dependency)          |
| qdldl        | 0.1.9.post1    | **None in any release** (osqp dependency)           |
| pyyaml       | 6.0.3          | None in any release (dev/docs dependency only)      |
| numpy        | 2.4.6          | From 2.5.2 - but numpy 2.5 requires Python >=3.12   |
| scipy        | 1.17.1         | From 1.18.1 - but scipy 1.18 requires Python >=3.12 |
| sparsediffpy | 0.6.1          | From 0.7.0 (released 2026-10-08)                    |
| markupsafe   | 3.0.3          | From 3.0.4 (dev/docs dependency only)               |

pandas 3.0.6 and scikit-learn 1.9.1 (the `diversimax` extra) already have cp315 wheels and still support 3.11.

`uv lock --upgrade` does not move numpy or scipy forward. Because we support 3.11, the resolver can't pick numpy 2.5 or scipy 1.18. The one-week `exclude-newer` cooldown also holds back very recent releases such as sparsediffpy 0.7.0.

### Building from source is not viable

`uv sync --python 3.15 --extra=diversimax --extra=cli --frozen` failed within seconds while building `sparsediffpy` from source (pulled in by cvxpy): `Could NOT find BLAS`. Even with BLAS installed, scipy 1.17.1 would still need a Fortran compiler and OpenBLAS to build. End users running `pip install sortition-algorithms` on 3.15 would hit the same failure.

### numpy has already dropped 3.11

numpy 2.5 and scipy 1.18 both require Python >=3.12, in line with the SPEC 0 support schedule. So 3.11 has already lost the newest numpy and scipy releases. We just haven't been affected yet, because 2.4.x and 1.17.x still resolve. CPython 3.11 itself reaches end of life in October 2027.

## Options

### 1. Wait (current state of the branch)

Leave the branch unmerged until cvxpy, highspy, osqp and qdldl publish cp315 wheels. Then refresh the lock and run `uv sync --python 3.15` locally before merging. This is the lowest-effort option. It still needs one of options 2 or 3 to handle numpy and scipy.

### 2. Keep 3.11 and use version-specific lower bounds for numpy/scipy

Add environment markers so uv forks the resolution, for example:

```toml
"numpy>=2.3.4",
"numpy>=2.5.2; python_version >= '3.15'",
```

scipy is only a transitive dependency (via cvxpy and scikit-learn), so it would need the same treatment through `[tool.uv] constraint-dependencies`. Then 3.11 keeps getting numpy 2.4.x / scipy 1.17.x, and 3.15 gets versions with wheels. The downside is a more complicated lockfile, and testing against different numpy majors across the matrix.

### 3. Drop 3.11 and make 3.12 the minimum

Set `requires-python = ">=3.12"`, remove the 3.11 classifier and matrix entry, and move the Codecov upload condition to the new lowest version. The lock can then use numpy 2.5 / scipy 1.18 everywhere, with no markers. This follows numpy and scipy's own support policy, and it would also bring `[tool.ruff] target-version = "py312"` in line with `requires-python`.

Note that **dropping 3.11 does not by itself unblock 3.15**. cvxpy, highspy, osqp and qdldl are still missing cp315 wheels. This should probably be decided separately, on its own merits: who still runs 3.11 downstream?

### 4. Merge now with the 3.15 job marked experimental

Add `continue-on-error: ${{ matrix.python-version == '3.15' }}` to the job, and leave out the 3.15 classifier until the job passes. CI then reports 3.15 status without going red, and the job turns green on its own once wheels exist and the lock is refreshed. The downside is that people tend to ignore a job that's allowed to fail.

### 5. Build from source in CI

Install BLAS, LAPACK and gfortran in CI and build the missing packages from source. Not recommended: it's slow and fragile, and it doesn't help end users, who would still have no wheels.

## Suggested path

Options 1 and 3 together. Decide on dropping 3.11 now, as a separate change. Keep this branch until cvxpy and highspy (the direct dependencies) publish cp315 wheels, then rebase, refresh the lock and merge.
