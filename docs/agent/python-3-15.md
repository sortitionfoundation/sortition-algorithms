# Python 3.15 support: findings and options

First written 2026-10-14 and updated 2026-11-04, on the `python-3.15-support` branch. See `README.md` in this directory - this is a snapshot, so check PyPI and the code before acting on it.

## What the branch changes

- It is based on the SPEC 0 change (PR #176), which sets `requires-python = ">=3.12"` and relocks so that numpy 2.5.4 and scipy 1.18.1 are used on every Python version. The SPEC 0 policy is documented in `docs/quickstart.md`.
- `pyproject.toml`: adds the `Programming Language :: Python :: 3.15` classifier.
- `.github/workflows/main.yml`: adds `"3.15"` to the `tests-and-type-check` matrix.

CPython 3.15.2 is a stable release, so `actions/setup-python` needs no `allow-prereleases` setting.

The 3.15 CI job is expected to fail at the `uv sync` step until the blockers below are resolved.

## Findings

### Our code works on 3.15

With every missing package built from source (see below), the full test suite passed on CPython 3.15.2: 483 passed and 9 skipped, the same result as 3.12 to 3.14. mypy was also clean. The remaining work is all about upstream packaging, not our code.

### Wheel availability

On a Python version with no matching wheel, uv falls back to building the sdist. Requires-python metadata is not the problem: none of these packages exclude 3.15. The problem is that compiled `cp314` wheels cannot load on 3.15. Only `abi3` (stable ABI) wheels are forward-compatible, which is why clarabel is not a blocker.

| Package      | Locked version | cp315 wheels on PyPI?                                                       |
| ------------ | -------------- | --------------------------------------------------------------------------- |
| cvxpy        | 1.9.3          | **None in any release**                                                     |
| sparsediffpy | 0.6.1          | From 0.7.0, but **cvxpy 1.9.3 pins `sparsediffpy<0.7.0`** (cvxpy dependency) |
| highspy      | 1.15.1         | **None in any release**                                                     |
| osqp         | 1.1.3          | **None in any release** (cvxpy dependency)                                  |
| qdldl        | 0.1.9.post1    | **None in any release** (osqp dependency)                                   |
| pyyaml       | 6.0.3          | None in any release (dev/docs dependency only)                              |

These are now resolved: numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, scikit-learn 1.9.1, scs 3.3.1, markupsafe 3.0.4, memray 1.20.0 and cffi 2.1.1 all have cp315 wheels in the lock.

### Upstream status

- **cvxpy**: master has moved to `sparsediffpy >= 0.7.0, < 0.8.0` and started testing 3.15 ([cvxpy#3548](https://github.com/cvxpy/cvxpy/pull/3548), merged 2026-10-12). The next release should fix both cvxpy and sparsediffpy.
- **highspy**: [HiGHS#3444](https://github.com/ERGO-Code/HiGHS/issues/3444) "Python 3.15 wheels for highspy" is open and was active on 2026-11-03.
- **osqp / qdldl**: no 3.15 issues or PRs found. The last osqp-python commit was 2026-06-12, and the last qdldl-python commit was 2026-02-19. **These are the most likely long-term blockers.**
- **pyyaml**: several open issues, plus an open PR ([pyyaml#990](https://github.com/yaml/pyyaml/pull/990)). It only affects dev and docs tooling.

### Building from source works, given BLAS

Without BLAS, `uv sync --python 3.15 --locked` fails building sparsediffpy 0.6.1: `Could NOT find BLAS`.

With OpenBLAS available, the whole stack builds and installs. The local test used:

```bash
OB=/home/linuxbrew/.linuxbrew/opt/openblas
CMAKE_PREFIX_PATH=$OB PKG_CONFIG_PATH=$OB/lib/pkgconfig BLA_VENDOR=OpenBLAS \
  uv sync --python 3.15 --locked
```

This took about 80 seconds of wall-clock time on a many-core machine (around 730s of CPU time), so expect several minutes on a GitHub runner. On Ubuntu, `apt-get install libopenblas-dev` should be the equivalent.

This does not help end users. `pip install sortition-algorithms` on 3.15 would need a C/C++ compiler, CMake and BLAS.

## Options

### 1. Wait

Leave the branch unmerged until cvxpy, highspy, osqp and qdldl publish cp315 wheels. Then refresh the lock (`uv lock -P cvxpy -P highspy -P osqp -P qdldl -P sparsediffpy`), check that `uv sync --python 3.15 --locked` installs from wheels only, and merge.

### 2. Build from source in CI for the 3.15 job only

Install `libopenblas-dev` in the 3.15 job, and leave out the 3.15 classifier until wheels exist. CI then really tests our code on 3.15 and stays green, without PyPI advertising support most users can't install. When wheels arrive, remove the extra step and add the classifier. The cost is a slower 3.15 job, plus a build step that may break as sdists change.

### 3. Merge with the 3.15 job marked experimental

Add `continue-on-error: ${{ matrix.python-version == '3.15' }}` to the job, and leave out the 3.15 classifier. The job turns green on its own once wheels exist and the lock is refreshed. The downside is that people tend to ignore a job that's allowed to fail.

## Suggested path

Option 2 now, then option 1's cleanup once wheels arrive. If osqp and qdldl are still missing cp315 wheels long after cvxpy and highspy have them, ask upstream (or check whether cvxpy has made osqp optional) rather than carrying the source build indefinitely.
