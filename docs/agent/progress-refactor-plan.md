# Progress Refactor Plan

Status: design agreed, awaiting implementation. This document captures the decisions
made in the design discussion and the concrete plan for landing the work.

## Goal

Provide a structured API so that applications consuming `sortition_algorithms`
can receive live progress updates during long-running selection runs. Currently
the library only emits `logger.debug(...)` messages — fine for the CLI tailing
stdout, but unhelpful for a Flask UI talking to a Celery worker via a database
row.

A run can take 10+ minutes; without progress feedback the calling app has to
guess whether it's stuck.

## Approach: ProgressReporter protocol (observer pattern)

We add a `ProgressReporter` Protocol with three methods. Library code emits
events; calling apps attach an implementation that routes events wherever they
want (DB row, WebSocket, stdout spinner, etc.). A no-op default means existing
callers don't have to change anything.

A flat phase model — no nesting. Each `start_phase` call implicitly ends the
previous phase. Inner loops stay silent. Hierarchical context (e.g. a retry
loop wrapping algorithm phases) is handled by emitting separate phase events
and letting callers track context themselves if they want to.

### Design decisions

| Decision            | Choice                                    | Rationale                                                                                                            |
| ------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Pattern             | Observer (Protocol-based reporter object) | Most structured / extensible. Threads cleanly through the call stack, default null impl, fits the existing codebase. |
| Phase model         | Flat (no nesting)                         | Outer-loop-only is enough for known use cases. Nesting adds API complexity for no real benefit.                      |
| Throttling          | Caller's responsibility                   | Library can't know what the sink looks like. Doc example shows how.                                                  |
| Composite reporters | Doc example only                          | YAGNI; trivial for callers to write themselves.                                                                      |
| Reporter exceptions | Swallowed + logged via wrapper            | A buggy reporter must not kill a 10-minute selection run.                                                            |
| `message` field     | Library always supplies                   | Library knows what's actually happening; reporters stay dumb.                                                        |
| Default reporter    | `NullProgressReporter`                    | Backward compat; existing callers untouched.                                                                         |
| CLI built-in        | `RichProgressReporter`                    | Real implementation in tree. Rich gives spinner + bar + ETA in a single line.                                        |

## Prerequisite: tries-loop refactor

The `for tries in range(settings.max_attempts):` loop in `run_stratification`
must be moved into `find_random_sample_legacy` first. Only the legacy algorithm
ever raises retryable errors, so the loop only meaningfully wraps legacy. Once
that's done, the ILP algorithms (maximin/leximin/nash/diversimax) have no
outer wrapping loop and their natural top-level optimization loops become
_the_ outermost progress phase — no awkward "trial 1 of 1" wrapper.

That refactor is captured separately and lands first.

## New module: `src/sortition_algorithms/progress.py`

```python
# ABOUTME: Progress reporting protocol and default implementations.
# ABOUTME: Library code emits events; callers attach a reporter to receive them.

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

from sortition_algorithms.utils import logger


@runtime_checkable
class ProgressReporter(Protocol):
    """Receives live progress events from long-running selection operations.

    Phases are flat: each ``start_phase`` implicitly ends the previous one.
    The library never throttles; implementations should throttle if their
    sink is expensive (DB writes, network calls). See docs for examples.
    """

    def start_phase(
        self,
        name: str,
        total: int | None = None,
        *,
        message: str | None = None,
    ) -> None:
        """Begin a new phase of work.

        Args:
            name: Stable machine-readable identifier (e.g. "multiplicative_weights").
            total: Number of work units expected, if known. None for convergence
                loops with no fixed total.
            message: Human-readable description. Always supplied by the library.
        """
        ...

    def update(self, current: int, *, message: str | None = None) -> None:
        """Report progress within the current phase.

        Args:
            current: Work units completed so far in this phase. For phases with
                ``total=None``, this is just an iteration count.
            message: Human-readable update. Always supplied by the library.
        """
        ...

    def end_phase(self) -> None:
        """Mark the current phase complete."""
        ...


class NullProgressReporter:
    """No-op reporter. Used as default when no reporter is supplied."""

    def start_phase(self, name: str, total: int | None = None, *, message: str | None = None) -> None:
        pass

    def update(self, current: int, *, message: str | None = None) -> None:
        pass

    def end_phase(self) -> None:
        pass


class ErrorSwallowingReporter:
    """Wrap a reporter so exceptions in any method are caught and logged.

    Used internally by ``coerce_reporter`` so library code never has to worry
    about a buggy caller-supplied reporter killing a long-running selection.
    """

    def __init__(self, delegate: ProgressReporter) -> None:
        self._delegate = delegate

    def start_phase(self, name: str, total: int | None = None, *, message: str | None = None) -> None:
        try:
            self._delegate.start_phase(name, total, message=message)
        except Exception as e:
            logger.warning(
                f"Progress reporter raised in start_phase({name!r}): {e}",
                exc_info=True,
            )

    def update(self, current: int, *, message: str | None = None) -> None:
        try:
            self._delegate.update(current, message=message)
        except Exception as e:
            logger.warning(
                f"Progress reporter raised in update({current}): {e}",
                exc_info=True,
            )

    def end_phase(self) -> None:
        try:
            self._delegate.end_phase()
        except Exception as e:
            logger.warning(
                f"Progress reporter raised in end_phase: {e}",
                exc_info=True,
            )


_NULL_REPORTER: ProgressReporter = NullProgressReporter()


def coerce_reporter(reporter: ProgressReporter | None) -> ProgressReporter:
    """Internal helper used by library entry points.

    Returns the no-op singleton if ``reporter`` is None. Otherwise wraps the
    supplied reporter in an ``ErrorSwallowingReporter`` so exceptions in
    caller code can never propagate into algorithm internals.

    Avoids double-wrapping if ``reporter`` is already a NullProgressReporter
    or ErrorSwallowingReporter.
    """
    if reporter is None:
        return _NULL_REPORTER
    if isinstance(reporter, NullProgressReporter | ErrorSwallowingReporter):
        return reporter
    return ErrorSwallowingReporter(reporter)


@contextmanager
def phase(
    reporter: ProgressReporter,
    name: str,
    total: int | None = None,
    *,
    message: str | None = None,
) -> Iterator[ProgressReporter]:
    """Context manager wrapper around start_phase / end_phase.

    Ensures ``end_phase`` is called even if the body raises. Library code
    should always use this rather than calling start/end manually.
    """
    reporter.start_phase(name, total, message=message)
    try:
        yield reporter
    finally:
        reporter.end_phase()
```

### Notes on the protocol design

- `Protocol` + `runtime_checkable` so callers don't have to inherit anything,
  but `isinstance` works for tests and `coerce_reporter`.
- `NullProgressReporter` is stateless, so `_NULL_REPORTER` is safe as a
  module-level singleton.
- Public API takes `ProgressReporter | None` so the type signature is honest
  about "you don't need to pass anything".
- `name` and `message` are deliberately separate. CLI displays `message`
  directly; a Flask UI might branch on `name` to show different icons/colors.
- `phase()` is a top-level function rather than a method on the protocol so
  implementations don't have to provide it.

## Threading the parameter through

Add `progress_reporter: ProgressReporter | None = None` (keyword-only) to:

**Public entry points:**

- `core.run_stratification`
- `core.find_random_sample`
- `committee_generation.find_distribution_maximin`
- `committee_generation.find_distribution_leximin`
- `committee_generation.find_distribution_nash`
- `committee_generation.diversimax.find_distribution_diversimax`
- `committee_generation.legacy.find_random_sample_legacy` (the post-refactor wrapper)

**Private loop helpers that emit events:**

- `_run_multiplicative_weights_phase` (`common.py`) — bounded, total = `multiplicative_weights_rounds`
- `_run_maximin_optimization_loop` (`maximin.py`) — convergence, total = None
- `_run_nash_optimization_loop` (`nash.py`) — convergence, total = None
- `_run_leximin_main_loop` (`leximin.py`) — bounded, total = `people.count`
- The legacy retry loop (post-refactor) — bounded, total = `max_attempts`
- `find_distribution_diversimax` — single "running" phase, no per-iteration updates

**Inner loops that stay silent** (per outer-loop-only decision):

- `_run_leximin_column_generation_loop`
- `_run_maximin_heuristic_for_additional_committees`
- `_find_committees_for_uncovered_agents`
- `_relax_infeasible_quotas`

## Phase contract

The library emits phases with the following stable `name` values. These names
are part of the public API: callers may switch on them to customise display.
Adding new phases is non-breaking; renaming an existing one is breaking.

| name                     | total                                           | When                                                    |
| ------------------------ | ----------------------------------------------- | ------------------------------------------------------- |
| `legacy_attempt`         | `max_attempts`                                  | each retry of legacy algorithm                          |
| `multiplicative_weights` | `multiplicative_weights_rounds` (typically 200) | initial diverse-committee search (maximin/leximin/nash) |
| `maximin_optimization`   | `None`                                          | maximin convergence loop                                |
| `nash_optimization`      | `None`                                          | Nash convergence loop                                   |
| `leximin_outer`          | `people.count`                                  | leximin's "fix probabilities" outer loop                |
| `diversimax`             | `None`                                          | diversimax single-shot ILP — emitted once, no updates   |

## Sample call-site conversion: multiplicative weights

```python
# committee_generation/common.py

from sortition_algorithms.progress import ProgressReporter, coerce_reporter, phase


def _run_multiplicative_weights_phase(
    solver: Solver,
    agent_vars: dict[str, Any],
    multiplicative_weights_rounds: int,
    progress_reporter: ProgressReporter | None = None,
) -> tuple[set[frozenset[str]], set[str]]:
    reporter = coerce_reporter(progress_reporter)
    committees: set[frozenset[str]] = set()
    covered_agents: set[str] = set()

    weights = {agent_id: random_provider().uniform(0.99, 1.0) for agent_id in agent_vars}

    with phase(
        reporter,
        "multiplicative_weights",
        total=multiplicative_weights_rounds,
        message=f"Searching for diverse committees ({multiplicative_weights_rounds} rounds)",
    ):
        for i in range(multiplicative_weights_rounds):
            solver.set_objective(
                solver_sum(weights[agent_id] * agent_vars[agent_id] for agent_id in agent_vars),
                SolverSense.MAXIMIZE,
            )
            solver.optimize()
            new_committee = ilp_results_to_committee(solver, agent_vars)

            is_new_committee = new_committee not in committees
            if is_new_committee:
                committees.add(new_committee)
                for agent_id in new_committee:
                    covered_agents.add(agent_id)

            _update_multiplicative_weights_after_committee_found(
                weights, new_committee, agent_vars, not is_new_committee
            )

            reporter.update(
                i + 1,
                message=(
                    f"Round {i + 1}/{multiplicative_weights_rounds}: "
                    f"{len(committees)} committees found"
                ),
            )
            logger.debug(
                f"Multiplicative weights phase, round {i + 1}/{multiplicative_weights_rounds}. "
                f"Discovered {len(committees)} committees so far."
            )

    return committees, covered_agents
```

The existing `logger.debug(...)` line stays — diagnostic logging and progress
reporting are now distinct concerns and don't interfere with each other.

For convergence loops (Nash, maximin main, leximin outer), the shape is the
same but `total=None` and `current` is just an iteration counter.

## CLI built-in: `RichProgressReporter`

Goal: when the `sortition_algorithms` CLI runs, the user sees a single line of
progress that updates in place, with a spinner so they know the run isn't
stuck even during convergence loops with no known total.

`click.progressbar` only handles a single bar with a known total and doesn't
combine spinner + bar + transient updates well. **rich** has
`rich.progress.Progress` with `SpinnerColumn`, `TextColumn`, `BarColumn`,
`MofNCompleteColumn`, `TimeElapsedColumn` — much better fit. Add `rich` to the
existing `cli` optional-dependency group in `pyproject.toml` (currently just
holds `click`).

```python
# Sketch only — exact column choice can be tuned
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)


class RichProgressReporter:
    """ProgressReporter that displays a single rich progress line on stdout.

    Use as a context manager so the underlying ``Progress`` is properly stopped
    on exit:

        with RichProgressReporter() as reporter:
            run_stratification(..., progress_reporter=reporter)
    """

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=False,
        )
        self._task_id: TaskID | None = None

    def __enter__(self) -> "RichProgressReporter":
        self._progress.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._progress.stop()

    def start_phase(self, name: str, total: int | None = None, *, message: str | None = None) -> None:
        if self._task_id is not None:
            self._progress.remove_task(self._task_id)
        self._task_id = self._progress.add_task(message or name, total=total)

    def update(self, current: int, *, message: str | None = None) -> None:
        if self._task_id is None:
            return
        kwargs: dict[str, object] = {"completed": current}
        if message is not None:
            kwargs["description"] = message
        self._progress.update(self._task_id, **kwargs)

    def end_phase(self) -> None:
        if self._task_id is None:
            return
        task = self._progress.tasks[self._task_id]
        if task.total is not None:
            self._progress.update(self._task_id, completed=task.total)
```

Wired into `__main__.py` so the CLI defaults to using it when stdout is a
TTY, suppressed when stdout is piped or when `--no-progress` is passed.
`--verbose` controls log level and is a separate concern from progress
display.

## Doc deliverables

**New file: `docs/progress.md`**

1. **Why this exists** — long runs need feedback.
2. **Quickstart** — one-paragraph: pass a reporter to `run_stratification`.
3. **The protocol** — link to API ref.
4. **Phases emitted by the library** — table of stable `name` values.
5. **Throttling** — caller's job; show the DB-row example below.
6. **Recipes**:
   - Database-row reporter (Flask + Celery), throttled to 1Hz
   - stdout reporter
   - Composite/fan-out reporter
   - asyncio.Queue reporter for async web frameworks pushing to WebSockets/SSE
7. **Caveats** — exceptions are swallowed but logged; the library is
   single-threaded, but if a caller drives a reporter from multiple threads
   they're responsible for thread safety. Note this in both `docs/progress.md`
   and the `ProgressReporter` docstring.

**Updates to existing docs**

- `docs/cli.md`: mention the built-in `RichProgressReporter`.
- `docs/api-reference.md`: include `progress` module.
- `docs/modules.md`: include `progress` module.

### Recipe: Flask + Celery DB reporter (will live in docs/progress.md)

```python
import time

from sortition_algorithms.progress import ProgressReporter
from myapp import db
from myapp.models import SortitionRun


class DatabaseProgressReporter:
    """Persist the latest progress event to a SortitionRun row.

    Throttles writes to at most once per ``min_interval_seconds``. Phase
    transitions always flush immediately so the UI sees them without lag.
    """

    def __init__(self, run_id: int, *, min_interval_seconds: float = 1.0) -> None:
        self.run_id = run_id
        self.min_interval = min_interval_seconds
        self._last_write = 0.0
        self._phase_name = ""
        self._phase_total: int | None = None

    def start_phase(self, name: str, total: int | None = None, *, message: str | None = None) -> None:
        self._phase_name = name
        self._phase_total = total
        self._write(current=0, message=message or name, force=True)

    def update(self, current: int, *, message: str | None = None) -> None:
        self._write(current=current, message=message)

    def end_phase(self) -> None:
        pass

    def _write(self, *, current: int, message: str | None, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_write) < self.min_interval:
            return
        self._last_write = now
        SortitionRun.query.filter_by(id=self.run_id).update({
            "phase": self._phase_name,
            "progress_current": current,
            "progress_total": self._phase_total,
            "progress_message": message or "",
        })
        db.session.commit()
```

### Recipe: Composite reporter (will live in docs/progress.md)

```python
class CompositeProgressReporter:
    """Forward every event to multiple child reporters."""

    def __init__(self, *reporters: ProgressReporter) -> None:
        self._children = reporters

    def start_phase(self, name: str, total: int | None = None, *, message: str | None = None) -> None:
        for r in self._children:
            r.start_phase(name, total, message=message)

    def update(self, current: int, *, message: str | None = None) -> None:
        for r in self._children:
            r.update(current, message=message)

    def end_phase(self) -> None:
        for r in self._children:
            r.end_phase()
```

## Implementation phases

Each phase should be committable independently and leave the codebase green.

1. **Phase 0: tries-loop refactor.** Captured separately. Lands first.
2. **Phase 1: scaffolding.** Add `progress.py` with the protocol, null impl,
   error-swallowing wrapper, helpers, and `phase()` context manager. Add unit
   tests for those (recording reporter, exception swallowing, double-wrap
   avoidance, context manager `end_phase` on exception).
3. **Phase 2: thread parameter through.** Add `progress_reporter` keyword-only
   param to all public entry points and private loop helpers, defaulting to
   None. Don't emit anything yet. This phase should be a no-op for all
   existing tests.
4. **Phase 3: emit events.** Add `start_phase` / `update` / `end_phase` calls
   in all the loops listed under "Threading the parameter through". Tests use
   a recording reporter to assert call sequences for each algorithm.
5. **Phase 4: CLI integration.** Add `rich` to the existing `cli`
   optional-dependency group in `pyproject.toml`, implement
   `RichProgressReporter`, wire into `__main__.py` (TTY-gated, with
   `--no-progress` flag).
6. **Phase 5: docs.** Write `docs/progress.md`. Update `cli.md`,
   `api-reference.md`, `modules.md`.

## Resolved decisions

All previously open questions have been answered and folded into the plan
above. Captured here for traceability:

- **`rich` packaging.** Added to the existing `cli` optional-dependency group
  in `pyproject.toml` (alongside `click`), not a required dep.
- **When `RichProgressReporter` is active.** On by default when stdout is a
  TTY, suppressed when piped or when `--no-progress` is passed. `--verbose`
  remains a separate log-level concern.
- **Diversimax reporting.** Single "running" phase with `total=None` and no
  per-iteration updates. No timer thread.
- **`_relax_infeasible_quotas` reporting.** Out of scope for this round.
  Revisit only if it turns out to be a noticeable silent gap in practice.
- **Thread safety of reporters.** The library is single-threaded; callers
  driving a reporter from multiple threads are responsible for their own
  locking. Documented in `docs/progress.md` and in the `ProgressReporter`
  docstring — no code-level enforcement.

## Detailed task checklist

Tick boxes as work lands. Each phase should leave the tree green
(`just test` and `just check`) and be a self-contained commit.

### Phase 0 — tries-loop refactor

- [x] Move `for tries in range(settings.max_attempts):` from
      `core.run_stratification` into `find_random_sample_legacy`
      (shipped in commit 6722b9a)

### Phase 1 — scaffolding (`progress.py` + tests)

Module: `src/sortition_algorithms/progress.py`

- [x] Create `progress.py` with the two `# ABOUTME:` header lines
- [x] Define `ProgressReporter` Protocol (`@runtime_checkable`) with
      `start_phase(name, total=None, *, message=None)`,
      `update(current, *, message=None)`, `end_phase()`
- [x] Implement `NullProgressReporter` (stateless no-op)
- [x] Implement `ErrorSwallowingReporter` wrapping a delegate, catching and
      `logger.warning(..., exc_info=True)`-ing exceptions in each method
- [x] Add module-level `_NULL_REPORTER: ProgressReporter = NullProgressReporter()`
      singleton
- [x] Implement `coerce_reporter(reporter)` helper:
      returns `_NULL_REPORTER` for `None`, returns the reporter unchanged if
      it's already a `NullProgressReporter` or `ErrorSwallowingReporter`,
      otherwise wraps it
- [x] Implement `phase(reporter, name, total=None, *, message=None)`
      context manager (top-level function, not a Protocol method)
- [x] Add thread-safety note to `ProgressReporter` docstring

Tests: `tests/test_progress.py`

- [x] Add a `RecordingProgressReporter` test helper that captures the call
      sequence (and reuse it across this phase and Phase 3)
- [x] `NullProgressReporter` accepts all method calls and does nothing
- [x] `coerce_reporter(None)` returns the singleton
- [x] `coerce_reporter(NullProgressReporter())` returns the same instance
      (no double-wrap)
- [x] `coerce_reporter(ErrorSwallowingReporter(...))` returns the same
      instance (no double-wrap)
- [x] `coerce_reporter(plain_reporter)` returns an `ErrorSwallowingReporter`
      wrapping it
- [x] `ErrorSwallowingReporter` swallows exceptions raised in `start_phase`,
      `update`, and `end_phase`, and emits a `logger.warning` for each
- [x] `phase()` context manager calls `start_phase` then `end_phase` on
      normal exit
- [x] `phase()` calls `end_phase` even when the body raises (and the
      original exception still propagates)

### Phase 2 — thread the parameter through

No event emission yet. All existing tests must remain green.

Public entry points (add `progress_reporter: ProgressReporter | None = None`,
keyword-only, default `None`):

- [x] `core.run_stratification`
- [x] `core.find_random_sample`
- [x] `committee_generation.find_distribution_maximin`
- [x] `committee_generation.find_distribution_leximin`
- [x] `committee_generation.find_distribution_nash`
- [x] `committee_generation.diversimax.find_distribution_diversimax`
- [x] `committee_generation.legacy.find_random_sample_legacy`

Private loop helpers (same signature addition):

- [x] `committee_generation.common._run_multiplicative_weights_phase`
- [x] `committee_generation.maximin._run_maximin_optimization_loop`
- [x] `committee_generation.nash._run_nash_optimization_loop`
- [x] `committee_generation.leximin._run_leximin_main_loop`

Intermediary helpers (added during wiring — not in original checklist but
required to thread the param from public entries to leaf helpers):

- [x] `committee_generation.common.generate_initial_committees`

Wiring:

- [x] At each public entry point, call `coerce_reporter(progress_reporter)`
      once and pass the coerced reporter down the call stack. (Two
      exceptions: `find_random_sample_legacy` and
      `find_distribution_diversimax` have no Phase-2 downstream calls that
      take `progress_reporter`, so they only accept the param without
      coercing — coercing + first emission both land in Phase 3.)
- [x] Confirm `just test` and `just check` are still green with no
      behavioural changes

### Phase 3 — emit events

For each emission site below, call `start_phase` / `update` / `end_phase`
(prefer the `phase()` context manager) per the message strings in the plan.
Existing `logger.debug(...)` lines stay alongside.

- [x] `legacy_attempt` phase in `find_random_sample_legacy` retry loop —
      `total=settings.max_attempts`, update once per attempt
- [x] `multiplicative_weights` phase in `_run_multiplicative_weights_phase`
      — `total=multiplicative_weights_rounds`, update once per round
- [x] `maximin_optimization` phase in `_run_maximin_optimization_loop` —
      `total=None`, update each iteration with iteration counter
- [x] `nash_optimization` phase in `_run_nash_optimization_loop` —
      `total=None`, update each iteration with iteration counter
- [x] `leximin_outer` phase in `_run_leximin_main_loop` —
      `total=people.count`, update as the outer "fix probabilities" loop
      progresses
- [x] `diversimax` phase in `find_distribution_diversimax` — `total=None`,
      single `start_phase` / `end_phase` pair, no per-iteration updates

Tests (extend `tests/test_progress.py` and/or add per-algorithm cases):

- [x] Recording reporter captures expected call sequence for legacy
      (including multiple attempts when retries happen)
- [x] Recording reporter captures expected call sequence for maximin
- [x] Recording reporter captures expected call sequence for nash
- [x] Recording reporter captures expected call sequence for leximin
      (skipped at runtime when Gurobi is unavailable)
- [x] Recording reporter captures expected call sequence for diversimax
- [x] Verify each algorithm still emits its existing diagnostic
      `logger.debug` lines (no regression in logging)
- [x] A reporter that raises in any method does not break a real selection
      run (smoke test wiring `ErrorSwallowingReporter` end-to-end)

### Phase 4 — CLI integration

- [x] Add `rich` to the existing `cli` optional-dependency group in
      `pyproject.toml` (alongside `click`)
- [x] `uv lock` and `uv sync --extra cli`
- [x] Create `src/sortition_algorithms/progress_rich.py` (with the two
      `# ABOUTME:` header lines). This module is the **only** place that
      imports `rich`, keeping `progress.py` and the rest of the library
      free of any `rich` dependency so library users who don't install the
      `[cli]` extra are unaffected
- [x] Implement `RichProgressReporter` in `progress_rich.py` with
      `__enter__` / `__exit__`, `start_phase` removing any prior task and
      adding a new one, `update` updating completion + optional
      description, `end_phase` filling the bar to `total` when known
- [x] Add `--no-progress` flag to the CLI in `__main__.py`
- [x] In `__main__.py`, import `RichProgressReporter` from
      `sortition_algorithms.progress_rich` and instantiate it as a context
      manager when `sys.stdout.isatty()` and `--no-progress` is not set;
      otherwise pass `None`. `progress_rich` must not be imported from
      anywhere in the library other than `__main__.py` and its tests
- [x] Automated test: pipe the CLI through `CliRunner` (non-TTY) and
      confirm a successful happy-path run with the new wiring
- [x] Automated test: invoke the CLI with `--no-progress` and confirm
      a successful happy-path run
- [x] Automated test: `_cli_progress_reporter` helper yields `None` for
      `--no-progress` and for non-TTY stdout, and yields a real
      `RichProgressReporter` for a TTY without `--no-progress`

### Phase 5 — docs

`docs/progress.md` (new file):

- [x] "Why this exists" — long runs need feedback
- [x] Quickstart — pass a reporter to `run_stratification`
- [x] The protocol — link to API ref
- [x] Phases emitted by the library — table of stable `name` values (copy
      from this plan, keep in sync)
- [x] Throttling — caller's responsibility, with the DB-row example
- [x] Recipe: Database-row reporter (Flask + Celery), throttled to 1Hz
- [x] Recipe: stdout reporter
- [x] Recipe: composite / fan-out reporter
- [x] Recipe: `asyncio.Queue` reporter for WebSockets/SSE
- [x] Caveats: exceptions are swallowed but logged; reporters must be
      thread-safe if the caller drives them across threads

Updates to existing docs:

- [x] `docs/cli.md` — mention `RichProgressReporter` and `--no-progress`
- [x] `docs/api-reference.md` — include `progress` module
- [x] `docs/modules.md` — include `progress` module
- [x] Cross-link `docs/progress.md` from the main docs index and add it
      to the `mkdocs.yml` nav

Final sweep:

- [x] `just test` green
- [x] `just check` green
- [x] `just docs-test` green
