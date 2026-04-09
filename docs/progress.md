# Progress Reporting

Long-running selection algorithms can take 10+ minutes on real-world pools.
The library emits structured progress events while it runs so calling
applications can show live feedback to users — a CLI spinner, a progress row
in a database, a WebSocket push to a browser, anything.

## Why this exists

Without progress events, an application calling `run_stratification` has to
guess whether a long-running selection is healthy or wedged. Logging a
`logger.debug` line is fine for the CLI tailing stdout, but unhelpful when
the selection is running in a Celery worker that talks to a Flask UI via a
database row.

The progress reporting API gives library callers a structured stream of
events so they can route progress wherever it needs to go.

## Quickstart

Pass any object satisfying the `ProgressReporter` protocol via the
`progress_reporter` keyword argument:

```python
from sortition_algorithms.core import run_stratification
from sortition_algorithms.progress_rich import RichProgressReporter

with RichProgressReporter() as reporter:
    success, panels, report = run_stratification(
        features=features,
        people=people,
        number_people_wanted=100,
        settings=settings,
        progress_reporter=reporter,
    )
```

If you don't pass anything, the library uses a no-op default and behaves
exactly as before.

## The protocol

```python
class ProgressReporter(Protocol):
    def start_phase(
        self,
        name: str,
        total: int | None = None,
        *,
        message: str | None = None,
    ) -> None: ...

    def update(self, current: int, *, message: str | None = None) -> None: ...

    def end_phase(self) -> None: ...
```

Any class with these three methods works — `Protocol` means you don't have
to inherit anything. The full reference lives in the
[Modules](modules.md) page under `sortition_algorithms.progress`.

Phases are flat: each `start_phase` implicitly ends the previous one.
`name` is a stable machine-readable identifier; `message` is a
human-readable description that the library always supplies. Use `name` if
you want to switch on the phase to (e.g.) display different icons; use
`message` directly if you just want to show text.

## Phases emitted by the library

The phase names below are part of the public API: callers may switch on them
to customise display. Adding new phases is non-breaking; renaming an
existing one would be a breaking change.

| name                     | total                                           | When                                                    |
| ------------------------ | ----------------------------------------------- | ------------------------------------------------------- |
| `legacy_attempt`         | `max_attempts`                                  | each retry of the legacy algorithm                      |
| `multiplicative_weights` | `multiplicative_weights_rounds` (typically 200) | initial diverse-committee search (maximin/leximin/nash) |
| `maximin_optimization`   | `None`                                          | maximin convergence loop                                |
| `nash_optimization`      | `None`                                          | Nash convergence loop                                   |
| `leximin_outer`          | `people.count`                                  | leximin's "fix probabilities" outer loop                |
| `diversimax`             | `None`                                          | diversimax single-shot ILP — emitted once, no updates   |

A `total` of `None` means it's a convergence loop with no fixed work
budget; `current` is then just an iteration counter.

## Throttling

The library does not throttle. It calls `update` every iteration of the
inner loop, which can be hundreds of times per second on a fast solver.
That's fine for in-memory sinks but expensive for anything that talks to a
database or the network. **Throttling is the caller's responsibility**.

The database recipe below shows the standard pattern: timestamp the last
write, drop updates within `min_interval_seconds` of the previous one, but
always flush phase transitions immediately so the UI sees them without lag.

## Recipes

### Database-row reporter (Flask + Celery), throttled to 1Hz

A long-running selection in a Celery worker writes its current phase and
progress into a `SortitionRun` row. A Flask view polls that row to render a
progress bar in the UI.

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

### stdout reporter

For ad-hoc scripts, a five-line reporter that prints each event:

```python
class StdoutProgressReporter:
    def start_phase(self, name, total=None, *, message=None):
        print(f"[{name}] {message or ''} (total={total})")

    def update(self, current, *, message=None):
        print(f"  {current}: {message or ''}")

    def end_phase(self):
        print("  done")
```

### Composite (fan-out) reporter

Send every event to multiple sinks at once — for example, a database row
*and* a stdout line in development:

```python
from sortition_algorithms.progress import ProgressReporter


class CompositeProgressReporter:
    """Forward every event to multiple child reporters."""

    def __init__(self, *reporters: ProgressReporter) -> None:
        self._children = reporters

    def start_phase(self, name, total=None, *, message=None):
        for r in self._children:
            r.start_phase(name, total, message=message)

    def update(self, current, *, message=None):
        for r in self._children:
            r.update(current, message=message)

    def end_phase(self):
        for r in self._children:
            r.end_phase()
```

### asyncio.Queue reporter for WebSockets / SSE

For async web frameworks pushing progress to a browser via WebSockets or
Server-Sent Events. Drop the reporter into a synchronous worker thread and
have the async side drain the queue:

```python
import asyncio


class AsyncQueueProgressReporter:
    """Push every event into an asyncio.Queue.

    The reporter itself is sync (so it can be called from a worker thread
    inside the library), but it pushes thread-safely into the queue using
    ``loop.call_soon_threadsafe``.
    """

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        self._queue = queue
        self._loop = loop

    def start_phase(self, name, total=None, *, message=None):
        self._push({"type": "start_phase", "name": name, "total": total, "message": message})

    def update(self, current, *, message=None):
        self._push({"type": "update", "current": current, "message": message})

    def end_phase(self):
        self._push({"type": "end_phase"})

    def _push(self, event: dict) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
```

The async side then awaits `queue.get()` and forwards each event to the
WebSocket / SSE stream.

## Caveats

### Exceptions are swallowed but logged

If your reporter raises in any method, the library catches the exception
and logs a warning to the `sortition_algorithms` logger — a buggy reporter
must never kill a 10-minute selection. Tracebacks are included via
`logger.warning(..., exc_info=True)`.

This means you should still test your reporter, but you don't need to wrap
every method in a try/except defensively.

### Thread safety

The library is single-threaded and only calls a reporter from one thread at
a time. If you drive a reporter from another thread (for example, reading
its state from a Flask request thread while the selection runs in a Celery
worker), **you are responsible for any locking**. The reporters bundled with
the library — `NullProgressReporter`, `ErrorSwallowingReporter`, and
`RichProgressReporter` — are safe to read concurrently from a single
external thread, but custom reporters should add their own locks if they
mutate state.

### Adding new phases is fine; renaming them is not

Phase `name` values are part of the public API. New phases can land in any
release; renaming an existing one would be a breaking change.
