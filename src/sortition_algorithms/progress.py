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

    Thread safety: the library is single-threaded and will only call a
    reporter from one thread at a time. Callers that drive a reporter from
    multiple threads (for example, reading progress from a web request
    thread while the selection runs in a worker thread) are responsible for
    their own locking.
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
