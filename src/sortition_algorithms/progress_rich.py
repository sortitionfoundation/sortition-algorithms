# ABOUTME: Rich-based ProgressReporter for the CLI. Imports rich, so this
# ABOUTME: module must only be imported by __main__.py and its tests.

from types import TracebackType

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
    on exit::

        with RichProgressReporter() as reporter:
            run_stratification(..., progress_reporter=reporter)

    Thread safety: rich's ``Progress`` handles its own internal locking, but
    the library is single-threaded and only calls a reporter from one thread
    at a time. If a caller drives this reporter from another thread, that's
    the caller's responsibility.
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
        self._task_total: int | None = None

    def __enter__(self) -> "RichProgressReporter":
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._progress.stop()

    def start_phase(self, name: str, total: int | None = None, *, message: str | None = None) -> None:
        if self._task_id is not None:
            self._progress.remove_task(self._task_id)
        self._task_id = self._progress.add_task(message or name, total=total)
        self._task_total = total

    def update(self, current: int, *, message: str | None = None) -> None:
        if self._task_id is None:
            return
        if message is None:
            self._progress.update(self._task_id, completed=current)
        else:
            self._progress.update(self._task_id, completed=current, description=message)

    def end_phase(self) -> None:
        if self._task_id is None:
            return
        if self._task_total is not None:
            self._progress.update(self._task_id, completed=self._task_total)
