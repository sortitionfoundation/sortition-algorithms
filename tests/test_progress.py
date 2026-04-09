"""ABOUTME: Tests for the progress reporting protocol and helpers in progress.py.
ABOUTME: Includes a recording reporter used here and reused by later phases.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from sortition_algorithms import progress
from sortition_algorithms.progress import (
    ErrorSwallowingReporter,
    NullProgressReporter,
    ProgressReporter,
    coerce_reporter,
    phase,
)


@dataclass
class RecordingProgressReporter:
    """Test helper that captures every event for later assertion."""

    events: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def start_phase(self, name: str, total: int | None = None, *, message: str | None = None) -> None:
        self.events.append(("start_phase", (name, total), {"message": message}))

    def update(self, current: int, *, message: str | None = None) -> None:
        self.events.append(("update", (current,), {"message": message}))

    def end_phase(self) -> None:
        self.events.append(("end_phase", (), {}))


class RaisingProgressReporter:
    """Reporter whose every method raises - used to test ErrorSwallowingReporter."""

    def start_phase(self, name: str, total: int | None = None, *, message: str | None = None) -> None:
        raise RuntimeError(f"boom in start_phase({name!r})")

    def update(self, current: int, *, message: str | None = None) -> None:
        raise RuntimeError(f"boom in update({current})")

    def end_phase(self) -> None:
        raise RuntimeError("boom in end_phase")


class TestRecordingProgressReporter:
    def test_records_start_update_end_in_order(self):
        reporter = RecordingProgressReporter()
        reporter.start_phase("phase_a", 10, message="starting a")
        reporter.update(3, message="three")
        reporter.end_phase()

        assert reporter.events == [
            ("start_phase", ("phase_a", 10), {"message": "starting a"}),
            ("update", (3,), {"message": "three"}),
            ("end_phase", (), {}),
        ]

    def test_satisfies_protocol(self):
        reporter = RecordingProgressReporter()
        assert isinstance(reporter, ProgressReporter)


class TestNullProgressReporter:
    def test_all_methods_are_no_ops(self):
        reporter = NullProgressReporter()
        reporter.start_phase("anything", 5, message="msg")
        reporter.update(2, message="msg")
        reporter.end_phase()

    def test_satisfies_protocol(self):
        assert isinstance(NullProgressReporter(), ProgressReporter)


class TestCoerceReporter:
    def test_none_returns_singleton(self):
        result = coerce_reporter(None)
        assert result is progress._NULL_REPORTER

    def test_none_returns_singleton_repeatedly(self):
        assert coerce_reporter(None) is coerce_reporter(None)

    def test_null_reporter_not_double_wrapped(self):
        null = NullProgressReporter()
        assert coerce_reporter(null) is null

    def test_error_swallowing_reporter_not_double_wrapped(self):
        wrapped = ErrorSwallowingReporter(RecordingProgressReporter())
        assert coerce_reporter(wrapped) is wrapped

    def test_plain_reporter_is_wrapped(self):
        recording = RecordingProgressReporter()
        result = coerce_reporter(recording)
        assert isinstance(result, ErrorSwallowingReporter)
        assert result._delegate is recording

    def test_wrapped_reporter_forwards_calls(self):
        recording = RecordingProgressReporter()
        result = coerce_reporter(recording)
        result.start_phase("p", 3, message="m")
        result.update(1, message="u")
        result.end_phase()

        assert recording.events == [
            ("start_phase", ("p", 3), {"message": "m"}),
            ("update", (1,), {"message": "u"}),
            ("end_phase", (), {}),
        ]


class TestErrorSwallowingReporter:
    def test_swallows_start_phase_exception(self, caplog: pytest.LogCaptureFixture):
        reporter = ErrorSwallowingReporter(RaisingProgressReporter())
        with caplog.at_level(logging.WARNING, logger="sortition_algorithms"):
            reporter.start_phase("phase_x", 10, message="m")

        assert any("start_phase" in record.message for record in caplog.records)
        assert any("phase_x" in record.message for record in caplog.records)

    def test_swallows_update_exception(self, caplog: pytest.LogCaptureFixture):
        reporter = ErrorSwallowingReporter(RaisingProgressReporter())
        with caplog.at_level(logging.WARNING, logger="sortition_algorithms"):
            reporter.update(7, message="m")

        assert any("update" in record.message and "7" in record.message for record in caplog.records)

    def test_swallows_end_phase_exception(self, caplog: pytest.LogCaptureFixture):
        reporter = ErrorSwallowingReporter(RaisingProgressReporter())
        with caplog.at_level(logging.WARNING, logger="sortition_algorithms"):
            reporter.end_phase()

        assert any("end_phase" in record.message for record in caplog.records)

    def test_logs_include_traceback(self, caplog: pytest.LogCaptureFixture):
        reporter = ErrorSwallowingReporter(RaisingProgressReporter())
        with caplog.at_level(logging.WARNING, logger="sortition_algorithms"):
            reporter.start_phase("p")

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records
        assert any(r.exc_info is not None for r in warning_records)

    def test_does_not_swallow_when_delegate_is_well_behaved(self):
        recording = RecordingProgressReporter()
        reporter = ErrorSwallowingReporter(recording)
        reporter.start_phase("ok", 5, message="m")
        reporter.update(2, message="u")
        reporter.end_phase()

        assert recording.events == [
            ("start_phase", ("ok", 5), {"message": "m"}),
            ("update", (2,), {"message": "u"}),
            ("end_phase", (), {}),
        ]


class TestPhaseContextManager:
    def test_calls_start_then_end_on_normal_exit(self):
        recording = RecordingProgressReporter()
        with phase(recording, "phase_a", 5, message="starting"):
            recording.update(2, message="halfway")

        assert recording.events == [
            ("start_phase", ("phase_a", 5), {"message": "starting"}),
            ("update", (2,), {"message": "halfway"}),
            ("end_phase", (), {}),
        ]

    def test_yields_the_reporter(self):
        recording = RecordingProgressReporter()
        with phase(recording, "p") as yielded:
            assert yielded is recording

    def test_calls_end_phase_when_body_raises(self):
        recording = RecordingProgressReporter()
        with pytest.raises(ValueError, match="boom"), phase(recording, "phase_a", message="starting"):
            raise ValueError("boom")

        assert recording.events == [
            ("start_phase", ("phase_a", None), {"message": "starting"}),
            ("end_phase", (), {}),
        ]

    def test_total_defaults_to_none(self):
        recording = RecordingProgressReporter()
        with phase(recording, "p"):
            pass

        assert recording.events[0] == ("start_phase", ("p", None), {"message": None})
