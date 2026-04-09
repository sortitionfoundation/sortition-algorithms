"""ABOUTME: Tests for the progress reporting protocol and helpers in progress.py.
ABOUTME: Includes a recording reporter used here and reused by later phases.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from sortition_algorithms import progress
from sortition_algorithms.committee_generation import find_distribution_maximin, find_distribution_nash
from sortition_algorithms.committee_generation.diversimax import DIVERSIMAX_AVAILABLE, find_distribution_diversimax
from sortition_algorithms.committee_generation.legacy import find_random_sample_legacy
from sortition_algorithms.committee_generation.leximin import GUROBI_AVAILABLE, find_distribution_leximin
from sortition_algorithms.progress import (
    ErrorSwallowingReporter,
    NullProgressReporter,
    ProgressReporter,
    coerce_reporter,
    phase,
)
from sortition_algorithms.progress_rich import RichProgressReporter
from tests.helpers import create_simple_features, create_simple_people, create_test_settings


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


def _phase_names(reporter: "RecordingProgressReporter") -> list[str]:
    return [e[1][0] for e in reporter.events if e[0] == "start_phase"]


def _events_of_type(reporter: "RecordingProgressReporter", kind: str) -> list[tuple]:
    return [e for e in reporter.events if e[0] == kind]


def _make_simple_selection_data(person_count: int = 8):
    features = create_simple_features(gender_min=1, gender_max=4, age_min=1, age_max=4)
    settings = create_test_settings(columns_to_keep=["name", "email"])
    people = create_simple_people(features, settings, count=person_count)
    return features, people


class TestLegacyProgressEvents:
    def test_records_phase_events_for_successful_run(self):
        features, people = _make_simple_selection_data()
        reporter = RecordingProgressReporter()

        find_random_sample_legacy(
            people, features, 4, check_same_address_columns=[], max_attempts=5, progress_reporter=reporter
        )

        names = _phase_names(reporter)
        assert names == ["legacy_attempt"]

        starts = _events_of_type(reporter, "start_phase")
        assert starts[0] == (
            "start_phase",
            ("legacy_attempt", 5),
            {"message": "Running legacy algorithm (up to 5 attempts)"},
        )

        updates = _events_of_type(reporter, "update")
        assert len(updates) >= 1
        assert updates[0][1] == (1,)
        assert "Legacy algorithm attempt 1/5" in updates[0][2]["message"]

        ends = _events_of_type(reporter, "end_phase")
        assert len(ends) == 1

    def test_legacy_emits_diagnostic_log_lines(self, caplog: pytest.LogCaptureFixture):
        features, people = _make_simple_selection_data()
        reporter = RecordingProgressReporter()

        with caplog.at_level(logging.WARNING, logger="sortition_algorithms_user"):
            find_random_sample_legacy(
                people, features, 4, check_same_address_columns=[], max_attempts=3, progress_reporter=reporter
            )

        assert any("Trial number" in r.message for r in caplog.records)


@pytest.mark.slow
class TestMaximinProgressEvents:
    def test_records_phase_events(self):
        features, people = _make_simple_selection_data()
        reporter = RecordingProgressReporter()

        find_distribution_maximin(features, people, 4, check_same_address_columns=[], progress_reporter=reporter)

        names = _phase_names(reporter)
        assert "multiplicative_weights" in names
        assert "maximin_optimization" in names
        # multiplicative_weights phase runs before maximin_optimization
        assert names.index("multiplicative_weights") < names.index("maximin_optimization")

        # multiplicative_weights starts with the right total (people.count)
        mw_start = next(e for e in reporter.events if e[0] == "start_phase" and e[1][0] == "multiplicative_weights")
        assert mw_start[1][1] == people.count

        # maximin_optimization is a convergence loop, total=None
        mx_start = next(e for e in reporter.events if e[0] == "start_phase" and e[1][0] == "maximin_optimization")
        assert mx_start[1][1] is None

        # both phases end
        ends = _events_of_type(reporter, "end_phase")
        assert len(ends) == 2

    def test_maximin_emits_diagnostic_log_lines(self, caplog: pytest.LogCaptureFixture):
        features, people = _make_simple_selection_data()
        reporter = RecordingProgressReporter()

        with caplog.at_level(logging.DEBUG, logger="sortition_algorithms"):
            find_distribution_maximin(features, people, 4, check_same_address_columns=[], progress_reporter=reporter)

        assert any("Multiplicative weights phase" in r.message for r in caplog.records)


@pytest.mark.slow
class TestNashProgressEvents:
    def test_records_phase_events(self):
        features, people = _make_simple_selection_data()
        reporter = RecordingProgressReporter()

        find_distribution_nash(features, people, 4, check_same_address_columns=[], progress_reporter=reporter)

        names = _phase_names(reporter)
        assert "multiplicative_weights" in names
        assert "nash_optimization" in names
        assert names.index("multiplicative_weights") < names.index("nash_optimization")

        nash_start = next(e for e in reporter.events if e[0] == "start_phase" and e[1][0] == "nash_optimization")
        assert nash_start[1][1] is None

        ends = _events_of_type(reporter, "end_phase")
        assert len(ends) == 2


@pytest.mark.slow
@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Leximin requires Gurobi")
class TestLeximinProgressEvents:
    def test_records_phase_events(self):
        features, people = _make_simple_selection_data()
        reporter = RecordingProgressReporter()

        find_distribution_leximin(features, people, 4, check_same_address_columns=[], progress_reporter=reporter)

        names = _phase_names(reporter)
        assert "multiplicative_weights" in names
        assert "leximin_outer" in names

        leximin_start = next(e for e in reporter.events if e[0] == "start_phase" and e[1][0] == "leximin_outer")
        assert leximin_start[1][1] == people.count


@pytest.mark.slow
@pytest.mark.skipif(not DIVERSIMAX_AVAILABLE, reason="Diversimax optional dependencies missing")
class TestDiversimaxProgressEvents:
    def test_records_single_phase_no_updates(self):
        features, people = _make_simple_selection_data()
        reporter = RecordingProgressReporter()

        find_distribution_diversimax(features, people, 4, check_same_address_columns=[], progress_reporter=reporter)

        names = _phase_names(reporter)
        assert names == ["diversimax"]

        start = next(e for e in reporter.events if e[0] == "start_phase")
        assert start[1] == ("diversimax", None)
        assert start[2]["message"].startswith("Running diversimax")

        # diversimax should emit no per-iteration updates
        updates = _events_of_type(reporter, "update")
        assert updates == []

        ends = _events_of_type(reporter, "end_phase")
        assert len(ends) == 1


class TestRaisingReporterDoesNotBreakSelection:
    def test_legacy_run_succeeds_with_raising_reporter(self, caplog: pytest.LogCaptureFixture):
        features, people = _make_simple_selection_data()
        raising = RaisingProgressReporter()

        with caplog.at_level(logging.WARNING, logger="sortition_algorithms"):
            committees, _ = find_random_sample_legacy(
                people, features, 4, check_same_address_columns=[], max_attempts=5, progress_reporter=raising
            )

        assert len(committees) == 1
        assert len(committees[0]) == 4
        # exceptions from the reporter should have been swallowed and logged
        assert any("Progress reporter raised" in r.message for r in caplog.records)

    @pytest.mark.slow
    def test_maximin_run_succeeds_with_raising_reporter(self, caplog: pytest.LogCaptureFixture):
        features, people = _make_simple_selection_data()
        raising = RaisingProgressReporter()

        with caplog.at_level(logging.WARNING, logger="sortition_algorithms"):
            committees, probabilities, _ = find_distribution_maximin(
                features, people, 4, check_same_address_columns=[], progress_reporter=raising
            )

        assert len(committees) >= 1
        assert abs(sum(probabilities) - 1.0) < 1e-5
        assert any("Progress reporter raised" in r.message for r in caplog.records)


class TestRichProgressReporter:
    def test_satisfies_protocol(self):
        assert isinstance(RichProgressReporter(), ProgressReporter)

    def test_context_manager_handles_full_lifecycle(self):
        with RichProgressReporter() as reporter:
            reporter.start_phase("phase_a", total=10, message="starting")
            reporter.update(5, message="halfway")
            reporter.update(10, message="done")
            reporter.end_phase()

    def test_handles_phase_with_no_total(self):
        with RichProgressReporter() as reporter:
            reporter.start_phase("phase_b", total=None, message="convergence loop")
            reporter.update(1)
            reporter.update(2)
            reporter.end_phase()

    def test_phase_replacement(self):
        with RichProgressReporter() as reporter:
            reporter.start_phase("phase_a", total=5, message="first")
            reporter.update(2)
            reporter.start_phase("phase_b", total=10, message="second")
            reporter.update(3)
            reporter.end_phase()

    def test_end_phase_before_start_is_noop(self):
        with RichProgressReporter() as reporter:
            reporter.end_phase()
            reporter.update(1)
