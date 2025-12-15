# ABOUTME: Tests for report message internationalization support
# ABOUTME: Verifies that RunReport correctly stores message codes and parameters for translation

from sortition_algorithms.report_messages import REPORT_MESSAGES, get_message
from sortition_algorithms.utils import ReportLevel, RunReport


class TestReportMessages:
    """Test report message template system."""

    def test_get_message_simple(self):
        """Test get_message with a simple message (no parameters)."""
        msg = get_message("loading_features_from_string")
        assert msg == "Loading features from string."

    def test_get_message_with_params(self):
        """Test get_message with parameters."""
        msg = get_message("loading_features_from_file", file_path="/path/to/features.csv")
        assert msg == "Loading features from file /path/to/features.csv."

    def test_get_message_with_multiple_params(self):
        """Test get_message with multiple parameters."""
        msg = get_message("distribution_stats", total_committees=150, non_zero_committees=42)
        assert msg == (
            "Algorithm produced distribution over 150 committees, out of which 42 are chosen with positive probability."
        )

    def test_all_messages_have_templates(self):
        """Verify all message codes have valid templates."""
        for code, template in REPORT_MESSAGES.items():
            assert isinstance(code, str)
            assert isinstance(template, str)
            assert len(code) > 0
            assert len(template) > 0


class TestRunReportTranslationSupport:
    """Test that RunReport stores translation data correctly."""

    def test_add_line_with_message_code(self):
        """Test that add_line stores message code and params."""
        report = RunReport()
        report.add_line(
            "Loading features from file /path/to/file.csv",
            message_code="loading_features_from_file",
            message_params={"file_path": "/path/to/file.csv"},
        )

        # Verify the message code and params are stored
        assert len(report._data) == 1
        line_level = report._data[0]
        assert line_level.message_code == "loading_features_from_file"
        assert line_level.message_params == {"file_path": "/path/to/file.csv"}
        assert line_level.line == "Loading features from file /path/to/file.csv"

    def test_add_line_without_message_code(self):
        """Test that add_line works without message code (backward compatibility)."""
        report = RunReport()
        report.add_line("Some custom message")

        assert len(report._data) == 1
        line_level = report._data[0]
        assert line_level.message_code is None
        assert line_level.message_params == {}
        assert line_level.line == "Some custom message"

    def test_add_line_and_log_with_message_code(self):
        """Test that add_line_and_log stores message code and params."""
        report = RunReport()
        import logging

        report.add_line_and_log(
            "Trial number: 5",
            logging.WARNING,
            message_code="trial_number",
            message_params={"trial": 5},
        )

        assert len(report._data) == 1
        line_level = report._data[0]
        assert line_level.message_code == "trial_number"
        assert line_level.message_params == {"trial": 5}
        assert line_level.line == "Trial number: 5"

    def test_translation_workflow_example(self):
        """
        Demonstrate how a web app would use message codes for translation.

        This test shows the intended usage pattern for i18n in web applications.
        """
        report = RunReport()

        # Library adds messages with both English text and translation data
        report.add_line(
            get_message("features_found", count=5),
            message_code="features_found",
            message_params={"count": 5},
        )

        report.add_line(
            get_message("trial_number", trial=3),
            ReportLevel.IMPORTANT,
            message_code="trial_number",
            message_params={"trial": 3},
        )

        # Web app can iterate through messages and translate them
        translated_messages = []
        for element in report._data:
            if hasattr(element, "message_code") and element.message_code:
                # In a real web app, this would be something like:
                # translated = _(f"report.{element.message_code}") % element.message_params

                # For this test, we'll just verify the data is available
                assert element.message_code in ["features_found", "trial_number"]
                assert isinstance(element.message_params, dict)

                # Simulate translation
                if element.message_code == "features_found":
                    # French translation example
                    translated = f"Nombre de fonctionnalités trouvées: {element.message_params['count']}"
                elif element.message_code == "trial_number":
                    # French translation example
                    translated = f"Numéro d'essai: {element.message_params['trial']}"

                translated_messages.append(translated)
            else:
                # Fallback to English for messages without translation data
                translated_messages.append(element.line)

        assert len(translated_messages) == 2
        assert "Nombre de fonctionnalités trouvées: 5" in translated_messages
        assert "Numéro d'essai: 3" in translated_messages

    def test_serialization_includes_translation_data(self):
        """Test that serialization preserves message code and params."""
        report = RunReport()
        report.add_line(
            get_message("loading_features_from_file", file_path="/test.csv"),
            message_code="loading_features_from_file",
            message_params={"file_path": "/test.csv"},
        )

        serialized = report.serialize()

        assert len(serialized["_data"]) == 1
        assert serialized["_data"][0]["message_code"] == "loading_features_from_file"
        assert serialized["_data"][0]["message_params"] == {"file_path": "/test.csv"}

    def test_deserialization_preserves_translation_data(self):
        """Test that deserialization preserves message code and params."""
        serialized = {
            "_data": [
                {
                    "line": "Loading features from file /test.csv.",
                    "level": 0,
                    "log_level": 0,
                    "message_code": "loading_features_from_file",
                    "message_params": {"file_path": "/test.csv"},
                }
            ]
        }

        report = RunReport.deserialize(serialized)

        assert len(report._data) == 1
        line_level = report._data[0]
        assert line_level.message_code == "loading_features_from_file"
        assert line_level.message_params == {"file_path": "/test.csv"}
        assert line_level.line == "Loading features from file /test.csv."
