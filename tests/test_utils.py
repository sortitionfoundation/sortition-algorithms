import logging
import re
from typing import ClassVar
from unittest.mock import patch

import pytest

from sortition_algorithms.errors import SelectionError, SelectionMultilineError
from sortition_algorithms.utils import ReportLevel, RunReport, get_cell_name


class TestGetCellName:
    short_headers = ("first", "second", "third")
    long_headers: ClassVar = [f"col{index}" for index in range(1, 100)]

    def test_get_cell_name_a1(self):
        assert get_cell_name(1, "first", self.short_headers) == "A1"

    def test_get_cell_name_z100(self):
        assert get_cell_name(100, "col26", self.long_headers) == "Z100"

    @pytest.mark.parametrize(
        "col_name,expected",
        [
            ("col27", "AA10"),
            ("col28", "AB10"),
            ("col52", "AZ10"),
            ("col53", "BA10"),
            ("col78", "BZ10"),
            ("col79", "CA10"),
        ],
    )
    def test_get_cell_name_beyond_z(self, col_name: str, expected: str):
        assert get_cell_name(10, col_name, self.long_headers) == expected

    def test_get_cell_name_not_in_headers(self):
        with pytest.raises(ValueError):
            get_cell_name(10, "unknown", self.short_headers)


class TestRunReport:
    def test_empty_report(self):
        report = RunReport()
        assert report.as_text() == ""
        assert report.as_html() == ""
        assert not bool(report)
        assert not report.has_content()

    def test_single_line_normal_level(self):
        report = RunReport()
        report.add_line("Test message")
        assert report.as_text() == "Test message"
        assert report.as_html() == "Test message"
        assert bool(report)
        assert report.has_content()

    def test_single_line_important_level(self):
        report = RunReport()
        report.add_line("Important message", ReportLevel.IMPORTANT)
        assert report.as_text() == "Important message"
        assert report.as_html() == "<b>Important message</b>"

    def test_single_line_critical_level(self):
        report = RunReport()
        report.add_line("Critical message", ReportLevel.CRITICAL)
        assert report.as_text() == "Critical message"
        assert report.as_html() == '<b style="color: red">Critical message</b>'

    def test_multiple_lines_mixed_levels(self):
        report = RunReport()
        report.add_line("Normal message")
        report.add_line("Important message", ReportLevel.IMPORTANT)
        report.add_line("Critical message", ReportLevel.CRITICAL)
        report.add_line("Another normal message")

        expected_text = "Normal message\nImportant message\nCritical message\nAnother normal message"
        assert report.as_text() == expected_text

        expected_html = (
            "Normal message<br />\n"
            "<b>Important message</b><br />\n"
            '<b style="color: red">Critical message</b><br />\n'
            "Another normal message"
        )
        assert report.as_html() == expected_html

    def test_multiple_lines_same_level(self):
        report = RunReport()
        report.add_line("First critical", ReportLevel.CRITICAL)
        report.add_line("Second critical", ReportLevel.CRITICAL)

        expected_text = "First critical\nSecond critical"
        assert report.as_text() == expected_text

        expected_html = '<b style="color: red">First critical</b><br />\n<b style="color: red">Second critical</b>'
        assert report.as_html() == expected_html

    def test_empty_lines(self):
        report = RunReport()
        report.add_line("")
        report.add_line("", ReportLevel.IMPORTANT)
        report.add_line("", ReportLevel.CRITICAL)

        expected_text = "\n\n"
        assert report.as_text() == expected_text

        expected_html = '<br />\n<b></b><br />\n<b style="color: red"></b>'
        assert report.as_html() == expected_html

    def test_lines_with_special_html_characters(self):
        report = RunReport()
        report.add_line("Message with <tags> & ampersands")
        report.add_line("Important with <script>", ReportLevel.IMPORTANT)
        report.add_error(SelectionError("This & that"))

        # Text output should preserve characters as-is
        expected_text = "Message with <tags> & ampersands\nImportant with <script>\nThis & that"
        assert report.as_text() == expected_text

        # HTML output should escape special characters
        expected_html = (
            "Message with &lt;tags&gt; &amp; ampersands<br />\n<b>Important with &lt;script&gt;</b><br />\n"
            "<b>This &amp; that</b>"
        )
        assert report.as_html() == expected_html

    def test_multi_line_errors(self):
        report = RunReport()
        report.add_line("straight")
        report.add_error(SelectionMultilineError(["line 1", "line 2", "line 3"]))
        report.add_line("curved")

        expected_text = "straight\nline 1\nline 2\nline 3\ncurved"
        assert report.as_text() == expected_text

        expected_html = "straight<br />\n<b>line 1<br />line 2<br />line 3</b><br />\ncurved"
        assert report.as_html() == expected_html

    def test_default_level_is_normal(self):
        report = RunReport()
        report.add_line("Default level message")

        # Should behave same as explicitly setting NORMAL
        report_explicit = RunReport()
        report_explicit.add_line("Default level message", ReportLevel.NORMAL)

        assert report.as_text() == report_explicit.as_text()
        assert report.as_html() == report_explicit.as_html()

    def test_simple_table(self):
        report = RunReport()
        headers = ["Name", "Age", "City"]
        data = [["Alice", 25, "NYC"], ["Bob", 30, "LA"]]
        report.add_table(headers, data)

        # Text output uses tabulate with simple format
        text_output = report.as_text()
        assert "Name" in text_output
        assert "Age" in text_output
        assert "City" in text_output
        assert "Alice" in text_output
        assert "Bob" in text_output
        # Should have blank lines before and after table
        assert text_output.startswith("\n")
        assert text_output.endswith("\n")

        # HTML output uses tabulate with html format
        html_output = report.as_html()
        assert "<table>" in html_output
        assert re.search(r"<th[^>]*>Name", html_output)
        assert re.search(r"<td[^>]*>Alice", html_output)

    def test_table_with_mixed_data_types(self):
        report = RunReport()
        headers = ["Product", "Price", "In Stock"]
        data = [["Widget", 19.99, 100], ["Gadget", 29.50, 0]]
        report.add_table(headers, data)

        text_output = report.as_text()
        assert "19.99" in text_output
        assert "100" in text_output

        html_output = report.as_html()
        assert re.search(r"<td[^>]*>\s*19\.99", html_output)
        assert re.search(r"<td[^>]*>\s*100", html_output)

    def test_empty_table(self):
        report = RunReport()
        headers = ["Column1", "Column2"]
        data = []
        report.add_table(headers, data)

        text_output = report.as_text()
        assert "Column1" in text_output
        assert "Column2" in text_output

        html_output = report.as_html()
        # Empty table might not have headers in the HTML output, so just check for table structure
        assert "<table>" in html_output
        assert "</table>" in html_output

    def test_mixed_lines_and_tables(self):
        report = RunReport()
        report.add_line("Introduction")
        report.add_table(["Name", "Score"], [["Alice", 95], ["Bob", 87]])
        report.add_line("Summary", ReportLevel.IMPORTANT)
        report.add_error(SelectionError("It blew up"))
        report.add_error(SelectionError("passing problem"), is_fatal=False)

        text_output = report.as_text()
        assert "Introduction" in text_output
        assert "Alice" in text_output
        assert "Summary" in text_output
        assert "It blew up" in text_output
        assert "passing problem" in text_output

        html_output = report.as_html()
        assert "Introduction<br />" in html_output
        assert "<table>" in html_output
        assert "<b>Summary</b>" in html_output
        assert "<b>It blew up</b>" in html_output
        assert "passing problem" in html_output
        assert "<b>passing problem</b>" not in html_output

    def test_multiple_tables(self):
        report = RunReport()
        report.add_table(["A", "B"], [["1", "2"]])
        report.add_table(["X", "Y"], [["10", "20"]])

        text_output = report.as_text()
        # Should have both tables separated by newlines
        assert text_output.count("A") == 1
        assert text_output.count("X") == 1
        assert "1" in text_output
        assert "10" in text_output

        html_output = report.as_html()
        # Should have two separate HTML tables
        assert html_output.count("<table>") == 2

    @patch("sortition_algorithms.utils.user_logger")
    def test_add_line_with_logging(self, mock_user_logger):
        report = RunReport()
        report.add_line_and_log("Test message", logging.INFO)

        # Should log the message
        mock_user_logger.log.assert_called_once_with(level=logging.INFO, msg="Test message")

        # Should include in report by default
        assert "Test message" in report.as_text()
        assert "Test message" in report.as_html()

    @patch("sortition_algorithms.utils.user_logger")
    def test_add_line_without_logging(self, mock_user_logger):
        report = RunReport()
        report.add_line("Test message")  # No log_level specified

        # Should not call logger
        mock_user_logger.log.assert_not_called()

        # Should include in report
        assert "Test message" in report.as_text()
        assert "Test message" in report.as_html()

    @patch("sortition_algorithms.utils.user_logger")
    def test_include_logged_parameter(self, mock_user_logger):
        report = RunReport()
        report.add_line("Normal message")  # No logging
        report.add_line_and_log("Logged message", logging.INFO)  # With logging

        # With include_logged=True (default)
        text_with_logged = report.as_text()
        html_with_logged = report.as_html()
        assert "Normal message" in text_with_logged
        assert "Logged message" in text_with_logged
        assert "Normal message" in html_with_logged
        assert "Logged message" in html_with_logged

        # With include_logged=False
        text_without_logged = report.as_text(include_logged=False)
        html_without_logged = report.as_html(include_logged=False)
        assert "Normal message" in text_without_logged
        assert "Logged message" not in text_without_logged
        assert "Normal message" in html_without_logged
        assert "Logged message" not in html_without_logged

    def test_add_lines_method(self):
        report = RunReport()
        lines = ["First line", "Second line", "Third line"]
        report.add_lines(lines, ReportLevel.IMPORTANT)

        text_output = report.as_text()
        html_output = report.as_html()

        for line in lines:
            assert line in text_output
            assert line in html_output

        # Check HTML formatting for important level
        assert "<b>First line</b>" in html_output
        assert "<b>Second line</b>" in html_output
        assert "<b>Third line</b>" in html_output

    def test_add_report_method(self):
        report1 = RunReport()
        report1.add_line("From report 1")
        report1.add_table(["Col"], [["Data1"]])

        report2 = RunReport()
        report2.add_line("From report 2", ReportLevel.CRITICAL)

        combined = RunReport()
        combined.add_line("Combined header")
        combined.add_report(report1)
        combined.add_report(report2)

        text_output = combined.as_text()
        html_output = combined.as_html()

        # All content should be present
        assert "Combined header" in text_output
        assert "From report 1" in text_output
        assert "From report 2" in text_output
        assert "Data1" in text_output

        assert "Combined header" in html_output
        assert "From report 1" in html_output
        assert '<b style="color: red">From report 2</b>' in html_output
        assert "<table>" in html_output

    @patch("sortition_algorithms.utils.user_logger")
    def test_mixed_logged_and_unlogged_with_tables(self, mock_user_logger):
        report = RunReport()
        report.add_line("Unlogged line")
        report.add_line_and_log("Logged line", logging.DEBUG)
        report.add_table(["Header"], [["Row1"]])
        report.add_line("Another unlogged", ReportLevel.IMPORTANT)

        # Tables should always appear regardless of include_logged
        text_with_logged = report.as_text(include_logged=True)
        text_without_logged = report.as_text(include_logged=False)

        # Tables should be in both
        assert "Header" in text_with_logged
        assert "Header" in text_without_logged
        assert "Row1" in text_with_logged
        assert "Row1" in text_without_logged

        # Unlogged lines should be in both
        assert "Unlogged line" in text_with_logged
        assert "Unlogged line" in text_without_logged
        assert "Another unlogged" in text_with_logged
        assert "Another unlogged" in text_without_logged

        # Logged line should only be in include_logged=True version
        assert "Logged line" in text_with_logged
        assert "Logged line" not in text_without_logged

    @patch("sortition_algorithms.utils.user_logger")
    def test_empty_report_with_only_logged_lines(self, mock_user_logger):
        report = RunReport()
        report.add_line_and_log("Only logged", logging.INFO)

        # With logged lines included
        assert report.as_text(include_logged=True) == "Only logged"
        assert report.as_html(include_logged=True) == "Only logged"

        # Without logged lines - should be empty
        assert report.as_text(include_logged=False) == ""
        assert report.as_html(include_logged=False) == ""

    def test_last_error(self):
        """Check we get the last error added"""
        report = RunReport()
        report.add_error(SelectionError("passing problem"))
        report.add_line("Only logged")
        report.add_error(SelectionError("BIG problem"))
        report.add_line("We're outta here")
        assert str(report.last_error()) == "BIG problem"

    def test_last_error_when_no_error_added(self):
        """Check we get None if no error added"""
        report = RunReport()
        report.add_line("Only logged")
        assert report.last_error() is None
