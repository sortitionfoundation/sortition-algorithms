import pytest

from sortition_algorithms.utils import ReportLevel, RunReport


class TestRunReport:
    def test_empty_report(self):
        report = RunReport()
        assert report.as_text() == ""
        assert report.as_html() == ""

    def test_single_line_normal_level(self):
        report = RunReport()
        report.add_line("Test message")
        assert report.as_text() == "Test message"
        assert report.as_html() == "Test message"

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

        # Text output should preserve characters as-is
        expected_text = "Message with <tags> & ampersands\nImportant with <script>"
        assert report.as_text() == expected_text

        # HTML output preserves the characters (no escaping is done)
        expected_html = "Message with &lt;tags&gt; &amp; ampersands<br />\n<b>Important with &lt;script&gt;</b>"
        assert report.as_html() == expected_html

    def test_default_level_is_normal(self):
        report = RunReport()
        report.add_line("Default level message")

        # Should behave same as explicitly setting NORMAL
        report_explicit = RunReport()
        report_explicit.add_line("Default level message", ReportLevel.NORMAL)

        assert report.as_text() == report_explicit.as_text()
        assert report.as_html() == report_explicit.as_html()

    def test_add_table_not_implemented(self):
        report = RunReport()
        with pytest.raises(NotImplementedError):
            report.add_table("some table")
