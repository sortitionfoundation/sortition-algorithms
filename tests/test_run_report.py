import re

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

        # HTML output should escape special characters
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

        text_output = report.as_text()
        assert "Introduction" in text_output
        assert "Alice" in text_output
        assert "Summary" in text_output

        html_output = report.as_html()
        assert "Introduction<br />" in html_output
        assert "<table>" in html_output
        assert "<b>Summary</b>" in html_output

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
