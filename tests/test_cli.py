import csv
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from sortition_algorithms.__main__ import cli
from tests.helpers import candidates_csv_path, create_settings_file_for_fixtures, features_csv_path


def get_rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file, strict=True)
        rows = list(reader)
        return rows


@pytest.mark.slow
def test_csv_happy_path(tmp_path):
    runner = CliRunner()
    settings_file = tmp_path / "settings.toml"
    create_settings_file_for_fixtures(settings_file)

    # files to write to
    selected_csv_path = tmp_path / "selected.csv"
    remaining_csv_path = tmp_path / "remaining.csv"

    result = runner.invoke(
        cli,
        [
            "csv",
            f"--settings={settings_file}",
            f"--features-csv={features_csv_path}",
            f"--people-csv={candidates_csv_path}",
            f"--selected-csv={selected_csv_path}",
            f"--remaining-csv={remaining_csv_path}",
            "--number-wanted=22",
        ],
    )
    assert result.exit_code == 0, f"Exit code: {result.exit_code}, output: {result.output}"

    selected_rows = get_rows_from_csv(selected_csv_path)
    assert len(selected_rows) == 22


def test_gen_sample_happy_path(tmp_path):
    runner = CliRunner()
    settings_file = tmp_path / "settings.toml"
    create_settings_file_for_fixtures(settings_file)

    # this file will be written to:
    people_csv_path = tmp_path / "people.csv"

    result = runner.invoke(
        cli,
        [
            "gen-sample",
            f"--settings={settings_file}",
            f"--features-csv={features_csv_path}",
            f"--people-csv={people_csv_path}",
            "--number-wanted=10",
        ],
    )
    assert result.exit_code == 0, f"Exit code: {result.exit_code}, output: {result.output}"

    people_rows = get_rows_from_csv(people_csv_path)
    assert len(people_rows) == 10
