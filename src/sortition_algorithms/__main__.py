import logging
from pathlib import Path

import click

from sortition_algorithms import adapters, core, people_features
from sortition_algorithms.settings import Settings
from sortition_algorithms.utils import RunReport, set_log_level


def echo_all(msgs: list[str]) -> None:
    for msg in msgs:
        click.echo(msg)


def echo_report(report: RunReport) -> None:
    click.echo(report.as_text())


@click.group()
def cli() -> None:
    """A command line tool to exercise the sortition algorithms."""
    pass


@cli.command()
@click.option(
    "-S",
    "--settings",
    envvar="SORTITION_SETTINGS",
    type=click.Path(dir_okay=False),
    required=True,
    help="Settings for the sortition run. Will auto-create if not present.",
)
@click.option(
    "-f",
    "--features-csv",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to CSV with features defined.",
)
@click.option(
    "-p",
    "--people-csv",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to CSV with people defined.",
)
@click.option(
    "-s",
    "--selected-csv",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
    help="Path to CSV file to write selected people to.",
)
@click.option(
    "-r",
    "--remaining-csv",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
    help="Path to CSV file to write remaining people to.",
)
@click.option(
    "-n",
    "--number-wanted",
    type=click.IntRange(min=1),
    required=True,
    help="Number of people to select.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="If used, produce extra detailed logging.",
)
def csv(
    settings: str,
    features_csv: str,
    people_csv: str,
    selected_csv: str,
    remaining_csv: str,
    number_wanted: int,
    verbose: bool,
) -> None:
    """Do sortition with CSV files."""
    if verbose:
        set_log_level(logging.DEBUG)
    adapter = adapters.CSVAdapter()
    settings_obj, report = Settings.load_from_file(Path(settings))
    echo_report(report)
    features, report = adapter.load_features_from_file(Path(features_csv))
    echo_report(report)

    people, report = adapter.load_people_from_file(Path(people_csv), settings_obj, features)
    echo_report(report)

    success, people_selected, report = core.run_stratification(features, people, number_wanted, settings_obj)
    echo_report(report)
    if not success:
        raise click.ClickException("Selection not successful, no files written.")

    selected_rows, remaining_rows, _ = core.selected_remaining_tables(
        people, people_selected[0], features, settings_obj
    )
    with (
        open(selected_csv, "w", newline="") as selected_f,
        open(remaining_csv, "w", newline="") as remaining_f,
    ):
        adapter.selected_file = selected_f
        adapter.remaining_file = remaining_f
        adapter.output_selected_remaining(selected_rows, remaining_rows)


@cli.command()
@click.option(
    "-S",
    "--settings",
    envvar="SORTITION_SETTINGS",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Settings for the sortition run. Will auto-create if not present.",
)
@click.option(
    "--auth-json-file",
    envvar="SORTITION_GDOC_AUTH",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to file with OAuth2 details to access google account.",
)
@click.option("--gen-rem-tab/--no-gen-rem-tab", default=True, help="Generate a 'Remaining' tab.")
@click.option("-g", "--gsheet-name", required=True, help="Name of GDoc Spreadsheet to use.")
@click.option(
    "-f",
    "--feature-tab-name",
    default="Categories",
    required=True,
    help="Name of tab containing features/categories.",
)
@click.option(
    "-p",
    "--people-tab-name",
    default="Categories",
    required=True,
    help="Name of tab containing people/respondents.",
)
@click.option(
    "-s",
    "--selected-tab-name",
    default="Selected",
    required=True,
    help="Name of tab to write selected people to.",
)
@click.option(
    "-r",
    "--remaining-tab-name",
    default="Remaining",
    help="Name of tab to write remaining people to.",
)
@click.option(
    "-n",
    "--number-wanted",
    type=click.IntRange(min=1),
    required=True,
    help="Number of people to select.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="If used, produce extra detailed logging.",
)
def gsheet(
    settings: str,
    auth_json_file: str,
    gen_rem_tab: bool,
    gsheet_name: str,
    feature_tab_name: str,
    people_tab_name: str,
    selected_tab_name: str,
    remaining_tab_name: str,
    number_wanted: int,
    verbose: bool,
) -> None:
    """Do sortition with Google Spreadsheets."""
    if verbose:
        set_log_level(logging.DEBUG)
    adapter = adapters.GSheetAdapter(Path(auth_json_file), gen_rem_tab)
    settings_obj, report = Settings.load_from_file(Path(settings))
    echo_report(report)

    adapter.set_g_sheet_name(gsheet_name)
    features, report = adapter.load_features(feature_tab_name)
    echo_report(report)
    if features is None:
        raise click.ClickException("Could not load features, exiting.")

    people, report = adapter.load_people(people_tab_name, settings_obj, features)
    echo_report(report)
    if people is None:
        raise click.ClickException("Could not load people, exiting.")

    success, people_selected, report = core.run_stratification(features, people, number_wanted, settings_obj)
    echo_report(report)
    if not success:
        raise click.ClickException("Selection not successful, no files written.")

    selected_rows, remaining_rows, _ = core.selected_remaining_tables(
        people, people_selected[0], features, settings_obj
    )
    adapter.selected_tab_name = selected_tab_name
    adapter.remaining_tab_name = remaining_tab_name
    adapter.output_selected_remaining(selected_rows, remaining_rows, settings_obj)


@cli.command()
@click.option(
    "-S",
    "--settings",
    envvar="SORTITION_SETTINGS",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Settings for the sortition run. Will auto-create if not present.",
)
@click.option(
    "-f",
    "--features-csv",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to CSV with features defined.",
)
@click.option(
    "-p",
    "--people-csv",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
    help="Path to CSV to write sample people to.",
)
@click.option(
    "-n",
    "--number-wanted",
    type=click.IntRange(min=1),
    required=True,
    help="Number of people to write.",
)
def gen_sample(settings: str, features_csv: str, people_csv: str, number_wanted: int) -> None:
    """Generate a sample CSV file of people compatible with features and settings."""
    adapter = adapters.CSVAdapter()
    settings_obj, report = Settings.load_from_file(Path(settings))
    echo_report(report)
    features, report = adapter.load_features_from_file(Path(features_csv))
    echo_report(report)
    with open(people_csv, "w", newline="") as people_f:
        people_features.create_readable_sample_file(features, people_f, number_wanted, settings_obj)
