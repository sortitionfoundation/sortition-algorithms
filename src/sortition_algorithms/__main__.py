from pathlib import Path

import click

from sortition_algorithms import adapters, core, people_features
from sortition_algorithms.settings import Settings


def echo_all(msgs: list[str]) -> None:
    for msg in msgs:
        click.echo(msg)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("settings_file")
@click.argument("features_csv")
@click.argument("people_csv")
@click.argument("selected_csv")
@click.argument("remaining_csv")
@click.option("-n", "--number-wanted", type=int)
def csv(
    settings_file: str,
    features_csv: str,
    people_csv: str,
    selected_csv: str,
    remaining_csv: str,
    number_wanted: int,
) -> None:
    adapter = adapters.CSVAdapter()
    settings, msg = Settings.load_from_file(settings_file_path=Path(settings_file))
    echo_all([msg])
    features, msgs = adapter.load_features_from_file(Path(features_csv))
    echo_all(msgs)
    people, msgs = adapter.load_people_from_file(Path(people_csv), settings, features)
    echo_all(msgs)
    success, people_selected, msgs = core.run_stratification(features, people, number_wanted, settings)
    echo_all(msgs)
    selected_rows, remaining_rows, _ = core.selected_remaining_tables(people, people_selected[0], features, settings)
    with (
        open(selected_csv, "w", newline="") as selected_f,
        open(remaining_csv, "w", newline="") as remaining_f,
    ):
        adapter.selected_file = selected_f
        adapter.remaining_file = remaining_f
        adapter.output_selected_remaining(selected_rows, remaining_rows)


@cli.command()
def gsheet() -> None:
    pass


@cli.command()
@click.argument("settings_file")
@click.argument("features_csv")
@click.argument("people_csv")
@click.option("-n", "--number-wanted", type=int)
def gen_sample(settings_file: str, features_csv: str, people_csv: str, number_wanted: int) -> None:
    adapter = adapters.CSVAdapter()
    settings, msg = Settings.load_from_file(settings_file_path=Path(settings_file))
    echo_all([msg])
    features, msgs = adapter.load_features_from_file(Path(features_csv))
    echo_all(msgs)
    with open(people_csv, "w", newline="") as people_f:
        people_features.create_readable_sample_file(features, people_f, number_wanted, settings)
