from pathlib import Path

import pytest

from sortition_algorithms import settings


def create_settings(
    columns_to_keep: list[str] | None = None,
    check_same_address: bool = False,
    check_same_address_columns: list[str] | None = None,
    selection_algorithm: str = "legacy",
) -> settings.Settings:
    columns_to_keep = columns_to_keep or []
    check_same_address_columns = check_same_address_columns or []
    return settings.Settings(
        id_column="nationbuilder_id",
        columns_to_keep=columns_to_keep,
        check_same_address=check_same_address,
        check_same_address_columns=check_same_address_columns,
        max_attempts=3,
        selection_algorithm=selection_algorithm,
        random_number_seed=1234,
        json_file_path=Path("/tmp"),  # noqa: S108
    )


@pytest.mark.parametrize("alg", settings.SELECTION_ALGORITHMS)
def test_selection_algorithms_accepted(alg):
    settings = create_settings(selection_algorithm=alg)
    assert settings.selection_algorithm == alg


def test_selection_algorithms_blocked_for_unknown():
    with pytest.raises(ValueError):
        create_settings(selection_algorithm="unknown")


# TODO: consider more tests
