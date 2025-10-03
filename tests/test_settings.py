import tomllib

import pytest
from cattrs import ClassValidationError

from sortition_algorithms import settings
from tests.helpers import create_test_settings


class TestSettingsConstructor:
    """Test the Settings class constructor and validation."""

    @pytest.mark.parametrize("alg", settings.SELECTION_ALGORITHMS)
    def test_selection_algorithms_accepted(self, alg):
        settings_obj = create_test_settings(selection_algorithm=alg)
        assert settings_obj.selection_algorithm == alg

    def test_selection_algorithms_blocked_for_unknown(self):
        with pytest.raises(ValueError, match="selection_algorithm unknown is not one of"):
            create_test_settings(selection_algorithm="unknown")

    def test_valid_settings_creation(self):
        """Test creating a valid Settings object."""
        settings_obj = create_test_settings(
            columns_to_keep=["name", "email"],
            check_same_address=True,
            check_same_address_columns=["address1", "postcode"],
            selection_algorithm="maximin",
        )
        assert settings_obj.id_column == "id"
        assert settings_obj.columns_to_keep == ["name", "email"]
        assert settings_obj.check_same_address is True
        assert settings_obj.check_same_address_columns == ["address1", "postcode"]
        assert settings_obj.selection_algorithm == "maximin"
        assert settings_obj.max_attempts == 100
        assert settings_obj.random_number_seed == 42

    def test_id_column_must_be_string(self):
        """Test that id_column must be a string."""
        with pytest.raises(TypeError):
            settings.Settings(
                id_column=123,  # t
                columns_to_keep=[],
            )

    def test_columns_to_keep_must_be_list(self):
        """Test that columns_to_keep must be a list."""
        with pytest.raises(TypeError, match="columns_to_keep must be a LIST of strings"):
            create_test_settings(columns_to_keep="not_a_list")  # t

    def test_columns_to_keep_must_be_list_of_strings(self):
        """Test that columns_to_keep must be a list of strings."""
        with pytest.raises(TypeError, match="columns_to_keep must be a list of STRINGS"):
            create_test_settings(columns_to_keep=["valid", 123, "also_valid"])  # t

    def test_check_same_address_must_be_bool(self):
        """Test that check_same_address must be a boolean."""
        with pytest.raises(TypeError):
            settings.Settings(
                id_column="test",
                columns_to_keep=[],
                check_same_address="not_a_bool",  # t
            )

    def test_check_same_address_columns_must_be_list(self):
        """Test that check_same_address_columns must be a list."""
        with pytest.raises(TypeError, match="check_same_address_columns must be a LIST of strings"):
            create_test_settings(check_same_address_columns="not_a_list")  # t

    def test_check_same_address_columns_must_be_strings(self):
        """Test that check_same_address_columns must contain strings."""
        with pytest.raises(TypeError, match="check_same_address_columns must be a list of STRINGS"):
            create_test_settings(check_same_address_columns=["valid", 123])  # t

    def test_check_same_address_true_requires_columns(self):
        """Test that check_same_address=True requires columns to be specified."""
        with pytest.raises(
            ValueError,
            match="check_same_address is TRUE but there are no columns listed",
        ):
            create_test_settings(check_same_address=True, check_same_address_columns=[])

    def test_check_same_address_false_allows_empty_columns(self):
        """Test that check_same_address=False allows empty columns list."""
        settings_obj = create_test_settings(check_same_address=False, check_same_address_columns=[])
        assert settings_obj.check_same_address is False
        assert settings_obj.check_same_address_columns == []

    def test_max_attempts_must_be_int(self):
        """Test that max_attempts must be an integer."""
        with pytest.raises(TypeError):
            settings.Settings(
                id_column="test",
                columns_to_keep=[],
                max_attempts="not_an_int",  # t
            )

    def test_random_number_seed_must_be_int(self):
        """Test that random_number_seed must be an integer."""
        with pytest.raises(TypeError):
            settings.Settings(
                id_column="test",
                columns_to_keep=[],
                random_number_seed="not_an_int",  # t
            )

    def test_full_columns_to_keep_includes_address_columns(self):
        """Test that address columns are included"""
        settings_obj = create_test_settings(columns_to_keep=["a", "b", "c"], check_same_address_columns=["d", "e"])
        assert settings_obj.full_columns_to_keep == ["a", "b", "c", "d", "e"]

    def test_full_columns_to_keep_excludes_address_column_duplicates(self):
        """Test that address columns are included, but not added if they are duplicates"""
        settings_obj = create_test_settings(columns_to_keep=["a", "b", "c"], check_same_address_columns=["d", "c"])
        # note, no duplicate "c"
        assert settings_obj.full_columns_to_keep == ["a", "b", "c", "d"]


class TestSettingsLoadFromFile:
    """Test the Settings.load_from_file() class method."""

    def test_load_from_existing_file(self, tmp_path):
        """Test loading settings from an existing TOML file."""
        toml_content = """
id_column = "test_id"
check_same_address = true
check_same_address_columns = ["addr1", "postcode"]
max_attempts = 50
columns_to_keep = ["name", "email", "phone"]
selection_algorithm = "nash"
random_number_seed = 42
"""
        settings_file_path = tmp_path / "settings.toml"
        settings_file_path.write_text(toml_content)

        settings_obj, report = settings.Settings.load_from_file(settings_file_path)

        assert settings_obj.id_column == "test_id"
        assert settings_obj.check_same_address is True
        assert settings_obj.check_same_address_columns == ["addr1", "postcode"]
        assert settings_obj.max_attempts == 50
        assert settings_obj.columns_to_keep == ["name", "email", "phone"]
        assert settings_obj.selection_algorithm == "nash"
        assert settings_obj.random_number_seed == 42
        assert report.as_text() == ""

    def test_load_from_nonexistent_file_creates_default(self, tmp_path):
        """Test that loading from a non-existent file creates a default settings file."""
        settings_file_path = tmp_path / "new_settings.toml"

        assert not settings_file_path.exists()

        settings_obj, report = settings.Settings.load_from_file(settings_file_path)

        # Check that the file was created
        assert settings_file_path.exists()

        # Check that the message indicates the file was created
        report_text = report.as_text()
        assert "Wrote default settings to" in report_text
        assert "restart this app" in report_text

        # Check that the settings have default values
        assert settings_obj.id_column == "nationbuilder_id"
        assert settings_obj.selection_algorithm == "maximin"
        assert settings_obj.max_attempts == 100
        assert settings_obj.random_number_seed == 0

    def test_load_with_check_same_address_false(self, tmp_path):
        """Test loading settings with check_same_address set to false."""
        toml_content = """
id_column = "test_id"
check_same_address = false
check_same_address_columns = ["some", "columns"]  # these should be ignored
max_attempts = 10
columns_to_keep = ["name"]
selection_algorithm = "legacy"
random_number_seed = 0
"""
        settings_file_path = tmp_path / "settings.toml"
        settings_file_path.write_text(toml_content)

        settings_obj, report = settings.Settings.load_from_file(settings_file_path)

        assert settings_obj.check_same_address is False
        assert settings_obj.check_same_address_columns == []  # Should be reset to empty list
        report_text = report.as_text()
        assert "WARNING" in report_text
        assert "do NOT check if respondents have same address" in report_text

    def test_load_with_invalid_toml_content(self, tmp_path):
        """Test loading settings with invalid TOML content raises appropriate error."""
        invalid_toml = """
id_column = "test"
selection_algorithm = "invalid_algorithm"
check_same_address = true
check_same_address_columns = []
max_attempts = 10
columns_to_keep = ["name"]
random_number_seed = 0
"""
        settings_file_path = tmp_path / "settings.toml"
        settings_file_path.write_text(invalid_toml)

        with pytest.raises(
            ClassValidationError,
            match="While structuring Settings",
        ) as excinfo:
            settings.Settings.load_from_file(settings_file_path)
        assert excinfo.group_contains(
            ValueError,
            match="check_same_address is TRUE but",
        )

    def test_load_with_malformed_toml_file(self, tmp_path):
        """Test loading settings from a malformed TOML file raises appropriate error."""
        malformed_toml = """
id_column = "test"
this is not valid TOML syntax [[[
"""
        settings_file_path = tmp_path / "settings.toml"
        settings_file_path.write_text(malformed_toml)

        with pytest.raises(tomllib.TOMLDecodeError):
            settings.Settings.load_from_file(settings_file_path)
