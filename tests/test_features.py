import pytest

from sortition_algorithms.features import (
    FEATURE_FILE_FIELD_NAMES,
    FEATURE_FILE_FIELD_NAMES_FLEX,
    FEATURE_FILE_FIELD_NAMES_FLEX_OLD,
    FEATURE_FILE_FIELD_NAMES_OLD,
    FeatureValueMinMax,
    maximum_selection,
    minimum_selection,
    read_in_features,
)


def test_read_in_features_without_flex():
    """
    Test a basic import with a single feature/category
    """
    head = FEATURE_FILE_FIELD_NAMES
    body = [
        {"feature": "gender", "value": "male", "min": "4", "max": "6"},
        {"feature": "gender", "value": "female", "min": "4", "max": "6"},
        {"feature": "gender", "value": "non-binary-other", "min": "0", "max": "1"},
    ]
    features, _ = read_in_features(head, body)
    assert list(features.keys()) == ["gender"]
    assert sorted(features["gender"].keys()) == ["female", "male", "non-binary-other"]
    assert minimum_selection(features) == 8
    assert maximum_selection(features) == 13


def test_read_in_features_with_flex():
    """
    Test a basic import with a single feature/category
    """
    head = FEATURE_FILE_FIELD_NAMES_FLEX
    body = [
        {
            "feature": "gender",
            "value": "male",
            "min": "4",
            "max": "6",
            "min_flex": "4",
            "max_flex": "6",
        },
        {
            "feature": "gender",
            "value": "female",
            "min": "4",
            "max": "6",
            "min_flex": "4",
            "max_flex": "6",
        },
        {
            "feature": "gender",
            "value": "non-binary-other",
            "min": "0",
            "max": "1",
            "min_flex": "0",
            "max_flex": "1",
        },
    ]
    features, _ = read_in_features(head, body)
    assert list(features.keys()) == ["gender"]
    assert sorted(features["gender"].keys()) == ["female", "male", "non-binary-other"]
    assert minimum_selection(features) == 8
    assert maximum_selection(features) == 13


def test_read_in_features_without_flex_old_names():
    """
    Test a basic import with a single feature/category
    """
    head = FEATURE_FILE_FIELD_NAMES_OLD
    body = [
        {"category": "gender", "name": "male", "min": "4", "max": "6"},
        {"category": "gender", "name": "female", "min": "4", "max": "6"},
        {"category": "gender", "name": "non-binary-other", "min": "0", "max": "1"},
    ]
    features, _ = read_in_features(head, body)
    assert list(features.keys()) == ["gender"]
    assert sorted(features["gender"].keys()) == ["female", "male", "non-binary-other"]
    assert minimum_selection(features) == 8
    assert maximum_selection(features) == 13


def test_read_in_features_with_flex_old_names():
    """
    Test a basic import with a single feature/category
    """
    head = FEATURE_FILE_FIELD_NAMES_FLEX_OLD
    body = [
        {
            "category": "gender",
            "name": "male",
            "min": "4",
            "max": "6",
            "min_flex": "4",
            "max_flex": "6",
        },
        {
            "category": "gender",
            "name": "female",
            "min": "4",
            "max": "6",
            "min_flex": "4",
            "max_flex": "6",
        },
        {
            "category": "gender",
            "name": "non-binary-other",
            "min": "0",
            "max": "1",
            "min_flex": "0",
            "max_flex": "1",
        },
    ]
    features, _ = read_in_features(head, body)
    assert list(features.keys()) == ["gender"]
    assert sorted(features["gender"].keys()) == ["female", "male", "non-binary-other"]
    assert minimum_selection(features) == 8
    assert maximum_selection(features) == 13


class TestReadInFeaturesMultipleFeatures:
    """Test reading in features with multiple feature types."""

    def test_multiple_features_without_flex(self):
        """Test reading multiple features without flex columns."""
        head = FEATURE_FILE_FIELD_NAMES
        body = [
            {"feature": "gender", "value": "male", "min": "2", "max": "4"},
            {"feature": "gender", "value": "female", "min": "2", "max": "4"},
            {"feature": "age", "value": "18-30", "min": "2", "max": "3"},
            {"feature": "age", "value": "31-50", "min": "2", "max": "5"},
            {"feature": "age", "value": "51+", "min": "1", "max": "2"},
        ]
        features, _ = read_in_features(head, body)

        # Check we have both features
        assert sorted(features.keys()) == ["age", "gender"]

        # Check feature values
        assert sorted(features["gender"].keys()) == ["female", "male"]
        assert sorted(features["age"].keys()) == ["18-30", "31-50", "51+"]

        # Check min/max calculations
        # gender: min=4, max=8; age: min=5, max=10
        # minimum_selection takes the max of individual feature minimums: max(4, 5) = 5
        # maximum_selection takes the min of individual feature maximums: min(8, 10) = 8
        assert minimum_selection(features) == 5
        assert maximum_selection(features) == 8

    def test_multiple_features_with_flex(self):
        """Test reading multiple features with flex columns."""
        head = FEATURE_FILE_FIELD_NAMES_FLEX
        body = [
            {
                "feature": "gender",
                "value": "male",
                "min": "2",
                "max": "4",
                "min_flex": "1",
                "max_flex": "5",
            },
            {
                "feature": "gender",
                "value": "female",
                "min": "2",
                "max": "4",
                "min_flex": "1",
                "max_flex": "5",
            },
            {
                "feature": "education",
                "value": "university",
                "min": "1",
                "max": "3",
                "min_flex": "0",
                "max_flex": "4",
            },
            {
                "feature": "education",
                "value": "secondary",
                "min": "2",
                "max": "4",
                "min_flex": "1",
                "max_flex": "5",
            },
        ]
        features, messages = read_in_features(head, body)

        assert sorted(features.keys()) == ["education", "gender"]
        assert minimum_selection(features) == 4  # max(4, 3) = 4
        assert maximum_selection(features) == 7  # min(8, 7) = 7
        assert "Number of features: 2" in messages


class TestReadInFeaturesErrorHandling:
    """Test error handling in read_in_features function."""

    def test_invalid_headers_missing_required_field(self):
        """Test that missing required headers raise appropriate error."""
        head = ["feature", "value", "min"]  # missing "max"
        body = [{"feature": "gender", "value": "male", "min": "1"}]

        with pytest.raises(ValueError, match="Did not find required column name 'max'"):
            read_in_features(head, body)

    def test_invalid_headers_duplicate_field(self):
        """Test that duplicate headers raise appropriate error."""
        head = ["feature", "value", "min", "max", "min"]  # duplicate "min"
        body = [{"feature": "gender", "value": "male", "min": "1", "max": "2"}]

        with pytest.raises(ValueError, match="Found MORE THAN 1 column named 'min'"):
            read_in_features(head, body)

    def test_extra_headers_ignored(self):
        """Test that missing required headers raise appropriate error."""
        head = [
            "feature",
            "value",
            "min",
            "max",
            "suggest min",
            "suggest max",
        ]  # extra "suggest min/max" headers
        body = [{"feature": "gender", "value": "male", "min": "1", "max": "2"}]
        features, _ = read_in_features(head, body)

        assert list(features.keys()) == ["gender"]
        assert list(features["gender"].keys()) == ["male"]

    def test_blank_feature_name_skipped(self):
        """Test that rows with blank feature names are skipped."""
        head = FEATURE_FILE_FIELD_NAMES
        body = [
            {
                "feature": "",
                "value": "male",
                "min": "1",
                "max": "2",
            },  # blank feature, should be skipped
            {"feature": "gender", "value": "female", "min": "2", "max": "3"},
        ]
        features, _ = read_in_features(head, body)

        assert list(features.keys()) == ["gender"]
        assert list(features["gender"].keys()) == ["female"]

    def test_blank_value_raises_error(self):
        """Test that blank feature values raise an error."""
        head = FEATURE_FILE_FIELD_NAMES
        body = [{"feature": "gender", "value": "", "min": "1", "max": "2"}]

        with pytest.raises(ValueError, match="found a blank cell"):
            read_in_features(head, body)

    def test_blank_min_raises_error(self):
        """Test that blank min values raise an error."""
        head = FEATURE_FILE_FIELD_NAMES
        body = [{"feature": "gender", "value": "male", "min": "", "max": "2"}]

        with pytest.raises(ValueError, match="found a blank cell"):
            read_in_features(head, body)

    def test_blank_max_raises_error(self):
        """Test that blank max values raise an error."""
        head = FEATURE_FILE_FIELD_NAMES
        body = [{"feature": "gender", "value": "male", "min": "1", "max": ""}]

        with pytest.raises(ValueError, match="found a blank cell"):
            read_in_features(head, body)

    def test_blank_flex_values_raise_error(self):
        """Test that blank flex values raise an error when flex headers are present."""
        head = FEATURE_FILE_FIELD_NAMES_FLEX
        body = [
            {
                "feature": "gender",
                "value": "male",
                "min": "1",
                "max": "2",
                "min_flex": "",
                "max_flex": "3",
            }
        ]
        with pytest.raises(ValueError, match="found a blank min_flex or max_flex cell"):
            read_in_features(head, body)

    def test_inconsistent_flex_values_raise_error(self):
        """Test that flex values outside min/max range raise an error."""
        head = FEATURE_FILE_FIELD_NAMES_FLEX
        body = [
            {
                "feature": "gender",
                "value": "male",
                "min": "2",
                "max": "4",
                "min_flex": "3",  # min_flex > min, which is invalid
                "max_flex": "5",
            }
        ]
        with pytest.raises(
            ValueError,
            match="flex values must be equal or outside the max and min values",
        ):
            read_in_features(head, body)

    def test_inconsistent_min_max_across_features(self):
        """Test error when minimum selection exceeds maximum selection across features."""
        head = FEATURE_FILE_FIELD_NAMES
        body = [
            {
                "feature": "gender",
                "value": "male",
                "min": "5",
                "max": "6",
            },  # min total: 10
            {
                "feature": "gender",
                "value": "female",
                "min": "5",
                "max": "6",
            },  # max total: 12
            {
                "feature": "age",
                "value": "young",
                "min": "1",
                "max": "2",
            },  # min total: 2
            {"feature": "age", "value": "old", "min": "1", "max": "2"},  # max total: 4
        ]
        # gender min total = 10, age max total = 4, so min > max which is inconsistent
        with pytest.raises(ValueError, match="Inconsistent numbers in min and max"):
            read_in_features(head, body)


class TestReadInFeaturesDataTypes:
    """Test handling of different data types in input."""

    def test_string_values_stripped(self):
        """Test that string values are properly stripped of whitespace."""
        head = FEATURE_FILE_FIELD_NAMES
        body = [
            {"feature": "  gender  ", "value": "  male  ", "min": "1", "max": "2"},
            {"feature": "gender", "value": "female", "min": "2", "max": "3"},
        ]
        features, _ = read_in_features(head, body)

        assert "gender" in features
        assert sorted(features["gender"].keys()) == ["female", "male"]

    def test_numeric_feature_names_and_values(self):
        """Test that numeric feature names and values are handled correctly."""
        head = FEATURE_FILE_FIELD_NAMES
        body = [
            {"feature": 123, "value": 456, "min": "1", "max": "2"},
            {"feature": 123, "value": 789, "min": "2", "max": "3"},
        ]
        features, _ = read_in_features(head, body)

        assert "123" in features
        assert sorted(features["123"].keys()) == ["456", "789"]


class TestReadInFeaturesOldColumnNames:
    """Test backward compatibility with old column names."""

    def test_old_column_names_without_flex(self):
        """Test that old column names (category/name) work correctly."""
        head = ["category", "name", "min", "max"]
        body = [
            {"category": "gender", "name": "male", "min": "1", "max": "2"},
            {"category": "gender", "name": "female", "min": "2", "max": "3"},
        ]
        features, _ = read_in_features(head, body)

        assert "gender" in features
        assert sorted(features["gender"].keys()) == ["female", "male"]

    def test_old_column_names_with_flex(self):
        """Test that old column names with flex columns work correctly."""
        head = FEATURE_FILE_FIELD_NAMES_FLEX_OLD
        body = [
            {
                "category": "gender",
                "name": "male",
                "min": "1",
                "max": "2",
                "min_flex": "0",
                "max_flex": "3",
            },
            {
                "category": "gender",
                "name": "female",
                "min": "2",
                "max": "3",
                "min_flex": "1",
                "max_flex": "4",
            },
        ]
        features, _ = read_in_features(head, body)

        assert "gender" in features
        assert sorted(features["gender"].keys()) == ["female", "male"]


class TestFeatureValueCounts:
    """Test the FeatureValueCounts class directly."""

    def test_feature_value_counts_creation(self):
        """Test creating FeatureValueCounts objects."""
        counts = FeatureValueMinMax(min=1, max=5)

        assert counts.min == 1
        assert counts.max == 5
        assert counts.min_flex == 0
        assert counts.max_flex == -1  # MAX_FLEX_UNSET

    def test_feature_value_counts_with_flex(self):
        """Test creating FeatureValueCounts with flex values."""
        counts = FeatureValueMinMax(min=2, max=4, min_flex=1, max_flex=6)

        assert counts.min == 2
        assert counts.max == 4
        assert counts.min_flex == 1
        assert counts.max_flex == 6

    def test_feature_value_counts_set_default_max_flex(self):
        """Test setting default max_flex when unset."""
        counts = FeatureValueMinMax(min=1, max=3)  # max_flex defaults to MAX_FLEX_UNSET

        counts.set_default_max_flex(10)
        assert counts.max_flex == 10

    def test_feature_value_counts_set_default_max_flex_already_set(self):
        """Test that set_default_max_flex doesn't override existing values."""
        counts = FeatureValueMinMax(min=1, max=3, max_flex=5)

        counts.set_default_max_flex(10)
        assert counts.max_flex == 5  # Should remain unchanged
