import pytest

from sortition_algorithms.features import (
    FEATURE_FILE_FIELD_NAMES,
    FEATURE_FILE_FIELD_NAMES_FLEX,
    FEATURE_FILE_FIELD_NAMES_FLEX_OLD,
    FEATURE_FILE_FIELD_NAMES_OLD,
    FeatureCollection,
    FeatureValueMinMax,
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
    assert list(features.feature_values()) == [("gender", ["male", "female", "non-binary-other"])]
    assert features.minimum_selection() == 8
    assert features.maximum_selection() == 13


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
    assert list(features.feature_values()) == [("gender", ["male", "female", "non-binary-other"])]
    assert features.minimum_selection() == 8
    assert features.maximum_selection() == 13


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
    assert list(features.feature_values()) == [("gender", ["male", "female", "non-binary-other"])]
    assert features.minimum_selection() == 8
    assert features.maximum_selection() == 13


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
    assert list(features.feature_values()) == [("gender", ["male", "female", "non-binary-other"])]
    assert features.minimum_selection() == 8
    assert features.maximum_selection() == 13


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
        assert sorted(features.feature_names) == ["age", "gender"]

        # Check feature values
        feature_values_dict = dict(features.feature_values())
        assert sorted(feature_values_dict["gender"]) == ["female", "male"]
        assert sorted(feature_values_dict["age"]) == ["18-30", "31-50", "51+"]

        # Check min/max calculations
        # gender: min=4, max=8; age: min=4, max=10
        # minimum_selection takes the max of individual feature minimums: max(4, 5) = 5
        # maximum_selection takes the min of individual feature maximums: min(8, 10) = 8
        assert features.minimum_selection() == 5
        assert features.maximum_selection() == 8

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

        assert sorted(features.feature_names) == ["education", "gender"]
        assert features.minimum_selection() == 4  # max(4, 3) = 4
        assert features.maximum_selection() == 7  # min(8, 7) = 7
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

        assert features.feature_names == ["gender"]
        assert dict(features.feature_values())["gender"] == ["male"]

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

        assert features.feature_names == ["gender"]
        assert dict(features.feature_values())["gender"] == ["female"]

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

        feature_values_dict = dict(features.feature_values())
        assert "gender" in feature_values_dict
        assert sorted(feature_values_dict["gender"]) == ["female", "male"]

    def test_numeric_feature_names_and_values(self):
        """Test that numeric feature names and values are handled correctly."""
        head = FEATURE_FILE_FIELD_NAMES
        body = [
            {"feature": 123, "value": 456, "min": "1", "max": "2"},
            {"feature": 123, "value": 789, "min": "2", "max": "3"},
        ]
        features, _ = read_in_features(head, body)

        feature_values_dict = dict(features.feature_values())
        assert "123" in feature_values_dict
        assert sorted(feature_values_dict["123"]) == ["456", "789"]


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

        feature_values_dict = dict(features.feature_values())
        assert "gender" in feature_values_dict
        assert sorted(feature_values_dict["gender"]) == ["female", "male"]

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

        feature_values_dict = dict(features.feature_values())
        assert "gender" in feature_values_dict
        assert sorted(feature_values_dict["gender"]) == ["female", "male"]


class TestFeatureCollectionMethods:
    """Test the FeatureCollection class methods directly."""

    def test_empty_feature_collection(self):
        """Test behaviour of empty FeatureCollection."""
        features = FeatureCollection()

        assert features.feature_names == []
        assert list(features.feature_values()) == []
        assert list(features.feature_values_counts()) == []
        assert features.minimum_selection() == 0  # max of empty sequence
        assert features.maximum_selection() == 0  # min of empty sequence

    def test_feature_collection_add_and_access(self):
        """Test adding features and accessing them."""
        features = FeatureCollection()

        # Add some feature values
        counts1 = FeatureValueMinMax(min=1, max=3)
        counts2 = FeatureValueMinMax(min=2, max=4)

        features.add_feature("gender", "male", counts1)
        features.add_feature("gender", "female", counts2)

        # Test access methods
        assert features.feature_names == ["gender"]
        assert list(features.feature_values()) == [("gender", ["male", "female"])]

        # Test feature_values_counts
        values_counts = list(features.feature_values_counts())
        assert len(values_counts) == 2
        assert ("gender", "male", counts1) in values_counts
        assert ("gender", "female", counts2) in values_counts

    def test_feature_collection_min_max_calculations(self):
        """Test minimum and maximum selection calculations."""
        features = FeatureCollection()

        # Add first feature: min=3, max=7
        features.add_feature("gender", "male", FeatureValueMinMax(min=1, max=3))
        features.add_feature("gender", "female", FeatureValueMinMax(min=2, max=4))

        # Add second feature: min=4, max=6
        features.add_feature("age", "young", FeatureValueMinMax(min=2, max=3))
        features.add_feature("age", "old", FeatureValueMinMax(min=2, max=3))

        # minimum_selection should be max(3, 4) = 4
        # maximum_selection should be min(7, 6) = 6
        assert features.minimum_selection() == 4
        assert features.maximum_selection() == 6

    def test_feature_collection_check_min_max_valid(self):
        """Test that check_min_max passes for valid configurations."""
        features = FeatureCollection()
        features.add_feature("gender", "male", FeatureValueMinMax(min=1, max=3))
        features.add_feature("gender", "female", FeatureValueMinMax(min=1, max=3))

        # Should not raise an exception
        features.check_min_max()

    def test_feature_collection_check_min_max_invalid(self):
        """Test that check_min_max raises error for invalid configurations."""
        features = FeatureCollection()

        # First feature: min=5, max=6
        features.add_feature("gender", "male", FeatureValueMinMax(min=3, max=3))
        features.add_feature("gender", "female", FeatureValueMinMax(min=2, max=3))

        # Second feature: min=2, max=3
        features.add_feature("age", "young", FeatureValueMinMax(min=1, max=1))
        features.add_feature("age", "old", FeatureValueMinMax(min=1, max=2))

        # minimum_selection = max(5, 2) = 5
        # maximum_selection = min(6, 3) = 3
        # Since 5 > 3, this should raise an error
        with pytest.raises(ValueError, match="Inconsistent numbers in min and max"):
            features.check_min_max()

    def test_feature_collection_check_desired_valid(self):
        """Test that check_desired passes for valid desired numbers."""
        features = FeatureCollection()
        features.add_feature("gender", "male", FeatureValueMinMax(min=1, max=3))
        features.add_feature("gender", "female", FeatureValueMinMax(min=1, max=3))

        # Should not raise an exception for a desired number within range
        features.check_desired(4)  # Within [2, 6]

    def test_feature_collection_check_desired_too_low(self):
        """Test that check_desired raises error for desired number too low."""
        features = FeatureCollection()
        features.add_feature("gender", "male", FeatureValueMinMax(min=2, max=3))
        features.add_feature("gender", "female", FeatureValueMinMax(min=2, max=3))

        # Minimum is 4, so 3 should be too low
        with pytest.raises(Exception, match="out of the range"):
            features.check_desired(3)

    def test_feature_collection_check_desired_too_high(self):
        """Test that check_desired raises error for desired number too high."""
        features = FeatureCollection()
        features.add_feature("gender", "male", FeatureValueMinMax(min=1, max=2))
        features.add_feature("gender", "female", FeatureValueMinMax(min=1, max=2))

        # Maximum is 4, so 5 should be too high
        with pytest.raises(Exception, match="out of the range"):
            features.check_desired(5)

    def test_feature_collection_set_default_max_flex(self):
        """Test setting default max_flex values."""
        features = FeatureCollection()

        # Add features with unset max_flex
        counts1 = FeatureValueMinMax(min=1, max=3)  # max_flex will be MAX_FLEX_UNSET
        counts2 = FeatureValueMinMax(min=2, max=4)  # max_flex will be MAX_FLEX_UNSET

        features.add_feature("gender", "male", counts1)
        features.add_feature("gender", "female", counts2)

        # Set default max_flex
        features.set_default_max_flex()

        # Both should now have max_flex set to the max of all maximums (7)
        assert counts1.max_flex == 7
        assert counts2.max_flex == 7

    def test_feature_collection_get_counts(self):
        """Test getting counts for a particular feature and value."""
        features = FeatureCollection()
        counts1 = FeatureValueMinMax(min=1, max=3)
        counts2 = FeatureValueMinMax(min=2, max=4)
        features.add_feature("gender", "male", counts1)
        features.add_feature("gender", "female", counts2)

        counts = features.get_counts("gender", "male")

        assert counts == counts1

    def test_feature_collection_get_counts_no_match(self):
        """Test setting default max_flex values."""
        features = FeatureCollection()
        counts1 = FeatureValueMinMax(min=1, max=3)
        counts2 = FeatureValueMinMax(min=2, max=4)
        features.add_feature("gender", "male", counts1)
        features.add_feature("gender", "female", counts2)

        with pytest.raises(KeyError):
            features.get_counts("age", "male")

        with pytest.raises(KeyError):
            features.get_counts("gender", "non-binary-other")

    def test_feature_value_pairs_iterates_through_all(self):
        features = FeatureCollection()
        counts = FeatureValueMinMax(min=1, max=3)
        features.add_feature("gender", "male", counts)
        features.add_feature("gender", "female", counts)
        features.add_feature("age", "young", counts)
        features.add_feature("age", "middle-aged", counts)
        features.add_feature("age", "old", counts)

        feature_values = sorted(features.feature_value_pairs())
        assert feature_values == [
            ("age", "middle-aged"),
            ("age", "old"),
            ("age", "young"),
            ("gender", "female"),
            ("gender", "male"),
        ]


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
