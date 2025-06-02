from sortition_algorithms.features import FEATURE_FILE_FIELD_NAMES, read_in_features


def test_read_in_features_without_flex():
    head = FEATURE_FILE_FIELD_NAMES
    body = [
        {"category": "gender", "name": "male", "min": "4", "max": "6"},
        {"category": "gender", "name": "female", "min": "4", "max": "6"},
        {"category": "gender", "name": "non-binary-other", "min": "0", "max": "1"},
    ]
    features, _, min_select, max_select = read_in_features(head, body)
    assert list(features.feature_values()) == [("gender", ["male", "female", "non-binary-other"])]
    assert features.minimum_selection() == 8
    assert features.maximum_selection() == 13
