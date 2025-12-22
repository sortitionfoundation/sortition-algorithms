import pickle

from sortition_algorithms.errors import (
    BadDataError,
    InfeasibleQuotasCantRelaxError,
    InfeasibleQuotasError,
    ParseTableErrorMsg,
    ParseTableMultiError,
    ParseTableMultiValueErrorMsg,
    RetryableSelectionError,
    SelectionError,
    SelectionMultilineError,
    SortitionBaseError,
)


def assert_exception_equal(original: Exception, unpickled: Exception) -> None:
    """Helper to assert that two exceptions are equal in all their attributes."""
    assert type(original) is type(unpickled)
    assert str(original) == str(unpickled)
    assert original.args == unpickled.args


class TestSortitionBaseErrorPickle:
    def test_pickle_with_all_params(self):
        error = SortitionBaseError(
            message="Test error message",
            error_code="TEST_001",
            error_params={"key1": "value1", "key2": 42},
        )
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert_exception_equal(error, unpickled)
        assert unpickled.message == "Test error message"
        assert unpickled.error_code == "TEST_001"
        assert unpickled.error_params == {"key1": "value1", "key2": 42}
        assert unpickled.is_retryable is False

    def test_pickle_with_defaults(self):
        error = SortitionBaseError()
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert_exception_equal(error, unpickled)
        assert unpickled.message == ""
        assert unpickled.error_code == ""
        assert unpickled.error_params == {}
        assert unpickled.is_retryable is False

    def test_pickle_with_message_only(self):
        error = SortitionBaseError(message="Simple message")
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert_exception_equal(error, unpickled)
        assert unpickled.message == "Simple message"
        assert unpickled.error_code == ""
        assert unpickled.error_params == {}


class TestBadDataErrorPickle:
    def test_pickle_bad_data_error(self):
        error = BadDataError(
            message="Invalid data format",
            error_code="BAD_DATA_001",
            error_params={"field": "name"},
        )
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert_exception_equal(error, unpickled)
        assert unpickled.message == "Invalid data format"
        assert unpickled.error_code == "BAD_DATA_001"
        assert unpickled.error_params == {"field": "name"}


class TestSelectionErrorPickle:
    def test_pickle_selection_error(self):
        error = SelectionError(
            message="Selection failed",
            error_code="SEL_001",
            error_params={"attempt": 1},
        )
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert_exception_equal(error, unpickled)
        assert unpickled.message == "Selection failed"
        assert unpickled.error_code == "SEL_001"
        assert unpickled.error_params == {"attempt": 1}


class TestRetryableSelectionErrorPickle:
    def test_pickle_retryable_error(self):
        error = RetryableSelectionError(
            message="Temporary failure",
            error_code="RETRY_001",
            error_params={"retry_count": 3},
        )
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert_exception_equal(error, unpickled)
        assert unpickled.message == "Temporary failure"
        assert unpickled.error_code == "RETRY_001"
        assert unpickled.error_params == {"retry_count": 3}
        assert unpickled.is_retryable is True


class TestSelectionMultilineErrorPickle:
    def test_pickle_with_all_params(self):
        lines = ["Error line 1", "Error line 2", "Error line 3"]
        error = SelectionMultilineError(
            lines=lines,
            is_retryable=True,
            error_code="MULTI_001",
            error_params={"context": "test"},
        )
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert_exception_equal(error, unpickled)
        assert unpickled.all_lines == lines
        assert unpickled.is_retryable is True
        assert unpickled.error_code == "MULTI_001"
        assert unpickled.error_params == {"context": "test"}
        assert unpickled.lines() == lines

    def test_pickle_with_defaults(self):
        lines = ["Single error line"]
        error = SelectionMultilineError(lines=lines)
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert_exception_equal(error, unpickled)
        assert unpickled.all_lines == lines
        assert unpickled.is_retryable is False
        assert unpickled.error_code == ""
        assert unpickled.error_params == {}

    def test_pickle_html_output(self):
        lines = ["<Error> line 1", "Error & line 2"]
        error = SelectionMultilineError(lines=lines)
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert unpickled.to_html() == error.to_html()


class TestParseTableErrorMsgPickle:
    def test_pickle_parse_table_error_msg(self):
        error_msg = ParseTableErrorMsg(
            row=5,
            row_name="person_123",
            key="age",
            value="invalid",
            msg="Invalid age value",
            error_code="PARSE_001",
            error_params={"expected": "integer"},
        )
        pickled = pickle.dumps(error_msg)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert unpickled.row == 5
        assert unpickled.row_name == "person_123"
        assert unpickled.key == "age"
        assert unpickled.value == "invalid"
        assert unpickled.msg == "Invalid age value"
        assert unpickled.error_code == "PARSE_001"
        assert unpickled.error_params == {"expected": "integer"}
        assert str(unpickled) == str(error_msg)


class TestParseTableMultiValueErrorMsgPickle:
    def test_pickle_parse_table_multi_value_error_msg(self):
        error_msg = ParseTableMultiValueErrorMsg(
            row=10,
            row_name="feature_gender",
            keys=["min", "max"],
            values=["10", "5"],
            msg="Min cannot exceed max",
            error_code="PARSE_002",
            error_params={"min_value": 10, "max_value": 5},
        )
        pickled = pickle.dumps(error_msg)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert unpickled.row == 10
        assert unpickled.row_name == "feature_gender"
        assert unpickled.keys == ["min", "max"]
        assert unpickled.values == ["10", "5"]
        assert unpickled.msg == "Min cannot exceed max"
        assert unpickled.error_code == "PARSE_002"
        assert unpickled.error_params == {"min_value": 10, "max_value": 5}
        assert str(unpickled) == str(error_msg)


class TestParseTableMultiErrorPickle:
    def test_pickle_empty_error_list(self):
        error = ParseTableMultiError()
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert len(unpickled) == 0
        assert unpickled.all_errors == []
        assert unpickled.lines() == []

    def test_pickle_with_single_error(self):
        error_msg = ParseTableErrorMsg(
            row=3,
            row_name="person_456",
            key="gender",
            value="X",
            msg="Invalid gender",
            error_code="VAL_001",
        )
        error = ParseTableMultiError(errors=[error_msg])
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert len(unpickled) == 1
        assert len(unpickled.all_errors) == 1
        assert unpickled.all_errors[0].row == 3
        assert unpickled.all_errors[0].row_name == "person_456"
        assert unpickled.all_errors[0].msg == "Invalid gender"

    def test_pickle_with_multiple_errors(self):
        errors = [
            ParseTableErrorMsg(
                row=1,
                row_name="p1",
                key="age",
                value="abc",
                msg="Invalid age",
            ),
            ParseTableMultiValueErrorMsg(
                row=2,
                row_name="feature_x",
                keys=["min", "max"],
                values=["5", "3"],
                msg="Invalid range",
            ),
        ]
        error = ParseTableMultiError(errors=errors)
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert len(unpickled) == 2
        assert unpickled.all_errors[0].row == 1
        assert unpickled.all_errors[1].row == 2
        assert unpickled.lines() == error.lines()


class TestInfeasibleQuotasErrorPickle:
    def test_pickle_infeasible_quotas_error(self):
        # Create a minimal FeatureCollection
        from sortition_algorithms.features import read_in_features

        features_data = [
            {"feature": "gender", "value": "male", "min": "5", "max": "10"},
            {"feature": "gender", "value": "female", "min": "5", "max": "10"},
        ]
        head = ["feature", "value", "min", "max"]
        features, _, _ = read_in_features(head, features_data)

        output_lines = [
            "Gender quota cannot be met",
            "Suggested relaxation: male 3-12",
        ]
        error = InfeasibleQuotasError(features=features, output=output_lines)

        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert_exception_equal(error, unpickled)
        assert unpickled.features is not None
        assert len(unpickled.all_lines) == 3  # "The quotas are infeasible:" + 2 output lines
        assert unpickled.all_lines[0] == "The quotas are infeasible:"
        assert unpickled.all_lines[1] == "Gender quota cannot be met"
        assert unpickled.all_lines[2] == "Suggested relaxation: male 3-12"


class TestInfeasibleQuotasCantRelaxErrorPickle:
    def test_pickle_infeasible_quotas_cant_relax_error(self):
        error = InfeasibleQuotasCantRelaxError(
            message="No feasible relaxation found",
            error_code="INFEAS_001",
            error_params={"feature": "gender"},
        )
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)  # noqa: S301

        assert_exception_equal(error, unpickled)
        assert unpickled.message == "No feasible relaxation found"
        assert unpickled.error_code == "INFEAS_001"
        assert unpickled.error_params == {"feature": "gender"}
