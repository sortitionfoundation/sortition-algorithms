import enum
import html
import logging
import random
import secrets
import string
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any

from attrs import define, field
from cattrs import Converter
from requests.structures import CaseInsensitiveDict
from tabulate import tabulate

from sortition_algorithms import errors
from sortition_algorithms.report_messages import get_message

# Create a configured converter for RunReport serialization
_converter = Converter()

if TYPE_CHECKING:
    from _typeshed import SupportsLenAndGetItem


def get_cell_name(row: int, col_name: str, headers: Sequence[str]) -> str:
    """Given the column_name, get the spreadsheet cell name, eg "A5" """
    col_index = headers.index(col_name)
    if col_index > 25:
        col1 = ["", *string.ascii_uppercase][col_index // 26]
        col2 = string.ascii_uppercase[col_index % 26]
        col_name = f"{col1}{col2}"
    else:
        col_name = string.ascii_uppercase[col_index]
    return f"{col_name}{row}"


def default_logging_setup() -> tuple[logging.Logger, logging.Logger]:
    """Set both logger and user_logger to send output to stdout"""
    # we have two loggers
    # - user_logger is used for messages that any user should see
    # - logger is used for messages that only a developer or admin should need to see
    user_logger = logging.getLogger("sortition_algorithms_user")
    user_logger.setLevel(logging.INFO)
    if not user_logger.handlers:
        # no set up has been done yet - so we do it here
        user_logger.addHandler(logging.StreamHandler(sys.stdout))
    logger = logging.getLogger("sortition_algorithms")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # no set up has been done yet - so we do it here
        # this logger just goes straight to stdout - no timestamps or anything
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return user_logger, logger


def _override_handlers_for(logger: logging.Logger, new_handlers: list[logging.Handler]) -> None:
    # first get rid of the old handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # now add the new handlers
    for handler in new_handlers:
        logger.addHandler(handler)


def override_logging_handlers(
    user_logger_handlers: list[logging.Handler], logger_handlers: list[logging.Handler]
) -> None:
    """Replace the default handlers with other ones"""
    if user_logger_handlers:
        _override_handlers_for(logging.getLogger("sortition_algorithms_user"), user_logger_handlers)
    if logger_handlers:
        _override_handlers_for(logging.getLogger("sortition_algorithms"), logger_handlers)


def set_log_level(log_level: int) -> None:
    user_logger.setLevel(log_level)
    logger.setLevel(log_level)


user_logger, logger = default_logging_setup()


class ReportLevel(enum.Enum):
    NORMAL = 0
    IMPORTANT = 1
    CRITICAL = 2


@define(slots=True, eq=True)
class RunLineLevel:
    line: str
    level: ReportLevel
    log_level: int = logging.NOTSET
    message_code: str | None = None
    message_params: dict[str, Any] = field(factory=dict)


@define(slots=True, eq=True)
class RunTable:
    headers: list[str]
    data: list[list[str | int | float]]


@define(slots=True, eq=True)
class RunError:
    error: Exception
    is_fatal: bool


# Configure cattrs to handle union types in RunReport
def _structure_table_cell(obj: Any, _: Any) -> str | int | float:
    """Structure hook for table cell values"""
    if isinstance(obj, str | int | float):
        return obj
    return str(obj)


def _unstructure_exception(exc: Exception) -> dict[str, Any]:
    """Unstructure hook for exceptions - store type and args"""
    result: dict[str, Any] = {
        "type": f"{exc.__class__.__module__}.{exc.__class__.__name__}",
        "args": _converter.unstructure(exc.args),
    }
    # Store error_code and error_params if they exist
    if hasattr(exc, "error_code"):
        result["error_code"] = exc.error_code
    if hasattr(exc, "error_params"):
        result["error_params"] = exc.error_params
    # Special handling for SelectionMultilineError which has custom attributes
    if hasattr(exc, "all_lines"):
        result["all_lines"] = exc.all_lines
    if hasattr(exc, "all_errors"):
        result["all_errors"] = _converter.unstructure(exc.all_errors)
    # this is for InfeasibleQuotasError
    if hasattr(exc, "features"):
        result["features"] = _converter.unstructure(exc.features)
    return result


def _structure_exception(obj: dict[str, Any], _: Any) -> Exception:
    """Structure hook for exceptions - reconstruct from type and args"""
    exc_type_name = obj["type"]
    error_code = obj.get("error_code", "")
    error_params = obj.get("error_params", {})

    # Map type names to actual exception classes
    if exc_type_name == "sortition_algorithms.errors.SelectionError":
        # Get the message from args if available
        message = obj.get("args", ("",))[0] if obj.get("args") else ""
        return errors.SelectionError(message, error_code=error_code, error_params=error_params)
    elif exc_type_name == "sortition_algorithms.errors.SelectionMultilineError":
        if "all_lines" in obj:
            return errors.SelectionMultilineError(obj["all_lines"], error_code=error_code, error_params=error_params)
        # Fallback for edge cases - reconstruct from message string
        message = obj.get("args", ("",))[0] if obj.get("args") else ""
        lines = message.split("\n") if message else []
        return errors.SelectionMultilineError(lines, error_code=error_code, error_params=error_params)
    elif exc_type_name == "sortition_algorithms.errors.ParseTableMultiError":
        all_errors = _converter.structure(
            obj.get("all_errors", []),
            list[errors.ParseTableErrorMsg | errors.ParseTableMultiValueErrorMsg],
        )
        return errors.ParseTableMultiError(all_errors)
    elif exc_type_name == "sortition_algorithms.errors.InfeasibleQuotasError":
        # avoid circular import
        from sortition_algorithms.features import FeatureCollection

        features = _converter.structure(obj.get("features", {}), FeatureCollection)  # type: ignore[type-abstract]
        output = _converter.structure(obj.get("all_lines", ["dummy"]), list[str])[1:]
        return errors.InfeasibleQuotasError(features=features, output=output)
    elif exc_type_name.startswith("sortition_algorithms.errors."):
        # For other custom errors, try to find the class
        class_name = exc_type_name.split(".")[-1]
        exc_class = getattr(errors, class_name, errors.SortitionBaseError)
        # Get the message from args if available
        message = obj.get("args", ("",))[0] if obj.get("args") else ""
        return exc_class(message, error_code=error_code, error_params=error_params)
    else:
        # For built-in exceptions, reconstruct as generic Exception
        return Exception(*obj.get("args", ()))


_converter.register_structure_hook(str | int | float, _structure_table_cell)
_converter.register_unstructure_hook(Exception, _unstructure_exception)
_converter.register_structure_hook(Exception, _structure_exception)


@define(eq=True)
class RunReport:
    """A class to hold a report to show to the user at the end"""

    _data: list[RunLineLevel | RunTable | RunError] = field(factory=list)

    def __bool__(self) -> bool:
        """
        Basically, False is the report is empty, or True if there is some content. So you can do
        things like

        ```
        if run_report:
            print(f"Run Report\n\n{run_report.as_text()}")
        ```
        """
        return self.has_content()

    def serialize(self) -> dict[str, Any]:
        return _converter.unstructure(self)  # type: ignore[no-any-return]

    @classmethod
    def deserialize(cls, serialized_data: dict[str, Any]) -> "RunReport":
        return _converter.structure(serialized_data, cls)

    def has_content(self) -> bool:
        """
        False is the report is empty, or True if there is some content. So you can do
        things like

        ```
        if run_report.has_content():
            print(f"Run Report\n\n{run_report.as_text()}")
        ```
        """
        return bool(self._data)

    def add_line(
        self,
        line: str,
        level: ReportLevel = ReportLevel.NORMAL,
        message_code: str | None = None,
        message_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a line of text, and a level - so important/critical messages can be highlighted in the HTML report.

        Args:
            line: The English message text (for backward compatibility and standalone use)
            level: Importance level of the message
            message_code: Optional translation key for i18n (e.g., "loading_features_from_file")
            message_params: Optional parameters for message translation (e.g., {"file_path": "features.csv"})
        """
        self._data.append(RunLineLevel(line, level, message_code=message_code, message_params=message_params or {}))

    def add_line_and_log(
        self,
        line: str,
        log_level: int,
        message_code: str | None = None,
        message_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a line of text, and a level - so important/critical messages can be highlighted in the HTML report.

        This method will also log the message to the `user_logger`. This message can be shown to the user as
        the run is happening, so the user has feedback on what is going on while the run is in progress.

        When generating the report we can skip those messages, to avoid duplication. But if the user_logger
        has not been set up to be shown to the user during the run, we do want those messages to be in the
        final report.

        Args:
            line: The English message text (for backward compatibility and standalone use)
            log_level: Logging level for the message
            message_code: Optional translation key for i18n (e.g., "trial_number")
            message_params: Optional parameters for message translation (e.g., {"trial": 3})
        """
        self._data.append(
            RunLineLevel(
                line, ReportLevel.NORMAL, log_level, message_code=message_code, message_params=message_params or {}
            )
        )
        user_logger.log(level=log_level, msg=line)

    def add_message(self, code: str, level: ReportLevel = ReportLevel.NORMAL, **params: Any) -> None:
        """
        Add a translatable message using a message code and parameters.

        This is a convenience method that combines get_message() and add_line() in one call,
        making it simpler to add messages with translation support.

        Args:
            code: The message code from REPORT_MESSAGES (e.g., "loading_features_from_file")
            level: Importance level of the message
            **params: Parameters to substitute into the message template

        Example:
            >>> report.add_message("features_found", count=5)
            >>> report.add_message("trial_number", ReportLevel.IMPORTANT, trial=3)
        """
        message = get_message(code, **params)
        self.add_line(message, level=level, message_code=code, message_params=params)

    def add_message_and_log(self, code: str, log_level: int, **params: Any) -> None:
        """
        Add a translatable message using a message code and parameters, and log it.

        This is a convenience method that combines get_message() and add_line_and_log() in one call,
        making it simpler to add messages with translation support that are also logged.

        Args:
            code: The message code from REPORT_MESSAGES (e.g., "trial_number")
            log_level: Logging level for the message
            **params: Parameters to substitute into the message template

        Example:
            >>> report.add_message_and_log("trial_number", logging.WARNING, trial=3)
            >>> report.add_message_and_log("basic_solution_warning", logging.WARNING, algorithm="maximin", num_panels=150, num_agents=100, min_probs=0.001)
        """
        message = get_message(code, **params)
        self.add_line_and_log(message, log_level, message_code=code, message_params=params)

    def add_lines(self, lines: Iterable[str], level: ReportLevel = ReportLevel.NORMAL) -> None:
        """
        Add multiple lines of text with the same level.

        .. deprecated:: (next version)
            This method is deprecated. Functions should return RunReport instead of list[str],
            and callers should use add_report() to merge them. This provides better support
            for translation and structured reporting.
        """
        warnings.warn(
            "add_lines() is deprecated. Functions should return RunReport instead of list[str], "
            "and use add_report() to merge them.",
            DeprecationWarning,
            stacklevel=2,
        )
        for line in lines:
            self._data.append(RunLineLevel(line, level))

    def add_table(self, table_headings: list[str], table_data: list[list[str | int | float]]) -> None:
        self._data.append(RunTable(table_headings, table_data))

    def add_error(self, error: Exception, is_fatal: bool = True) -> None:
        self._data.append(RunError(error, is_fatal=is_fatal))

    def add_report(self, other: "RunReport") -> None:
        self._data += other._data

    def _element_to_text(self, element: RunLineLevel | RunTable | RunError, include_logged: bool) -> str | None:
        if isinstance(element, RunLineLevel):
            # we might want to skip lines that were already logged
            if include_logged or element.log_level == logging.NOTSET:
                return element.line
            else:
                # sometimes we want empty strings for blank lines, so here we return None
                # instead so the logged lines can be filtered out
                return None
        elif isinstance(element, RunTable):
            table_text = tabulate(element.data, headers=element.headers, tablefmt="simple")
            # we want a blank line before and after the table.
            return f"\n{table_text}\n"
        else:
            return str(element.error)

    def as_text(self, include_logged: bool = True) -> str:
        parts = [self._element_to_text(element, include_logged) for element in self._data]
        return "\n".join(p for p in parts if p is not None)

    def _line_to_html(self, line_level: RunLineLevel) -> str:
        tags = {
            ReportLevel.NORMAL: ("", ""),
            ReportLevel.IMPORTANT: ("<b>", "</b>"),
            ReportLevel.CRITICAL: ('<b style="color: red">', "</b>"),
        }
        start_tag, end_tag = tags[line_level.level]
        escaped_line = html.escape(line_level.line)
        return f"{start_tag}{escaped_line}{end_tag}"

    def _error_to_html(self, run_error: RunError) -> str:
        start_tag, end_tag = ("<b>", "</b>") if run_error.is_fatal else ("", "")
        if isinstance(run_error.error, errors.SortitionBaseError):
            return f"{start_tag}{run_error.error.to_html()}{end_tag}"
        # default to the string representation
        return f"{start_tag}{run_error.error}{end_tag}"

    def _element_to_html(self, element: RunLineLevel | RunTable | RunError, include_logged: bool) -> str | None:
        if isinstance(element, RunLineLevel):
            if include_logged or element.log_level == logging.NOTSET:
                return self._line_to_html(element)
            else:
                return None
        elif isinstance(element, RunTable):
            return tabulate(element.data, headers=element.headers, tablefmt="html")
        else:
            return self._error_to_html(element)

    def as_html(self, include_logged: bool = True) -> str:
        parts = [self._element_to_html(element, include_logged) for element in self._data]
        return "<br />\n".join(p for p in parts if p is not None)

    def last_error(self) -> Exception | None:
        for element in reversed(self._data):
            if isinstance(element, RunError):
                return element.error
        return None


def strip_str_int(value: str | int | float) -> str:
    return str(value).strip()


def normalise_dict(original: Mapping[str, str] | Mapping[str, str | int]) -> MutableMapping[str, str]:
    """
    Wraps a dict, and whenever we get a value from it, we convert to str and
    strip() whitespace
    """
    new_dict: MutableMapping[str, str] = CaseInsensitiveDict()
    for key, original_value in original.items():
        new_dict[key] = strip_str_int(original_value)
    return new_dict


class RandomProvider(ABC):
    """
    This is something of a hack. Mostly we want to use the `secrets` module.
    But for repeatable testing we might want to set the random.seed sometimes.

    So we have a global `_random_provider` which can be switched between an
    instance of this class that uses the `secrets` module and an instance that
    uses `random` with a seed. The switch is done by the `set_random_provider()`
    function.

    Then every time we want some randomness, we call `random_provider()` to get
    the current version of the global.
    """

    @classmethod
    @abstractmethod
    def uniform(cls, lower: float, upper: float) -> float: ...

    @classmethod
    @abstractmethod
    def randbelow(cls, upper: int) -> int: ...

    @classmethod
    @abstractmethod
    def choice(cls, seq: "SupportsLenAndGetItem[str]") -> str: ...


class GenRandom(RandomProvider):
    def __init__(self, seed: int) -> None:
        random.seed(seed)

    @classmethod
    def uniform(cls, lower: float, upper: float) -> float:
        return random.uniform(lower, upper)  # noqa: S311

    @classmethod
    def randbelow(cls, upper: int) -> int:
        return random.randrange(upper)  # noqa: S311

    @classmethod
    def choice(cls, seq: "SupportsLenAndGetItem[str]") -> str:
        return random.choice(seq)  # noqa: S311


class GenSecrets(RandomProvider):
    @classmethod
    def uniform(cls, lower: float, upper: float) -> float:
        assert upper > lower
        diff = upper - lower
        rand_int = secrets.randbelow(1_000_000)
        return lower + (rand_int * diff / 1_000_000)

    @classmethod
    def randbelow(cls, upper: int) -> int:
        return secrets.randbelow(upper)

    @classmethod
    def choice(cls, seq: "SupportsLenAndGetItem[str]") -> str:
        return secrets.choice(seq)


_random_provider: RandomProvider = GenSecrets()


def set_random_provider(seed: int | None = None) -> None:
    global _random_provider
    if seed:
        _random_provider = GenRandom(seed)
        return
    _random_provider = GenSecrets()


def random_provider() -> RandomProvider:
    return _random_provider
