# ABOUTME: Registry of all error messages for translation extraction and i18n support.
# ABOUTME: The N_() function marks strings for babel extraction without requiring translation dependencies.

from typing import Any


def N_(message: str) -> str:
    """
    No-op marker for translation extraction.

    This function does nothing at runtime but marks strings for extraction
    by babel/gettext tools. It's a common pattern for libraries that want
    to be translation-ready without depending on translation frameworks.

    Args:
        message: The message string to mark for extraction

    Returns:
        The same message string, unchanged
    """
    return message


# Error message templates
# These are organized by error type and use Python's % formatting for parameters
# The keys are error codes that can be used by consuming applications for translation

ERROR_MESSAGES = {
    # ========================================================================
    # Simple error messages (BadDataError, SelectionError, ValueError, etc.)
    # ========================================================================
    # BadDataError messages
    "missing_column": N_("No '%(column)s' column %(error_label)s found in %(data_container)s!"),
    "duplicate_column": N_("MORE THAN 1 '%(column)s' column %(error_label)s found in %(data_container)s!"),
    # SelectionError messages
    "person_not_found": N_("Failed to find person at position %(position)s in %(feature_name)s/%(feature_value)s"),
    "logic_error_tab_suffix": N_("Logic error - trying to create new tab before choosing suffix"),
    "spreadsheet_not_found": N_("Google spreadsheet not found: %(spreadsheet_name)s."),
    "tab_not_found": N_(
        "Error in Google sheet: no tab called '%(tab_name)s' found in spreadsheet '%(spreadsheet_title)s'."
    ),
    "variables_without_value": N_("It seems like some variables do not have a value. Original exception: %(error)s."),
    "already_selected_duplicate_headers": N_(
        "the header row in the %(tab_name)s tab contains duplicates: %(duplicates)s"
    ),
    # ConfigurationError messages
    "test_selection_multiple_selections": N_(
        "Running the test selection does not support generating a transparent lottery, so, if "
        "`test_selection` is true, `number_selections` must be 1."
    ),
    "legacy_multiple_selections": N_(
        "Currently, the legacy algorithm does not support generating a transparent lottery, "
        "so `number_selections` must be set to 1."
    ),
    "diversimax_multiple_selections": N_(
        "The diversimax algorithm does not support generating multiple committees, "
        "so `number_selections` must be set to 1."
    ),
    "unknown_selection_algorithm": N_(
        "Unknown selection algorithm %(algorithm)r, must be either 'legacy', 'leximin', 'maximin', 'diversimax', or 'nash'."
    ),
    "invalid_solver_backend": N_("solver_backend %(backend)s is not one of: %(valid_backends)s"),
    "unknown_solver_backend": N_("Unknown solver backend: %(backend)s"),
    "feature_column_not_found": N_("Could not find feature column, looked for column headers: %(column_names)s"),
    "feature_value_column_not_found": N_(
        "Could not find feature value column, looked for column headers: %(column_names)s"
    ),
    "missing_required_column": N_("Did not find required column name '%(field_name)s' in the input"),
    "duplicate_required_column": N_("Found MORE THAN 1 column named '%(field_name)s' in the input (found %(count)s)"),
    "unexpected_column_error": N_("Unexpected error in set of column names: %(headers)s"),
    # TypeError messages
    "check_same_address_not_list": N_("check_same_address_columns must be a LIST of strings"),
    "check_same_address_not_strings": N_("check_same_address_columns must be a list of STRINGS"),
    "check_same_address_empty": N_("check_same_address is TRUE but there are no columns listed to check."),
    "columns_to_keep_not_list": N_("columns_to_keep must be a LIST of strings"),
    "columns_to_keep_not_strings": N_("columns_to_keep must be a list of STRINGS"),
    "invalid_selection_algorithm": N_("selection_algorithm %(algorithm)s is not one of: %(valid_algorithms)s"),
    # RuntimeError messages
    "diversimax_not_available": N_(
        "Diversimax algorithm requires the optional 'diversimax' dependencies, which are not available"
    ),
    "gurobi_not_available": N_("Leximin algorithm requires Gurobi solver which is not available"),
    # ========================================================================
    # ParseTable error messages (for structured validation errors)
    # ========================================================================
    # Feature validation errors
    "no_value_set": N_("There is no %(field)s value set"),
    "not_a_number": N_("'%(value)s' is not a number"),
    "empty_feature_value": N_("Empty %(value_column_name)s in %(feature_column_name)s %(feature_name)s"),
    "min_greater_than_max": N_("Minimum (%(min)s) should not be greater than maximum (%(max)s)"),
    "min_flex_greater_than_min": N_("min_flex (%(min_flex)s) should not be greater than min (%(min)s)"),
    "max_flex_less_than_max": N_("max_flex (%(max_flex)s) should not be less than max (%(max)s)"),
    # People validation errors
    "value_not_in_feature": N_("Value '%(value)s' not in %(feature_column_name)s %(feature_name)s"),
    "empty_value_in_feature": N_("Empty value in %(feature_column_name)s %(feature_name)s"),
    # ========================================================================
    # ParseTable formatting templates (for error display context)
    # ========================================================================
    "parse_error_single_column": N_("%(msg)s: for row %(row)s, column header %(key)s"),
    "parse_error_multi_column": N_("%(msg)s: for row %(row)s, column headers %(keys)s"),
    # ========================================================================
    # Customise method templates (adding context to parse errors)
    # ========================================================================
    "parser_error_features": N_("Parser error(s) while reading features from '%(tab_name)s' worksheet"),
    "parser_error_people": N_("Parser error(s) while reading people from '%(tab_name)s' worksheet"),
    "parser_error_already_selected": N_("Parser error(s) while reading people from '%(tab_name)s' worksheet"),
    "parse_error_see_cell": N_("%(msg)s - see cell %(cell_name)s"),
    "parse_error_see_cells": N_("%(msg)s - see cells %(cell_names)s"),
    # ========================================================================
    # SelectionMultilineError - multiline error templates
    # ========================================================================
    "duplicate_id_header": N_("Found %(count)s IDs that have more than one row with different data"),
    "duplicate_id_row": N_("For ID '%(person_id)s' one row of data is: %(row_data)s"),
    "not_enough_people": N_(
        "Not enough people with the value '%(feature_value)s' in category '%(feature_name)s' - "
        "the minimum is %(min)s but we only have %(count)s"
    ),
    "desired_out_of_range": N_(
        "The number of people to select (%(desired)s) is out of the range of "
        "the numbers of people in the %(feature_name)s feature. It should be within "
        "[%(min)s, %(max)s]."
    ),
    # InfeasibleQuotasError
    "infeasible_quotas_header": N_(
        "It is not possible to hit all the targets with the current set of people. I suggest the following steps:"
    ),
    # Min/max error details
    "inconsistent_min_max_header": N_("Inconsistent numbers in min and max in the %(feature_column_name)s input:"),
    "smallest_maximum_detail": N_("The smallest maximum is %(max_val)s for %(feature_column_name)s '%(max_feature)s'"),
    "largest_minimum_detail": N_("The largest minimum is %(min_val)s for %(feature_column_name)s '%(min_feature)s'"),
    "min_max_fix_suggestion": N_(
        "You need to reduce the minimums for %(min_feature)s or increase the maximums for %(max_feature)s."
    ),
}


def get_message(code: str, **params: Any) -> str:
    """
    Get a formatted error message by code.

    This is a helper function for creating error messages that supports both
    English output and provides structured data for translation.

    Args:
        code: The error code to look up in ERROR_MESSAGES
        **params: Parameters to substitute into the message template

    Returns:
        The formatted message string

    Raises:
        KeyError: If the error code is not found in ERROR_MESSAGES

    Example:
        >>> get_message('missing_column', column='id', error_label='for people', data_container='CSV file')
        "No 'id' column for people found in CSV file!"
    """
    template = ERROR_MESSAGES[code]
    return template % params
