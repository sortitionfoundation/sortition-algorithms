# ABOUTME: Registry of all report messages for translation extraction and i18n support.
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


# Report message templates
# These are organized by category and use Python's % formatting for parameters
# The keys are message codes that can be used by consuming applications for translation

REPORT_MESSAGES = {
    # ========================================================================
    # Data loading messages
    # ========================================================================
    "loading_features_from_string": N_("Loading features from string."),
    "loading_people_from_string": N_("Loading people from string."),
    "loading_already_selected_from_string": N_("Loading already selected people from string."),
    "no_already_selected_data": N_("No already selected data provided, using empty data."),
    "loading_features_from_file": N_("Loading features from file %(file_path)s."),
    "loading_people_from_file": N_("Loading people from file %(file_path)s."),
    "loading_already_selected_from_file": N_("Loading already selected people from file %(file_path)s."),
    "no_already_selected_file": N_("No already selected file provided or file does not exist, using empty data."),
    "reading_gsheet_tab": N_("Reading in '%(tab_name)s' tab in above Google sheet."),
    "no_already_selected_tab": N_("No already selected tab specified, using empty data."),
    "features_found": N_("Number of features found: %(count)s"),
    # ========================================================================
    # Algorithm and selection messages
    # ========================================================================
    "using_legacy_algorithm": N_("Using legacy algorithm."),
    "gurobi_unavailable_switching": N_(
        "The leximin algorithm requires the optimization library Gurobi to be installed "
        "(commercial, free academic licenses available). Switching to the simpler "
        "maximin algorithm, which can be run using open source solvers."
    ),
    "distribution_stats": N_(
        "Algorithm produced distribution over %(total_committees)s committees, out of which "
        "%(non_zero_committees)s are chosen with positive probability."
    ),
    "basic_solution_warning": N_(
        "INFO: The distribution over panels is what is known as a 'basic solution'. There is no reason for concern "
        "about the correctness of your output, but we'd appreciate if you could reach out to panelot"
        "@paulgoelz.de with the following information: algorithm=%(algorithm)s, "
        "num_panels=%(num_panels)s, num_agents=%(num_agents)s, min_probs=%(min_probs)s."
    ),
    # ========================================================================
    # Selection process messages
    # ========================================================================
    "test_selection_warning": N_("WARNING: Panel is not selected at random! Only use for testing!"),
    "initial_state": N_("Initial: (selected = 0)"),
    "trial_number": N_("Trial number: %(trial)s"),
    "selection_success": N_("SUCCESS!! Final:"),
    "selection_failed": N_("Failed %(attempts)s times. Gave up."),
    "retry_after_error": N_("Failed one attempt. Selection Error raised - will retry. %(error)s"),
    "no_target_checks_multiple": N_("No target checks done for multiple selections - please see your output files."),
    "no_category_info_multiple": N_(
        "We do not calculate target details for multiple selections - please see your output files."
    ),
    # ========================================================================
    # Validation and warnings
    # ========================================================================
    "blank_id_skipped": N_("WARNING: blank cell found in ID column in row %(row)s - skipped that line!"),
    "duplicate_people_header": N_("WARNING: Duplicate people found:"),
    "duplicate_person_details": N_("Person with ID '%(person_id)s' appears %(count)s times in rows: %(rows)s"),
    # ========================================================================
    # Settings messages
    # ========================================================================
    "using_settings": N_("Using these settings: %(settings)s"),
    "random_seed_set": N_("Random seed set to: %(seed)s"),
}


def get_message(code: str, **params: Any) -> str:
    """
    Get a formatted report message by code.

    This is a helper function for creating report messages that supports both
    English output and provides structured data for translation.

    Args:
        code: The message code to look up in REPORT_MESSAGES
        **params: Parameters to substitute into the message template

    Returns:
        The formatted message string

    Raises:
        KeyError: If the message code is not found in REPORT_MESSAGES

    Example:
        >>> get_message('loading_features_from_file', file_path='/path/to/features.csv')
        'Loading features from file /path/to/features.csv.'
    """
    template = REPORT_MESSAGES[code]
    return template % params
