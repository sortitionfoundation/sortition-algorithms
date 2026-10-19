# ABOUTME: Helpers for opening a Google Sheet by URL or file id with precise access-failure classification.
# ABOUTME: Also holds the gspread client factory shared by GSheetDataSource and other callers.

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gspread
from gspread.urls import DRIVE_FILES_API_V3_URL, SPREADSHEET_DRIVE_URL
from gspread.utils import extract_id_from_url
from oauth2client.service_account import ServiceAccountCredentials

from sortition_algorithms.errors import (
    NotNativeGoogleSheetError,
    SpreadsheetNotFoundError,
    SpreadsheetNotSharedError,
    SpreadsheetReadOnlyError,
)

GSHEET_SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


def make_gsheet_client(
    auth_json_path: Path,
    request_timeout: float | tuple[float, float] | None = 60,
    http_client: type[gspread.HTTPClient] = gspread.BackOffHTTPClient,
) -> gspread.client.Client:
    """
    Build an authorised gspread client from a service account JSON file.

    Args:
    - auth_json_path - path to the file containing the google service account details.
    - request_timeout - How long to wait for the server to send data before giving up, as a float,
      or a (connect timeout, read timeout) tuple. Value for timeout is in seconds.
    - http_client - the gspread HTTP client class to use. The default, BackOffHTTPClient,
      sleeps and retries with doubling waits (up to 128 seconds) when Google answers with a
      rate limit or server error, which suits a long-running selection. Pass gspread.HTTPClient
      to fail fast instead, for example from a web request that must not stall.
    """
    creds = ServiceAccountCredentials.from_json_keyfile_name(str(auth_json_path), GSHEET_SCOPE)
    client = gspread.authorize(creds, http_client=http_client)
    client.set_timeout(request_timeout)
    return client


def service_account_email(client: gspread.client.Client) -> str:
    """The email of the service account the client authenticates as, or "" if unknown."""
    auth = getattr(client.http_client, "auth", None)
    return str(getattr(auth, "service_account_email", "") or "")


@dataclass(frozen=True)
class GSheetInfo:
    """What the service account can see and do with an opened Google Sheet."""

    spreadsheet: gspread.Spreadsheet
    file_id: str
    title: str
    url: str
    mimetype: str
    can_edit: bool
    service_account_email: str = ""

    def require_writable(self) -> None:
        """Raise SpreadsheetReadOnlyError if the service account cannot edit the sheet."""
        if not self.can_edit:
            raise SpreadsheetReadOnlyError(
                spreadsheet_name=self.url,
                title=self.title,
                service_account_email=self.service_account_email,
            )


def _is_drive_not_found(error: gspread.exceptions.APIError) -> bool:
    if error.code != 404:
        return False
    errors = error.error.get("errors") or []
    return bool(errors) and errors[0].get("reason") == "notFound"


def _classify_drive_not_found(
    client: gspread.client.Client, file_id: str, url_or_id: str, drive_error: gspread.exceptions.APIError
) -> None:
    """
    Drive reports both "does not exist" and "not shared with you" as 404, so ask
    the Sheets API, which distinguishes them (403 vs 404). Always raises: the
    matching error, or the original Drive error if the Sheets API says anything
    unexpected.
    """
    try:
        client.open_by_key(file_id)
    except PermissionError as err:
        raise SpreadsheetNotSharedError(
            spreadsheet_name=url_or_id, service_account_email=service_account_email(client)
        ) from err
    except gspread.exceptions.SpreadsheetNotFound as err:
        raise SpreadsheetNotFoundError(spreadsheet_name=url_or_id) from err
    except Exception as err:
        # the Sheets API did not give a classification we understand, so the
        # Drive error is the one the caller should see
        raise drive_error from err
    raise drive_error


@dataclass(frozen=True)
class _DriveFile:
    """What the Drive API says about a file that is a native Google Sheet."""

    file_id: str
    mimetype: str
    can_edit: bool


def _check_drive_file(client: gspread.client.Client, url_or_id: str) -> _DriveFile:
    """
    Ask the Drive API about the file and check it is a native Google Sheet.
    Raises the access errors documented on open_gsheet().
    """
    file_id = extract_id_from_url(url_or_id) if url_or_id.startswith("https://") else url_or_id
    try:
        response = client.http_client.request(
            "get",
            f"{DRIVE_FILES_API_V3_URL}/{file_id}",
            params={"supportsAllDrives": True, "fields": "mimeType,name,capabilities/canEdit"},
        )
    except gspread.exceptions.APIError as drive_error:
        if _is_drive_not_found(drive_error):
            _classify_drive_not_found(client, file_id, url_or_id, drive_error)
        raise
    metadata: dict[str, Any] = response.json()
    mimetype = metadata.get("mimeType", "")
    if mimetype != NotNativeGoogleSheetError.NATIVE_GSHEET_MIMETYPE:
        raise NotNativeGoogleSheetError(mimetype=mimetype, file_name=metadata.get("name", ""))
    return _DriveFile(
        file_id=file_id,
        mimetype=mimetype,
        can_edit=bool(metadata.get("capabilities", {}).get("canEdit", False)),
    )


def open_gsheet(client: gspread.client.Client, url_or_id: str) -> GSheetInfo:
    """
    Open a Google Sheet by URL or file id and report what the service account can do with it.

    Raises:
    - SpreadsheetNotFoundError - the id does not exist
    - SpreadsheetNotSharedError - exists, but not shared with this service account
    - NotNativeGoogleSheetError - the Drive file is not a native Google Sheet (eg an uploaded .xlsx)
    - gspread.exceptions.APIError - anything else from Google (quota, 5xx, bad credentials)
    - requests.exceptions.RequestException - network failures (connection errors, timeouts),
      propagated unwrapped from requests
    - google.auth.exceptions.GoogleAuthError - failure to refresh the access token,
      propagated unwrapped from google-auth
    """
    drive_file = _check_drive_file(client, url_or_id)
    spreadsheet = client.open_by_key(drive_file.file_id)
    return GSheetInfo(
        spreadsheet=spreadsheet,
        file_id=drive_file.file_id,
        title=spreadsheet.title,
        url=spreadsheet.url,
        mimetype=drive_file.mimetype,
        can_edit=drive_file.can_edit,
        service_account_email=service_account_email(client),
    )


@dataclass(frozen=True)
class GSheetSummary:
    """What the service account can see of a Google Sheet, without opening it for use."""

    file_id: str
    title: str
    url: str
    mimetype: str
    can_edit: bool
    worksheet_titles: tuple[str, ...]
    # when the Sheets metadata was fetched, so a caller that keeps the summary
    # can decide when it is too old to trust
    fetched_at: datetime = field(compare=False)
    service_account_email: str = ""


def inspect_gsheet(client: gspread.client.Client, url_or_id: str) -> GSheetSummary:
    """
    Report what the service account can do with a Google Sheet, including its tab
    titles, using one Drive and one Sheets request. Unlike open_gsheet() it does not
    build a gspread.Spreadsheet, which would cost a further Sheets request.

    Raises:
    - SpreadsheetNotFoundError - the id does not exist
    - SpreadsheetNotSharedError - exists, but not shared with this service account
    - NotNativeGoogleSheetError - the Drive file is not a native Google Sheet (eg an uploaded .xlsx)
    - gspread.exceptions.APIError - anything else from Google (quota, 5xx, bad credentials),
      including the Sheets API refusing a file that Drive allowed (eg the Sheets API
      is not enabled for the project)
    - requests.exceptions.RequestException - network failures (connection errors, timeouts),
      propagated unwrapped from requests
    - google.auth.exceptions.GoogleAuthError - failure to refresh the access token,
      propagated unwrapped from google-auth
    """
    drive_file = _check_drive_file(client, url_or_id)
    metadata = client.http_client.fetch_sheet_metadata(
        drive_file.file_id,
        params={"includeGridData": "false", "fields": "properties.title,sheets.properties.title"},
    )
    fetched_at = datetime.now(UTC)
    # the Sheets API leaves out fields that hold their default value, so an empty
    # title arrives as a missing key
    return GSheetSummary(
        file_id=drive_file.file_id,
        title=metadata.get("properties", {}).get("title", ""),
        url=SPREADSHEET_DRIVE_URL % drive_file.file_id,
        mimetype=drive_file.mimetype,
        can_edit=drive_file.can_edit,
        worksheet_titles=tuple(sheet.get("properties", {}).get("title", "") for sheet in metadata.get("sheets", [])),
        fetched_at=fetched_at,
        service_account_email=service_account_email(client),
    )
