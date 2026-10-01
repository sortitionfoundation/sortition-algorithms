"""
Tests for opening a Google Sheet via open_gsheet() with precise failure classification.

No live Google API: the Drive and Sheets responses are the bodies recorded
against the real API (gspread 6.2.1) for each case.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import gspread
import pytest
import requests
from gspread.urls import DRIVE_FILES_API_V3_URL

from sortition_algorithms import gsheet
from sortition_algorithms.errors import (
    NotNativeGoogleSheetError,
    SpreadsheetNotFoundError,
    SpreadsheetNotSharedError,
    SpreadsheetReadOnlyError,
)
from sortition_algorithms.gsheet import GSHEET_SCOPE, GSheetInfo, make_gsheet_client, open_gsheet, service_account_email

FILE_ID = "1B-S6esBj7rqbSJulqAZUh4-x1FXU-rPdNabMVDtsynM"
URL = f"https://docs.google.com/spreadsheets/d/{FILE_ID}/edit"
EMAIL = "demo-server@example.iam.gserviceaccount.com"
NATIVE = "application/vnd.google-apps.spreadsheet"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Recorded response bodies
DRIVE_NOT_FOUND_BODY = {
    "error": {
        "code": 404,
        "message": f"File not found: {FILE_ID}.",
        "errors": [
            {
                "message": f"File not found: {FILE_ID}.",
                "domain": "global",
                "reason": "notFound",
                "location": "fileId",
                "locationType": "parameter",
            }
        ],
    }
}
SHEETS_PERMISSION_DENIED_BODY = {
    "error": {"code": 403, "message": "The caller does not have permission", "status": "PERMISSION_DENIED"}
}
SHEETS_NOT_FOUND_BODY = {"error": {"code": 404, "message": "Requested entity was not found.", "status": "NOT_FOUND"}}
DRIVE_QUOTA_BODY = {
    "error": {
        "code": 403,
        "message": "Rate Limit Exceeded",
        "errors": [{"message": "Rate Limit Exceeded", "domain": "usageLimits", "reason": "rateLimitExceeded"}],
    }
}


def _response(status_code: int, body: dict[str, Any]) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(body).encode()
    return response


def _api_error(status_code: int, body: dict[str, Any]) -> gspread.exceptions.APIError:
    return gspread.exceptions.APIError(_response(status_code, body))


def _permission_error() -> PermissionError:
    """What gspread's open_by_key raises for a Sheets 403: bare PermissionError chained to the APIError."""
    error = PermissionError()
    error.__cause__ = _api_error(403, SHEETS_PERMISSION_DENIED_BODY)
    return error


def _spreadsheet_not_found() -> gspread.exceptions.SpreadsheetNotFound:
    error = gspread.exceptions.SpreadsheetNotFound(_response(404, SHEETS_NOT_FOUND_BODY))
    error.__cause__ = _api_error(404, SHEETS_NOT_FOUND_BODY)
    return error


def _make_client(
    drive: dict[str, Any] | Exception,
    sheets: Exception | None = None,
    email: str = EMAIL,
) -> MagicMock:
    """
    A fake gspread client. `drive` is the Drive files.get body, or the exception
    it raises. `sheets` is the exception open_by_key raises; when None it returns
    a fake Spreadsheet.
    """
    client = MagicMock()
    if isinstance(drive, Exception):
        client.http_client.request.side_effect = drive
    else:
        client.http_client.request.return_value.json.return_value = drive
    if sheets is not None:
        client.open_by_key.side_effect = sheets
    else:
        spreadsheet = MagicMock()
        spreadsheet.id = FILE_ID
        spreadsheet.title = "Demo Sheet"
        spreadsheet.url = f"https://docs.google.com/spreadsheets/d/{FILE_ID}"
        client.open_by_key.return_value = spreadsheet
    client.http_client.auth.service_account_email = email
    return client


def _drive_body(mimetype: str = NATIVE, can_edit: bool = True, name: str = "Demo Sheet") -> dict[str, Any]:
    return {"name": name, "mimeType": mimetype, "capabilities": {"canEdit": can_edit}}


def test_open_gsheet_shared_editable():
    client = _make_client(_drive_body())
    info = open_gsheet(client, URL)
    assert isinstance(info, GSheetInfo)
    assert info.file_id == FILE_ID
    assert info.title == "Demo Sheet"
    assert info.url == f"https://docs.google.com/spreadsheets/d/{FILE_ID}"
    assert info.mimetype == NATIVE
    assert info.can_edit is True
    assert info.service_account_email == EMAIL
    assert info.spreadsheet is client.open_by_key.return_value
    info.require_writable()  # no error
    client.http_client.request.assert_called_once_with(
        "get",
        f"{DRIVE_FILES_API_V3_URL}/{FILE_ID}",
        params={"supportsAllDrives": True, "fields": "mimeType,name,capabilities/canEdit"},
    )
    client.open_by_key.assert_called_once_with(FILE_ID)


def test_open_gsheet_shared_viewer_is_read_only():
    client = _make_client(_drive_body(can_edit=False))
    info = open_gsheet(client, URL)
    assert info.can_edit is False
    with pytest.raises(SpreadsheetReadOnlyError) as excinfo:
        info.require_writable()
    err = excinfo.value
    assert err.error_code == "spreadsheet_read_only"
    assert err.error_params == {
        "spreadsheet_name": info.url,
        "title": "Demo Sheet",
        "service_account_email": EMAIL,
    }
    assert "Demo Sheet" in str(err)
    assert EMAIL in str(err)


def test_open_gsheet_not_shared():
    drive_error = _api_error(404, DRIVE_NOT_FOUND_BODY)
    client = _make_client(drive_error, sheets=_permission_error())
    with pytest.raises(SpreadsheetNotSharedError) as excinfo:
        open_gsheet(client, URL)
    err = excinfo.value
    assert err.error_code == "spreadsheet_not_shared"
    assert err.error_params == {"spreadsheet_name": URL, "service_account_email": EMAIL}
    assert EMAIL in str(err)
    assert isinstance(err.__cause__, PermissionError)
    # the classification call was made exactly once
    client.open_by_key.assert_called_once_with(FILE_ID)


def test_open_gsheet_missing():
    drive_error = _api_error(404, DRIVE_NOT_FOUND_BODY)
    client = _make_client(drive_error, sheets=_spreadsheet_not_found())
    with pytest.raises(SpreadsheetNotFoundError) as excinfo:
        open_gsheet(client, URL)
    err = excinfo.value
    assert err.error_code == "spreadsheet_not_found"
    assert err.error_params == {"spreadsheet_name": URL}
    assert isinstance(err.__cause__, gspread.exceptions.SpreadsheetNotFound)


def test_open_gsheet_xlsx_is_not_native():
    client = _make_client(_drive_body(mimetype=XLSX, name="upload.xlsx"))
    with pytest.raises(NotNativeGoogleSheetError) as excinfo:
        open_gsheet(client, URL)
    err = excinfo.value
    assert err.mimetype == XLSX
    assert err.file_name == "upload.xlsx"
    assert err.error_code == "not_native_gsheet"
    # never tried to open a file that is not a Sheet
    client.open_by_key.assert_not_called()


def test_open_gsheet_non_404_drive_error_propagates():
    drive_error = _api_error(403, DRIVE_QUOTA_BODY)
    client = _make_client(drive_error)
    with pytest.raises(gspread.exceptions.APIError) as excinfo:
        open_gsheet(client, URL)
    assert excinfo.value is drive_error
    client.open_by_key.assert_not_called()


def test_open_gsheet_404_without_not_found_reason_propagates():
    body = {"error": {"code": 404, "message": "Odd", "errors": [{"reason": "somethingElse", "domain": "global"}]}}
    drive_error = _api_error(404, body)
    client = _make_client(drive_error)
    with pytest.raises(gspread.exceptions.APIError) as excinfo:
        open_gsheet(client, URL)
    assert excinfo.value is drive_error
    client.open_by_key.assert_not_called()


def test_open_gsheet_404_with_unexpected_classification_propagates_drive_error():
    drive_error = _api_error(404, DRIVE_NOT_FOUND_BODY)
    sheets_error = _api_error(500, {"error": {"code": 500, "message": "Internal error", "status": "INTERNAL"}})
    client = _make_client(drive_error, sheets=sheets_error)
    with pytest.raises(gspread.exceptions.APIError) as excinfo:
        open_gsheet(client, URL)
    assert excinfo.value is drive_error
    assert excinfo.value.__cause__ is sheets_error


def test_open_gsheet_404_but_sheets_opens_propagates_drive_error():
    drive_error = _api_error(404, DRIVE_NOT_FOUND_BODY)
    client = _make_client(drive_error)  # open_by_key succeeds
    with pytest.raises(gspread.exceptions.APIError) as excinfo:
        open_gsheet(client, URL)
    assert excinfo.value is drive_error


def test_open_gsheet_accepts_bare_file_id():
    client = _make_client(_drive_body())
    info = open_gsheet(client, FILE_ID)
    assert info.file_id == FILE_ID
    client.http_client.request.assert_called_once()
    assert client.http_client.request.call_args.args[1] == f"{DRIVE_FILES_API_V3_URL}/{FILE_ID}"
    client.open_by_key.assert_called_once_with(FILE_ID)


def test_open_gsheet_not_shared_without_service_account_email():
    drive_error = _api_error(404, DRIVE_NOT_FOUND_BODY)
    client = _make_client(drive_error, sheets=_permission_error())
    del client.http_client.auth.service_account_email
    with pytest.raises(SpreadsheetNotSharedError) as excinfo:
        open_gsheet(client, URL)
    assert excinfo.value.error_params["service_account_email"] == ""


def test_service_account_email_missing_auth():
    client = MagicMock(spec=[])
    client.http_client = MagicMock(spec=[])
    assert service_account_email(client) == ""


def test_gsheet_info_is_frozen():
    info = GSheetInfo(spreadsheet=MagicMock(), file_id="x", title="t", url="u", mimetype=NATIVE, can_edit=True)
    with pytest.raises(AttributeError):
        info.can_edit = False  # type: ignore[misc]


def test_make_gsheet_client_authorises_with_backoff_and_timeout():
    with (
        patch.object(gsheet.ServiceAccountCredentials, "from_json_keyfile_name") as from_keyfile,
        patch.object(gsheet.gspread, "authorize") as authorize,
    ):
        client = make_gsheet_client(Path("/some/auth.json"), request_timeout=(5, 30))
    from_keyfile.assert_called_once_with("/some/auth.json", GSHEET_SCOPE)
    authorize.assert_called_once_with(from_keyfile.return_value, http_client=gspread.BackOffHTTPClient)
    assert client is authorize.return_value
    client.set_timeout.assert_called_once_with((5, 30))


def test_make_gsheet_client_accepts_a_fail_fast_http_client():
    """A caller in a web request can opt out of the back-off retries."""
    with (
        patch.object(gsheet.ServiceAccountCredentials, "from_json_keyfile_name") as from_keyfile,
        patch.object(gsheet.gspread, "authorize") as authorize,
    ):
        make_gsheet_client(Path("/some/auth.json"), http_client=gspread.HTTPClient)
    authorize.assert_called_once_with(from_keyfile.return_value, http_client=gspread.HTTPClient)
