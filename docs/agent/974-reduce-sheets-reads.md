# 974: Reduce Google Sheets API reads

## Aim

A service using this library should be less likely to hit the Google Sheets
rate limit of **60 read requests per minute per user** (there is a separate
60 per minute limit for writes). The "user" is the service account, so every
check and every selection the service runs shares one budget.

Drive API calls have their own, much larger quota and are not the concern here.

Call counts below are from reading gspread 6.2.1 (the version in `uv.lock`).

## Where the reads go today

| Operation                                   | Sheets reads | Why                                                                                                      |
| ------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------- |
| Access check via `open_gsheet()` + tab list | 2            | `open_by_key()` fetches metadata (1), `worksheets()` fetches it again (1)                                |
| `GSheetDataSource` open by URL              | 1            | `open_gsheet()` → `open_by_key()`                                                                        |
| `GSheetDataSource` open by title            | 2            | `client.open()` builds a `Spreadsheet` (1), then `open_gsheet(found.id)` builds another (1)              |
| `read_feature_data`                         | 4            | `_tab_exists` → `worksheets()`, `worksheet(name)` (metadata again), `row_values(1)`, `get_all_records()` |
| `read_people_data`                          | 4            | same pattern                                                                                             |
| `read_already_selected_data`                | 3            | `_tab_exists` → `worksheets()`, `worksheet(name)`, `get()`                                               |
| `write_selected`                            | 1            | `_get_tab_titles()` → `worksheets()`                                                                     |
| `write_remaining`                           | 0            | reuses the suffix chosen in `write_selected`                                                             |
| `highlight_dupes`                           | 1            | `_get_tab()` → `worksheets()`                                                                            |
| `delete_old_output_tabs`                    | 1            | `worksheets()`                                                                                           |

A full selection by URL with an already-selected tab costs about **14 Sheets
reads** (1 open, 11 reading the tabs, 2 writing output). Most of them re-fetch the same spreadsheet metadata.

Relevant gspread facts:

- `Spreadsheet.__init__` calls `fetch_sheet_metadata()`, keeps only
  `properties` and discards the tab list.
- `Spreadsheet.worksheets()` and `Spreadsheet.worksheet(title)` each call
  `fetch_sheet_metadata()` again.
- `Worksheet.row_values(n)` and `Worksheet.get_all_records()` each make one
  `values.get` request.
- `Worksheet(spreadsheet, properties, spreadsheet_id, client)` makes no
  request. `Spreadsheet.worksheet()` builds them this way from the metadata.

## Option A: one Sheets read per access check (`inspect_gsheet`)

A successful check today costs **2 Sheets reads plus 1 Drive read**:

1. Drive `files.get` (mimetype, `canEdit`).
2. Sheets `spreadsheets.get`, inside `client.open_by_key()`.
3. Sheets `spreadsheets.get` again, in `spreadsheet.worksheets()`, to get the
   tab list back.

Reads 2 and 3 fetch the same thing.

### Proposal

Add to `sortition_algorithms/gsheet.py`:

```python
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
    Report what the service account can do with a Google Sheet, using one Drive
    and one Sheets request.
    """
```

`fetched_at` records when the summary was taken. A caller that caches it (for
example, the service showing a "sheet is ready" status) can then decide when to
check again without wrapping the summary in its own timestamped record.

- Use a timezone-aware UTC `datetime` (`datetime.now(UTC)`). A wall-clock time
  is the right choice here, unlike Option B's cache, which uses the monotonic
  clock. The summary may be stored, logged or shown to a person, and a
  monotonic value means nothing outside the process.
- Set it after the Sheets request returns, so it never claims to be fresher
  than the data.
- `field(compare=False)` keeps equality about the sheet's contents: two
  summaries of an unchanged sheet taken a minute apart compare equal.
- The library doesn't decide what "stale" means. That depends on the caller.
- Tests: assert `fetched_at` falls between timestamps taken just before and just
  after the call, rather than mocking the clock.

- It shares the Drive call and `_classify_drive_not_found()` with
  `open_gsheet()`. Factor the first half of `open_gsheet()` (id extraction,
  the Drive request, the 404 classification and the mimetype check) into a
  private helper that both call.
- Instead of `open_by_key()`, it calls
  `client.http_client.fetch_sheet_metadata(file_id, params={"includeGridData": "false", "fields": "properties.title,sheets.properties.title"})`.
  Passing `params` replaces gspread's default of `includeGridData=false`, so
  include it explicitly. The `fields` mask keeps the response small for a
  sheet with many tabs.
- Build `url` as `SPREADSHEET_DRIVE_URL % file_id` (from `gspread.urls`),
  which is what `Spreadsheet.url`, and so `GSheetInfo.url`, returns. Do not use
  the API's `spreadsheetUrl`: it ends in `/edit`, so the two functions would
  return different URLs for the same sheet.
- The tab list includes hidden tabs, as `worksheets()` does by default.
- It doesn't build a `gspread.Spreadsheet`, because building one costs a
  request. Callers that need to read or write (`GSheetDataSource`, the export
  adapter) keep using `open_gsheet()`.
- On the failure path the cost is the same as today (Drive 404, then one Sheets
  call to classify it).

### Errors

The docstring should list the errors precisely rather than saying "the same as
`open_gsheet()`". There is one difference. When Drive succeeds but Sheets
refuses, `open_by_key()` turns a 403 into a bare `PermissionError` and a 404
into `gspread.SpreadsheetNotFound`. A direct `fetch_sheet_metadata()` raises
`gspread.exceptions.APIError`. This happens for real when the Sheets API is not
enabled on the GCP project (403 `SERVICE_DISABLED`). `APIError` matches what
`open_gsheet()`'s docstring promises for "anything else from Google", so
`inspect_gsheet()` is arguably the more correct of the two. Decide whether
`open_gsheet()` should be brought in line (separate change).

### Why a new function and not a change to `GSheetInfo`

Adding `worksheet_titles` to `GSheetInfo` seems more natural. But the tab list
is only available from the metadata request that `Spreadsheet.__init__` makes
and then discards, so it can't be kept without a second request (which saves
nothing) or reaching into gspread internals. A separate function that never
builds a `Spreadsheet` is the clean version.

### Also needed

- Export `GSheetSummary` and `inspect_gsheet` from `sortition_algorithms/__init__.py`.
- Document them in `docs/adapters.md` ("Opening a sheet and checking access")
  and `docs/api-reference.md`.
- Tests in `tests/test_gsheet.py` alongside the `open_gsheet` ones: success
  (title, url, tabs, `can_edit`), bare file id, each Drive failure
  classification, not-native mimetype, and that exactly one Sheets request is
  made on success.

**Result: 1 Sheets read per successful check, down from 2.**

## Option B: fetch the tab list once per open in `GSheetDataSource`

Removes the repeated `worksheets()` and `worksheet(name)` calls.

- When `info` opens the spreadsheet, also fetch the metadata once
  (`spreadsheet.fetch_sheet_metadata()`) and build a cache of
  `Worksheet(spreadsheet, item["properties"], spreadsheet.id, spreadsheet.client)`
  keyed by title. That mirrors how `Spreadsheet.worksheet()` builds them, and
  constructing a `Worksheet` makes no request.
- `_get_tab`, `_tab_exists` and `_get_tab_titles` read from the cache. The
  readers use the cached `Worksheet` instead of calling
  `spreadsheet.worksheet(name)`.
- Keep the cache in step with our own changes: add to it in `_create_tab`,
  remove from it in `delete_old_output_tabs`. Clear it wherever `_info` is
  reset (`set_g_sheet_name`, and the name-change branch of `info`).
- Opening still costs 1 read for `Spreadsheet.__init__` plus 1 for the
  metadata fetch. Avoiding that second read means building `Spreadsheet`
  without its constructor, which relies on gspread internals. Not recommended.

### Keeping the cache fresh

Reading happens at the start of a run, so the cache is fresh then. Writing
happens after the selection algorithm, which usually takes seconds but has
taken half an hour on real selections. During that time someone may add,
rename or delete tabs, including the output tabs from an earlier run. A stale
cache then causes two concrete failures:

- `find_unused_tab_suffix` picks a suffix someone has since used, and
  `add_worksheet` fails because the tab name already exists.
- `delete_old_output_tabs` tries to delete a tab that is already gone, or
  misses one created since.

So:

- Record when the cache was filled, using `time.monotonic()` (the age is all
  we need, and it ignores wall-clock jumps).
- Add `_refresh_tab_cache_if_older_than(max_age_seconds)`, which re-fetches the
  metadata (1 read) only when the cache is older than that.
- Call it with a class attribute `tab_cache_max_age_seconds = 60`. Match
  `new_tab_default_size_rows`, so a caller or test can change it on an instance.
- Call it at the two points where tab names are decided:
  - at the start of `write_selected`, before `find_unused_tab_suffix`.
    `write_remaining` reuses that suffix, so it needs no refresh of its own.
  - at the start of `delete_old_output_tabs`. That is normally a separate CLI
    command, so the cache is usually fresh anyway.
- `highlight_dupes` runs straight after `write_remaining`, and the tab it needs
  is one we created and cached ourselves, so it needs no refresh.
- If a read finds a tab missing, re-fetch once before raising "tab not found".
  This covers someone fixing a tab name in the sheet and re-running in the
  same process (for example a long-lived service holding the data source).
- Tests: set `tab_cache_max_age_seconds = 0` to force a refresh, and leave it
  large to show none happens. Count requests on the client to check both cases.

A slow selection pays 1 extra read. A fast one pays nothing.

### As implemented

Two changes from the above, both simplifications:

- The tab list is fetched lazily, on first use, not when `info` opens the
  spreadsheet. Callers that only want `get_title()` or `can_edit` don't pay
  for it.
- Rather than calling a refresh at chosen points, every use of the tab list
  re-fetches it if it is at least `tab_cache_max_age_seconds` old. That gives
  the same read counts for a selection, and also covers a long-lived service
  that reads the input tabs again more than a minute later.
  `tab_cache_max_age_seconds` is an instance attribute set in `__init__`, as
  `new_tab_default_size_rows` is. A value of 0 means fetch on every use.

The fetch uses the `fields` mask `sheets.properties`. The "tab not found"
re-fetch costs an extra read even when the list was fetched a moment earlier,
which is fine on an error path.

### Saving

Saves 2 each on the features, people and already-selected tabs (the
`worksheets()` check and the `worksheet()` call both go), and 1 each in
`write_selected` and `highlight_dupes`. That is offset by the extra metadata
read on open, and by 1 refresh read when the selection is slow.
**About 14 → 7 reads per selection (8 for a slow one).**

## Option C: read each tab with one `values.get`

`read_feature_data` and `read_people_data` call `row_values(1)` and then
`get_all_records()`, which is two requests for the same data.
`read_already_selected_data` already does it in one `get()`.

- Replace both with one `worksheet.get(...)` and take the header from row 0,
  then build the records from the remaining rows.
- Must match `get_all_records()` behaviour exactly:
  - formatted values (its default `value_render_option`)
  - `numericise_ignore=["all"]` for people, numericising for features (the
    features path currently numericises, then `_stringify_records` turns the
    numbers back into strings; check whether that round trip changes anything,
    e.g. `"007"` → `7` → `"7"`, and whether keeping it is intended)
  - `expected_headers=[]` handling of duplicate headers
  - padding short rows and dropping trailing empty rows
- Existing tests for the readers should pin this down. Add tests for the edge
  cases above before changing the code.

Saves 1 read per tab: **2 per selection**.

## Option D: open by title without building a `Spreadsheet` twice

`info` calls `client.open(title)`, which searches Drive and then builds a
`Spreadsheet` (1 Sheets read), only to pass `found.id` to `open_gsheet()`,
which builds another.

- Use `client.list_spreadsheet_files(title)` (Drive only) to find the id,
  raising `SpreadsheetNotFoundError` when it returns nothing, then call
  `open_gsheet()` as now.
- Check how `client.open()` picks between several files with the same title,
  and keep the same choice.

Saves **1 read** when opening by title.

As implemented: `client.open()` takes the first file in the Drive results whose
`name` is exactly the title (the Drive search is limited to native Sheets), and
`info` now does the same with the `list_spreadsheet_files()` results.

## Option E: fetch all input tabs in one request

Use `spreadsheet.values_batch_get([features_range, people_range, already_selected_range])`
to get every input tab in one request, then hand each reader its slice.

- Brings a whole selection's reads down to about 3 (open, metadata, batch),
  from about 5 after B and C.
- Needs a restructure: today each `read_*_data` context manager fetches its
  own tab when it is entered, and they are called separately by the selection
  code. The data source would need a "prefetch" step, or to fetch everything
  on the first read and cache the result.
- Only worth it after B and C, and only if B and C prove not to be enough.

## Option F: fewer writes

Writes have their own 60 per minute quota, so these matter less, but they are
cheap to do:

- `delete_old_output_tabs` sends one `del_worksheet` (a `batch_update`) per
  tab. Send one `spreadsheet.batch_update` with a `deleteSheet` request per tab.
- `highlight_dupes` looks up the remaining tab again. Keep the `Worksheet`
  created in `write_remaining` and reuse it. (Option B also removes the read.)
- `write_selected` and `write_remaining` each make 3 writes (`add_worksheet`,
  `update`, `format`). These could be combined into one `batch_update`, but
  that means writing raw API requests instead of using gspread helpers. Not
  recommended.

As implemented: only the batched delete. It is atomic, so if one tab can't be
deleted (for example, someone deleted it by hand after the tab list was
fetched) none are, where before the tabs ahead of it in the list were deleted.
`highlight_dupes` is left alone: after Option B, `_get_tab()` finds the
remaining tab in the cache, which `write_remaining` added it to, so keeping the
`Worksheet` would save nothing.

## Outside this library

The service could avoid repeat work itself, for example by not re-reading a
sheet it checked seconds earlier, and by spacing out selections. Retries count
against the quota too: `BackOffHTTPClient` retries on 429, and each retry is
another request.

## Summary

| Option | Change                             | Reads saved                    | Effort | Risk                         |
| ------ | ---------------------------------- | ------------------------------ | ------ | ---------------------------- |
| A      | `inspect_gsheet()`                 | 1 per check                    | Small  | Low                          |
| B      | Cache tab list per open            | ~7 per selection               | Medium | Low (refreshed when stale)   |
| C      | One `get()` per tab                | 2 per selection                | Small  | Medium (parsing equivalence) |
| D      | Drive-only title lookup            | 1 per open by title            | Small  | Low                          |
| E      | One batch read for all tabs        | 2 more per selection after B+C | Large  | Medium                       |
| F      | Batch tab deletes, reuse worksheet | Writes only                    | Small  | Low                          |

Suggested order: A as its own PR (it is what the check needs). Then B + D in a
second PR, and C in a third, as C is the one most likely to change behaviour
subtly. Revisit E only if real usage still hits the limit.
