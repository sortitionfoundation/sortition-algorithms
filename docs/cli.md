# Command Line Interface

If you install `sortition_algorithms[cli]` you will be able to the CLI (command line interface):

```sh
$ sortition --help
Usage: sortition [OPTIONS] COMMAND [ARGS]...

  A command line tool to exercise the sortition algorithms.

Options:
  --help  Show this message and exit.

Commands:
  csv         Do sortition with CSV files.
  gen-sample  Generate a sample CSV file of people compatible with...
  gsheet      Do sortition with Google Spreadsheets.
```

This allows you to select a sample using the algorithms using data in CSV files or Google Spreadsheet.
It can also create a sample file based on a feature file.

## CSV Usage

Help text:

```sh
$ sortition csv --help
Usage: sortition csv [OPTIONS]

  Do sortition with CSV files.

Options:
  -S, --settings FILE             Settings for the sortition run. Will auto-
                                  create if not present.  [required]
  -f, --features-csv FILE         Path to CSV with features defined.
                                  [required]
  -p, --people-csv FILE           Path to CSV with people defined.  [required]
  -s, --selected-csv FILE         Path to CSV file to write selected people
                                  to.  [required]
  -r, --remaining-csv FILE        Path to CSV file to write remaining people
                                  to.  [required]
  -n, --number-wanted INTEGER RANGE
                                  Number of people to select.  [x>=1;
                                  required]
  --help                          Show this message and exit.
```

Example:

```sh
$ sortition csv \
  --number-wanted 22 \
  --settings path/to/sf_stratification_settings.toml \
  --features-csv tests/fixtures/features.csv \
  --people-csv tests/fixtures/candidates.csv \
  --selected-csv selected.csv \
  --remaining-csv remaining.csv
```

In the above example, the first 3 files must exist and will be read from. The last 2 will be written to.

## Google Spreadsheets Usage

Help text:

```sh
$ sortition gsheet --help
Usage: sortition gsheet [OPTIONS]

  Do sortition with Google Spreadsheets.

Options:
  -S, --settings FILE             Settings for the sortition run. Will auto-
                                  create if not present.  [required]
  --auth-json-file FILE           Path to file with OAuth2 details to access
                                  google account.  [required]
  --gen-rem-tab / --no-gen-rem-tab
                                  Generate a 'Remaining' tab.
  -g, --gsheet-name TEXT          Name of GDoc Spreadsheet to use.  [required]
  -f, --feature-tab-name TEXT     Name of tab containing features/categories.
                                  [required]
  -p, --people-tab-name TEXT      Name of tab containing people/respondents.
                                  [required]
  -s, --selected-tab-name TEXT    Name of tab to write selected people to.
                                  [required]
  -r, --remaining-tab-name TEXT   Name of tab to write remaining people to.
  -n, --number-wanted INTEGER RANGE
                                  Number of people to select.  [x>=1;
                                  required]
  --help                          Show this message and exit.
```

Example:

```sh
$ sortition gsheet \
  --number-wanted 52 \
  --auth-json-file path/to/secret_do_not_commit.json \
  --settings path/to/sf_stratification_settings.toml \
  --gsheet-name "Google Spreadsheet Name" \
  --feature-tab-name "Features" \
  --people-tab-name "Respondents" \
  --selected-tab-name "Original Selected" \
  --remaining-tab-name "Remaining - 1"
```

## Generating a sample file

Help text:

```sh
$ sortition gen-sample --help
Usage: sortition gen-sample [OPTIONS]

  Generate a sample CSV file of people compatible with features and settings.

Options:
  -S, --settings FILE             Settings for the sortition run. Will auto-
                                  create if not present.  [required]
  -f, --features-csv FILE         Path to CSV with features defined.
                                  [required]
  -p, --people-csv FILE           Path to CSV to write sample people to.
                                  [required]
  -n, --number-wanted INTEGER RANGE
                                  Number of people to write.  [x>=1; required]
  --help                          Show this message and exit.
```

Example:

```sh
uv run sortition gen-sample \
  --number-wanted 90 \
  --settings path/to/sf_stratification_settings.toml \
  --features-csv tests/fixtures/features.csv \
  --people-csv sample.csv
```
