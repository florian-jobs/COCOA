"""Tests for the offline COCOA index builder.

The tests are intentionally split into two levels:

1. Small unit tests for the individual transformation functions.
2. Integration tests that use random samples from every dataset CSV, inspect
   the four tables produced by ``build_real_index.main()``, and audit the
   existing full real index.

The integration tests use pytest's temporary directory. Therefore, they never
delete or overwrite ``db/cocoa_real.duckdb`` from the actual project.
"""
import json
from pathlib import Path
import zlib
import duckdb
import pandas as pd
import pytest
from examples import build_real_index
from examples.build_real_index import (
    build_main_tokenized,
    build_order_index_rows,
    melt_dataframe,
    tokenize_cell,
)
from src.DataAugmentation import get_cleaned_text

DATASET_DIR = Path("dataset")
CSV_FILES = sorted(DATASET_DIR.glob("*.csv"))
REAL_CONFIG = Path("config/cocoa_duckdb_config.json")

RANDOM_SEED = 42
SAMPLE_FRACTION = 0.10
MIN_SAMPLE_ROWS = 20
MAX_SAMPLE_ROWS = 200

TABLES = {
    "dt": "distinct_tokens",
    "mt": "main_tokenized",
    "mc": "max_column",
    "oi": "order_index",
}

# When no CSV exists, pytest reports an explicit skipped case instead of
# silently collecting zero parameterized tests.
CSV_CASES = CSV_FILES or [
    pytest.param(
        None,
        marks=pytest.mark.skip(reason="No dataset/*.csv files were found"),
        id="no-csv-files",
    )
]

def _read_random_sample(csv_path, *, reset_index=True):
    """
    Reads a random sample of rows from a CSV file, ensuring practical bounds on the
    sample size. This function utilizes a consistent random seed based on the file
    name to ensure deterministic sampling for the same inputs.

    Args:
        csv_path (str): Path to the CSV file to read the data from.
        reset_index (bool, optional): Whether to reset the index of the sampled data.
            Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing the randomly sampled rows from
        the CSV file.
    """
    source = pd.read_csv(csv_path)

    if source.empty:
        pytest.fail(f"CSV file contains no data rows: {csv_path}")

    # Use 10% of the rows, with practical lower and upper bounds.
    fraction_rows = round(len(source) * SAMPLE_FRACTION)
    sample_size = min(
        len(source),
        max(MIN_SAMPLE_ROWS, fraction_rows),
        MAX_SAMPLE_ROWS,
    )

    file_seed = zlib.crc32(csv_path.name.encode("utf-8"))

    sample = source.sample(
        n=sample_size,
        random_state=file_seed,
    )

    if reset_index:
        return sample.reset_index(drop=True)

    return sample

@pytest.fixture(
    scope="module",
    params=CSV_CASES,
    ids=lambda path: path.name if path is not None else "no-csv-files",
)
def csv_sample(request):
    csv_path = request.param
    return csv_path, _read_random_sample(csv_path)

def _reference_main_tokenized(source, tableid):
    """
    Generates a list of tokenized cell data from the provided DataFrame, correlating
    each value to its table and column identifiers.

    This function processes all non-NaN entries from the input DataFrame `source`,
    applies tokenization to each cell value, and associates it with a table ID, row
    ID, and a constructed identifier combining the table ID and column ID.

    Args:
        source (pd.DataFrame): Input pandas DataFrame containing the data to be
            tokenized and processed.
        tableid (str): Unique identifier for the table being processed.

    Returns:
        list[tuple]: A list of tuples, where each tuple represents a tokenized cell
        value along with its associated table ID, row ID (as an integer), and a
        combined identifier for the column.
    """
    expected = []

    for colid, colname in enumerate(source.columns):
        for rowid, value in source[colname].items():
            if pd.isna(value):
                continue

            expected.append(
                (
                    tokenize_cell(value),
                    tableid,
                    int(rowid),
                    f"{tableid}_{colid}",
                )
            )

    return expected

def _build_temporary_index(tmp_path, monkeypatch, sources):
    """
    Builds a temporary index in a testing environment.

    This function facilitates the creation of a temporary database index setup by
    utilizing provided source data and configuration. It creates a temporary
    directory structure with dataset, configuration, and database directories, and
    writes the necessary files required for the indexing process. The function
    mimics the real indexing procedure by calling the `build_real_index.main()`
    function, ensuring the database is constructed as expected.

    Args:
        tmp_path (Path): The base temporary directory path for creating the
            dataset, configuration, and database directories.
        monkeypatch (pytest.MonkeyPatch): A monkeypatch object used to temporarily
            change the working directory for testing purposes.
        sources (List[Tuple[str, DataFrame]]): A list of tuples where each tuple
            contains a filename and its corresponding source data as a DataFrame.

    Returns:
        Path: The path to the created test index database file.
    """
    dataset_dir = tmp_path / "dataset"
    config_dir = tmp_path / "config"
    db_dir = tmp_path / "db"

    dataset_dir.mkdir()
    config_dir.mkdir()
    db_dir.mkdir()

    for filename, source in sources:
        source.to_csv(dataset_dir / filename, index=False)

    config = {
        "connection": {"real": {"database": "db/test_index.duckdb"}},
        "tables": TABLES,
    }
    (config_dir / "cocoa_duckdb_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    build_real_index.main(argv=[])

    return db_dir / "test_index.duckdb"

# tokenize_cell() test.
def test_tokenize_cell_matches_get_cleaned_text_on_query_side():
    """
    Tests the consistency of cleaned text and tokenized cell values for query-side inputs.

    Tokenize_cell() must do exactly what enrich() does for query values
    (str + lower + get_cleaned_text) - including NOT stripping the ".0" pandas
    adds to whole numbers - or the SQL overlap join silently stops matching.

    Raises:
        AssertionError: If the tokenized cell value does not match the cleaned text
            representation for the given input.
    """
    assert tokenize_cell("2009.0") == get_cleaned_text("2009.0")

def test_tokenize_cell_leaves_real_decimals_alone():
    assert tokenize_cell("3.14") == "3 14"

def test_tokenize_cell_lowercases_like_the_query_side_does():
    assert tokenize_cell("New York") == "new york"

# melt_dataframe() test.
def test_melt_dataframe_matches_reference_for_random_csv_sample(csv_sample):
    """
    Tests whether the `melt_dataframe` function produces the expected output for a
    randomly generated CSV sample. The test implementation constructs the
    expected rows manually without relying on `pandas.melt()` to avoid
    duplication of the tested logic.

    Args:
        csv_sample (Tuple[Any, pandas.DataFrame]): A tuple containing a placeholder
            element and a pandas DataFrame retrieved from a randomly generated
            CSV sample. The placeholder element is not used in the test logic.
    """
    _, source = csv_sample
    actual = melt_dataframe(source).reset_index(drop=True)

    # Construct the expected rows with ordinary loops instead of pandas.melt().
    expected_rows = []
    for colid, colname in enumerate(source.columns):
        for rowid, value in enumerate(source[colname].tolist()):
            expected_rows.append((rowid, colname, value, colid))

    expected = pd.DataFrame(
        expected_rows,
        columns=["rowid", "colname", "value", "colid"],
    )

    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

def test_melt_dataframe_preserves_null_cells():
    """
    Tests that the `melt_dataframe` function preserves the null cells from the source
    DataFrame in the resulting DataFrame after the melt operation.

    Raises:
        AssertionError: If the number of rows in the resulting DataFrame does not match
        the expected value calculated as the product of the number of rows and columns
        in the source DataFrame.

        AssertionError: If the count of null values in the 'value' column of the resulting
        DataFrame does not match the total count of null values in the source DataFrame.
    """
    source = pd.DataFrame(
        {
            "City": ["Berlin", None, "Paris"],
            "Year": [2009.0, 2010.0, None],
        }
    )

    actual = melt_dataframe(source)

    assert len(actual) == source.shape[0] * source.shape[1]
    assert actual["value"].isna().sum() == source.isna().sum().sum()

def test_melt_dataframe_assigns_positional_column_ids():
    """
    Tests the `melt_dataframe` function to verify that it correctly assigns unique
    positional column IDs for each column in the given DataFrame.

    This test ensures that the output of the `melt_dataframe` function contains a
    mapping of the original column names to their corresponding positional IDs,
    sorted and validated against expected outcomes.

    Args:
        None

    Raises:
        AssertionError: If the output of the `melt_dataframe` function does not
        match the expected mapping of column names to positional IDs.
    """
    source = pd.DataFrame(
        {
            "City": ["Berlin"],
            "Year": [2010.0],
            "Score": [7.5],
            "Country": ["Germany"],
        }
    )

    actual = melt_dataframe(source)
    mapping = actual[["colname", "colid"]].drop_duplicates()

    assert mapping.to_records(index=False).tolist() == [
        ("City", 0),
        ("Year", 1),
        ("Score", 2),
        ("Country", 3),
    ]

# build_main_tokenized() tests.
def test_build_main_tokenized_matches_reference_for_random_csv_sample(csv_sample):
    """
    Verifies that the `build_main_tokenized` function produces the expected results
    for a given random CSV sample by comparing its output to a reference
    implementation.

    Args:
        csv_sample: Tuple containing a random CSV sample consisting of two elements:
            a placeholder (ignored) and a pandas DataFrame representing the source
            data.
    """
    _, source = csv_sample
    tableid = 7
    long_df = melt_dataframe(source)

    actual = build_main_tokenized(long_df, tableid)
    actual_rows = list(actual.itertuples(index=False, name=None))
    expected_rows = _reference_main_tokenized(source, tableid)

    assert actual_rows == expected_rows

def test_build_main_tokenized_drops_nulls_but_retains_duplicates():
    """
    Tests the `build_main_tokenized` function to ensure it drops null values but retains duplicate
    values, while preserving original row IDs.

    This test verifies the following:
    1. The function removes null cells from the input DataFrame without retaining them in the output.
    2. Duplicate values, such as multiple values for "Berlin," remain intact while retaining their
       corresponding original row IDs.
    3. The tokenization process ensures no null values exist in the "tokenized" column of the
       output DataFrame.

    Args:
        None

    Raises:
        AssertionError: If the output DataFrame does not meet the expected conditions, such as
        incorrect cell count, residual null cells in the "tokenized" column, or improper handling
        of duplicate values.
    """
    source = pd.DataFrame(
        {
            "City": ["Berlin", "Berlin", None],
            "Year": [2010.0, 2010.0, 2011.0],
        }
    )

    actual = build_main_tokenized(melt_dataframe(source), tableid=3)

    # There are six source cells and one null cell.
    assert len(actual) == 5
    assert actual["tokenized"].isna().sum() == 0

    # Both Berlin cells must still exist and keep their original row IDs.
    berlin_rows = actual[
        (actual["table_col_id"] == "3_0")
        & (actual["tokenized"] == "berlin")
        ]
    assert berlin_rows["rowid"].tolist() == [0, 1]

def test_build_main_tokenized_assigns_requested_tableid():
    source = pd.DataFrame({"City": ["Berlin"], "Year": [2010.0]})

    actual = build_main_tokenized(melt_dataframe(source), tableid=42)

    assert set(actual["tableid"]) == {42}
    assert set(actual["table_col_id"]) == {"42_0", "42_1"}

# build_order_index() test.
def test_build_order_index_creates_one_row_per_source_column():
    """
    Tests that the function `build_order_index_rows` creates one row in the resulting
    data structure for each column in the source DataFrame. This test verifies that
    the function correctly associates metadata to source columns, ensuring proper
    column mapping and metadata assignment for numerical and non-numerical entries.

    Args:
        None

    Raises:
        AssertionError: If the generated rows do not match the expected format, or if
            the serialized lists (`order_list`, `binary_list`, etc.) do not contain an
            entry for every source row in the DataFrame. Will also raise
            AssertionError if the `min_index` is invalid for the source data.

    Notes:
        - The `table_col_id` column in the resulting data structure is tested to
          confirm it contains unique IDs derived from the provided table ID and
          column index positions.
        - The `is_numeric` attribute is validated to ensure its correctness based on
          the type of source column data.
        - Serialization of `order_list` and `binary_list` is confirmed to have entries
          corresponding to each row in the input DataFrame.
        - Proper range bounds of `min_index` are checked to fall within the source
          DataFrame row count.
    """
    source = pd.DataFrame(
        {
            "City": ["Paris", "Berlin", "Tokyo", "Berlin"],
            "Year": [2012, 2010, 2013, 2010],
            "Score": [8.5, 7.0, 9.2, 7.0],
            "Country": ["France", "Germany", "Japan", "Germany"],
        }
    )

    actual = build_order_index_rows(source, tableid=5)

    assert actual["table_col_id"].tolist() == ["5_0", "5_1", "5_2", "5_3"]
    assert actual["is_numeric"].tolist() == [False, True, True, False]

    # Every serialized list must contain one entry for every source row.
    for row in actual.itertuples(index=False):
        assert len(row.order_list.split(",")) == len(source)
        assert len(row.binary_list.split(",")) == len(source)
        assert 0 <= row.min_index < len(source)

def test_build_order_index_encodes_numeric_values_in_sorted_order():
    """
    Tests the encoding of numeric values into a sorted order index.

    The function validates that the `order_index` output of the `build_order_index_rows`
    function correctly encodes numeric values in a sorted order. It ensures that the
    order list establishes a valid traversal order, that no rows are revisited, and that
    the traversal ends correctly at the maximum element. It further checks that all rows
    are covered and that the traversal reproduces the sorted order of the original values.

    Raises:
        AssertionError: If the `order_index` contains a cycle, the traversal does not cover
            all rows, or the sorted output does not match the source's sorted values.
    """
    source = pd.DataFrame({"Value": [30, 10, 20, 20]})
    index_row = build_order_index_rows(source, tableid=8).iloc[0]

    next_row = [int(value) for value in index_row["order_list"].split(",")]
    current_row = int(index_row["min_index"])
    visited_rows = []

    # Starting at min_index, each order_list entry points to the next row in
    # sorted order. The maximum element points to -1 and ends the traversal.
    while current_row != -1:
        assert current_row not in visited_rows, "order_index contains a cycle"
        visited_rows.append(current_row)
        current_row = next_row[current_row]

    actual_sorted_values = source.loc[visited_rows, "Value"].tolist()

    assert len(visited_rows) == len(source)
    assert set(visited_rows) == set(range(len(source)))
    assert actual_sorted_values == sorted(source["Value"].tolist())

# offline-phase tests.
def test_offline_builder_matches_ground_truth_for_all_csv_samples(
        tmp_path,
        monkeypatch,
):
    """
    Tests the offline builder functionality and verifies it matches the ground truth
    result for all available CSV samples. The function ensures the integrity of the
    process through multiple validations, including testing intermediate results
    at different steps of the process.

    Args:
        tmp_path: Temporary directory path used for testing data creation and storage.
        monkeypatch: pytest fixture used for dynamic modification of objects or
            environments during testing.
    """
    if not CSV_FILES:
        pytest.skip("No dataset/*.csv files were found")

    # main() sorts filenames before assigning table IDs. Keep the same explicit
    # order in the reference result so tableid 1, 2, 3, ... match correctly.
    sampled_sources = [
        (csv_path.name, _read_random_sample(csv_path))
        for csv_path in CSV_FILES
    ]

    db_path = _build_temporary_index(
        tmp_path,
        monkeypatch,
        sampled_sources,
    )

    expected_main = set()
    expected_order_columns = set()
    expected_max_columns = []

    # Read the temporary CSVs back before calculating ground truth. This uses
    # exactly the serialized values that main() consumed and avoids float
    # round-trip differences between an in-memory sample and its CSV form.
    for tableid, (filename, _) in enumerate(sampled_sources, start=1):
        source = pd.read_csv(tmp_path / "dataset" / filename)
        expected_main.update(_reference_main_tokenized(source, tableid))
        expected_order_columns.update(
            f"{tableid}_{colid}" for colid in range(len(source.columns))
        )
        expected_max_columns.append((tableid, len(source.columns) - 1))

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        # 1. main_tokenized: one row for each non-null source cell.
        actual_main = set(
            conn.execute(
                """
                SELECT tokenized, tableid, rowid, table_col_id
                FROM main_tokenized
                """
            ).fetchall()
        )
        assert actual_main == expected_main

        # 2. distinct_tokens: unique (token, source-column) pairs.
        actual_distinct = set(
            conn.execute(
                "SELECT tokenized, table_col_id FROM distinct_tokens"
            ).fetchall()
        )
        expected_distinct = {
            (tokenized, table_col_id)
            for tokenized, _, _, table_col_id in expected_main
        }
        assert actual_distinct == expected_distinct

        # 3. order_index: one index row for each source column.
        actual_order_columns = {
            row[0]
            for row in conn.execute(
                "SELECT table_col_id FROM order_index"
            ).fetchall()
        }
        assert actual_order_columns == expected_order_columns

        # 4. max_column: one correct zero-based maximum for every CSV table.
        assert conn.execute(
            "SELECT tableid, max_colid FROM max_column ORDER BY tableid"
        ).fetchall() == expected_max_columns
    finally:
        conn.close()

def test_offline_builder_keeps_an_all_null_final_column(
        tmp_path,
        monkeypatch,
):
    """
    Tests whether the offline builder retains a column with all null values in
    the final indexed results. It ensures that the process does not discard or
    misinterpret columns that contain only null values during transformation
    and storage.

    Args:
        tmp_path: A temporary path object used for creating a test-specific
            directory for file storage. Typically sourced from pytest fixtures.
        monkeypatch: A pytest fixture used to dynamically and temporarily
            modify or override functions, methods, or attributes for the
            test runtime.
    """
    source = pd.DataFrame(
        {
            "City": ["Berlin", "Paris", "Rome"],
            "Year": [2010.0, 2011.0, 2012.0],
            "AllNull": [None, None, None],
        }
    )

    db_path = _build_temporary_index(
        tmp_path,
        monkeypatch,
        [("all_null.csv", source)],
    )

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        # Three source columns mean that the last zero-based colid is 2.
        assert conn.execute(
            "SELECT tableid, max_colid FROM max_column"
        ).fetchall() == [(1, 2)]
    finally:
        conn.close()
