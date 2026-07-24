"""
Builds the 4 COCOA index tables from the real CSVs in dataset/ (writes to
db/cocoa_real.duckdb, separate from the demo DB).

Run with : uv run python -m examples.build_index

Then:
    uv run python run_cocoa.py --input dataset/presidential.csv --output <somewhere>/enriched.csv \
        --query-column County --target-column Votes \
        --k-c 5 --k-t 20 --db-config config/cocoa_duckdb_config.json --db-profile real
"""
import glob
import argparse
import json
import os
import duckdb
import pandas as pd

from src.DataAugmentation import create_index, get_cleaned_text

# Decided not to touch DataAugmentation.py thus redeclared here.
def tokenize_cell(value):
    """
    Tokenizes and processes the input cell value by converting it to a lowercase
    string and cleaning it.
    Tokenizes exactly like enrich() does for query values (str + lower +
    get_cleaned_text, src/DataAugmentation.py).

    Args:
        value: The input cell data to be tokenized and processed.

    Returns:
        str: The processed and cleaned text as a string.
    """
    return get_cleaned_text(str(value).lower())

# Decided not to touch DataAugmentation.py thus redeclared here.
def _is_numeric(s):
    """
    Determines if the given string represents a numeric value. A string is
    considered numeric if it can be successfully converted to a float or if
    it is the string 'nan' (case insensitive).

    Args:
        s (str): The input string to be evaluated for numeric content.

    Returns:
        bool: True if the string is numeric or represents 'nan',
        False otherwise.
    """
    if s.lower() == 'nan':
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False

# Decided not to touch DataAugmentation.py thus redeclared here.
def _is_numeric_list(values):
    """
    Checks if all elements in a list are numeric.

    This function verifies whether all elements in the given list are numeric values.
    It handles cases where elements are `None` or empty strings by treating them as 'nan',
    as instructed in the original README.

    Args:
        values (list): A list of elements to check for numeric validity.

    Returns:
        bool: True if all elements in the list are numeric, False otherwise.
    """
    return all(_is_numeric('nan' if (v is None or v == '') else str(v)) for v in values)

def melt_dataframe(df):
    """
    Converts a wide-format DataFrame into a long-format DataFrame, adding
    a row identifier and column positions as metadata.

    This function reshapes the input DataFrame by unpivoting its columns into
    rows, effectively transforming the data from a wide format to a long format.
    The resulting DataFrame includes additional metadata columns:
    `rowid` for the row index and `colid` for the column index in the original
    DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame in wide format.

    Returns:
        pd.DataFrame: A DataFrame in long format with additional metadata
        columns `rowid` and `colid`.
    """
    col_position = {col: i for i, col in enumerate(df.columns)}
    long_df = df.reset_index(names="rowid").melt(id_vars="rowid", var_name="colname", value_name="value")
    long_df["colid"] = long_df["colname"].map(col_position)
    return long_df

def build_main_tokenized(long_df, tableid):
    """
    Processes the input DataFrame to tokenize cell values, assign a table identifier,
    and combine table and column identifiers into a unique field. Returns a transformed
    DataFrame containing the processed data.

    Args:
        long_df (pandas.DataFrame): DataFrame containing the column 'value' that holds
            cell values to be tokenized, along with other relevant information such as
            table, column, and row identifiers.
        tableid (Any): Identifier representing the table associated with the input
            DataFrame.

    Returns:
        pandas.DataFrame: Transformed DataFrame with the following columns:
            - 'tokenized': Tokenized representation of the 'value' column.
            - 'tableid': The provided table identifier assigned to each row.
            - 'rowid': The row identifier carried over from the input DataFrame.
            - 'table_col_id': Concatenated identifier combining table and column IDs.
    """
    rows = long_df.dropna(subset=["value"]).copy()
    rows["tokenized"] = rows["value"].apply(tokenize_cell)
    rows["tableid"] = tableid
    rows["table_col_id"] = rows["tableid"].astype(str) + "_" + rows["colid"].astype(str)
    return rows[["tokenized", "tableid", "rowid", "table_col_id"]]

def build_order_index_rows(df, tableid):
    """
    Builds a DataFrame of order index rows for columns in a given DataFrame.

    This function processes the given DataFrame by iterating through its columns
    and generating metadata and index information for each column. Each column's
    details are stored as a dictionary, which includes its numeric status, minimum
    index, order list, and binary list. The collection of these dictionaries is
    finally assembled into a Pandas DataFrame.

    NaN cells dropped (unlike order_index,
    which needs the full column) as per instructions in original README.md.

    Args:
        df: A Pandas DataFrame of input data for which index rows are built.
        tableid: A string identifier for the table to uniquely identify columns.

    Returns:
        pd.DataFrame: A DataFrame containing order index rows with details about
        each column, such as numeric status, minimum index, order list, and binary
        list.
    """
    rows = []
    for colid, colname in enumerate(df.columns):
        column = df[colname]

        is_numeric = _is_numeric_list(column.tolist())

        min_index, order_list, binary_list = create_index(column.tolist())

        rows.append({
            "table_col_id": f"{tableid}_{colid}",
            "is_numeric": is_numeric,
            "min_index": int(min_index),
            "order_list": ",".join(str(x) for x in order_list),
            "binary_list": ",".join(str(x) for x in binary_list),
        })
    return pd.DataFrame(rows)

def main(argv=None):
    """
    Main function for creating and populating tables in a DuckDB database from a set of CSV files.

    This function performs the following operations:
    1. Reads a configuration file to determine the database path and table names.
    2. Ensures the target directory for the database exists and deletes any pre-existing database file.
    3. Creates and initializes tables in the DuckDB database.
    4. Processes CSV files, transforming their content and populating the respective tables.
    5. Registers and inserts processed data into the corresponding database tables.
    6. Outputs the row count for each table and confirms the database index creation.

    Raises:
        Any exceptions related to file I/O, database operations, or pandas operations will cause the
        function to terminate prematurely.

    Args:
        None

    Returns:
        None
    """
    # Parser for stating table corpora directory.
    parser = argparse.ArgumentParser(description="Run COCOA indexing.")
    parser.add_argument("--corpora", required=False,
                        help="Directory containing the table corpora. Defaults to dataset/.")
    # Yet to be implemented limit functionality to limit the number of processed csv's
    parser.add_argument("--limit", required=False, help="limit the number of csv's to process for testing purposes")
    args = parser.parse_args(argv)

    with open("config/cocoa_duckdb_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    db_path = config["connection"]["real"]["database"]
    tables = config["tables"]

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    if os.path.exists(db_path):
        os.remove(db_path)

    # Initialize tables and connection to db.
    conn = duckdb.connect(db_path)
    conn.execute(
        f"CREATE TABLE {tables['mt']} (tokenized TEXT, tableid INT NOT NULL, rowid INT NOT NULL, table_col_id TEXT NOT NULL)")
    conn.execute(f"CREATE TABLE {tables['dt']} (tokenized TEXT, table_col_id TEXT NOT NULL)")
    conn.execute(
        f"CREATE TABLE {tables['oi']} (table_col_id TEXT NOT NULL, is_numeric BOOLEAN, min_index INT NOT NULL, order_list TEXT, binary_list TEXT)")
    conn.execute(f"CREATE TABLE {tables['mc']} (tableid INT NOT NULL, max_colid INT NOT NULL, PRIMARY KEY (tableid))")

    # main_tokenized_parts = []
    # order_index_parts = []

    # Obtain all csv paths. If --corpora is specified, use that, else use dataset/. Possible error source: empty csv's.
    if args.corpora is not None:
        # print(f"Using corpora directory {args.corpora}").
        csv_paths = sorted(os.path.join(root, file)
                           for root, dirs, files in os.walk(args.corpora)
                           for file in files if
                           file.endswith(".csv"))
    else:
        csv_paths = sorted(
            glob.glob(
                os.path.join("dataset", "*.csv")))

    for tableid, path in enumerate(csv_paths, start=1):
        filename = os.path.basename(path)
        df = pd.read_csv(path)
        long_df = melt_dataframe(df)

        # main_tokenized_parts.append(build_main_tokenized(long_df, tableid))
        tmp_tokenized_df = build_main_tokenized(long_df, tableid)
        conn.register("tmp_tokenized", tmp_tokenized_df)
        conn.execute(f"INSERT INTO {tables['mt']} SELECT * FROM tmp_tokenized")

        # order_index_parts.append(build_order_index_rows(df, tableid))
        tmp_order_index_df = build_order_index_rows(df, tableid)
        conn.register("tmp_order_index", tmp_order_index_df)
        conn.execute(f"INSERT INTO {tables['oi']} SELECT * FROM tmp_order_index")

        conn.unregister("tmp_tokenized")
        conn.unregister("tmp_order_index")

    # # Union all parts into one big DF.
    # all_main_tokenized = pd.concat(main_tokenized_parts, ignore_index=True)
    # all_order_index = pd.concat(order_index_parts, ignore_index=True)
    #
    # # Fill db tables.
    # conn.register("main_tokenized_df", all_main_tokenized)
    # conn.execute(f"INSERT INTO {tables['mt']} SELECT * FROM main_tokenized_df")
    # conn.register("order_index_df", all_order_index)
    # conn.execute(f"INSERT INTO {tables['oi']} SELECT * FROM order_index_df")
    conn.execute(f"INSERT INTO {tables['dt']} SELECT DISTINCT tokenized, table_col_id FROM {tables['mt']}")

    # max_colid per table, derived from order_index (one row per column, incl.
    # all-NaN columns) unlike main_tokenized (NaN cells dropped there).
    # table_col_id is "{tableid}_{colid}" -> split back into the two parts and
    # take the highest colid per tableid.
    conn.execute(f"""
        INSERT INTO {tables['mc']}
        SELECT
            CAST(
                split_part(table_col_id, '_', 1) AS INTEGER) AS tableid,
                MAX(CAST(split_part(table_col_id, '_', 2) AS INTEGER)
                )
            AS max_colid
        FROM {tables['oi']}
        GROUP BY 1
    """)

    for t in tables.values():
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t}: {n} rows")

    conn.close()
    print(f"Real index built at {db_path}")

if __name__ == "__main__":
    main()
