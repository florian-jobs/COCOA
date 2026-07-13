# Full pipeline test: builds a tiny in-memory DB with the 4 COCOA tables,
# then checks COCOAHandler.enrich() output.
import duckdb
import pandas as pd

from src.DataAugmentation import COCOAHandler, create_index

def build_demo_db():
    """
    Builds an in-memory demonstration database using DuckDB.

    This function creates an example database structure in DuckDB, containing
    several tables with predefined values. The database simulates relationships
    between tokenized columns, rows, and indexed data for testing and demonstration
    purposes. It also calculates and inserts an index into the `order_index` table
    using the `create_index` function for numeric values.

    Returns:
        duckdb.DuckDBPyConnection: A connection to the in-memory DuckDB database.

    Raises:
        Exception: This function may raise exceptions if any SQL query fails or if
            `create_index` encounters issues.

    Tables Created:
        - distinct_tokens: Contains tokenized strings and their corresponding table
          column IDs.
        - main_tokenized: Maps table column IDs and tokenized strings to their
          respective row IDs.
        - max_column: Stores the maximum column ID for each table.
        - order_index: Stores index-related metadata, including minimum numeric
          value indices, order lists, and binary indices.

    Dependencies:
        - DuckDB: The function requires DuckDB installed.
        - create_index: An external function used to calculate indices for certain
          values.

    Notes:
        This function operates in-memory, meaning the database will not persist
        after the function execution ends. It is intended solely for demonstration
        and testing purposes.

    Warning:
        Ensure that the `create_index` function is defined and operational before
        using this function, as it directly depends on its output.
    """
    conn = duckdb.connect(":memory:")

    conn.execute("CREATE TABLE distinct_tokens (tokenized VARCHAR, table_col_id VARCHAR)")
    conn.execute(
        "INSERT INTO distinct_tokens VALUES ('apple', '1_0'), ('banana', '1_0'), ('cherry', '1_0')"
    )

    conn.execute("CREATE TABLE main_tokenized (table_col_id VARCHAR, tokenized VARCHAR, rowid INTEGER)")
    conn.execute(
        """
        INSERT INTO main_tokenized
        VALUES ('1_0', 'apple', 0),
               ('1_0', 'banana', 1),
               ('1_0', 'cherry', 2),
               ('1_1', '10', 0),
               ('1_1', '20', 1),
               ('1_1', '30', 2)
        """
    )

    conn.execute("CREATE TABLE max_column (tableid INTEGER, max_colid INTEGER)")
    conn.execute("INSERT INTO max_column VALUES (1, 1)")

    min_index, order_list, binary_list = create_index(["10", "20", "30"])
    conn.execute(
        "CREATE TABLE order_index "
        "(table_col_id VARCHAR, is_numeric BOOLEAN, min_index INTEGER, order_list VARCHAR, binary_list VARCHAR)"
    )
    conn.execute(
        "INSERT INTO order_index VALUES (?, ?, ?, ?, ?)",
        [
            "1_1",
            True,
            int(min_index),
            ",".join(str(x) for x in order_list),
            ",".join(str(x) for x in binary_list),
        ],
    )
    return conn

def test_enrich_joins_matching_external_column():
    """
    Tests the `enrich` method of the `COCOAHandler` class to validate joins matching an external column.

    This test ensures that the `enrich` method correctly maps data using the specified query and
    target columns, and verifies that only matching rows based on the join criteria are included
    in the result. Non-matching rows are expected to be ignored.

    Args:
        None

    Raises:
        AssertionError: If the resulting DataFrame does not contain the expected column or if
            the values in the resulting column (`1_1`) do not match the expected list of
            strings ["10", "20"] as intended.

    """
    conn = build_demo_db()
    tables = {"dt": "distinct_tokens", "mt": "main_tokenized", "mc": "max_column", "oi": "order_index"}

    # Matches apple/banana; cherry left out on purpose to check non-matches are ignored.
    data = pd.DataFrame({
        "query_column": ["Apple", "Banana"],
        "target_column": [5, 15],
    })

    cocoa = COCOAHandler(conn, tables)
    result = cocoa.enrich(data, k_c=1, k_t=1, query_column="query_column", target_column="target_column")
    conn.close()

    assert "1_1" in result.columns
    assert list(result["1_1"]) == ["10", "20"]
