import duckdb
import pandas as pd

from interface import COCOAResult, ColumnScore, run_cocoa_experiment
from src.DataAugmentation import create_index

TABLES = {"dt": "distinct_tokens", "mt": "main_tokenized", "mc": "max_column", "oi": "order_index"}

def build_demo_db():
    """
    Creates and populates an in-memory DuckDB database with predefined tables and data.

    This function sets up an in-memory database using DuckDB with tables such as
    `distinct_tokens`, `main_tokenized`, `max_column`, and `order_index`. These tables
    are populated with predefined data to simulate a demo environment for testing or
    development purposes. It also computes indexing information using the `create_index`
    function and stores the results in the `order_index` table.

    Returns:
        duckdb.DuckDBPyConnection: A connection object to the in-memory DuckDB database.

    Raises:
        Exception: If the database operations or index creation encounters an error.

    Note:
        This function assumes the existence of a helper function named `create_index`
        which is expected to generate index-related data (minimum index, ordering, and
        binary encoding lists).
    """
    # One external table (id 1): col 0 = apple/banana/cherry, col 1 = 10/20/30.
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

def test_run_cocoa_experiment_with_injected_connection():
    """
    Tests the `run_cocoa_experiment` function with an injected database connection.

    This test case initializes a demonstration database connection, constructs a sample
    dataframe, and executes the `run_cocoa_experiment` function with specified parameters.
    The test ensures that the injected connection remains open after the function call
    and verifies the correctness of the result, including its data and metadata content.

    This test also reconstructs certain aspects of the correlation results to validate
    that the expected column scores are returned.

    Raises:
        AssertionError: If any of the assertions related to the function's output or
        side effects fail.
    """
    conn = build_demo_db()

    data = pd.DataFrame({
        "query_column": ["Apple", "Banana"],
        "target_column": [5, 15],
    })

    result = run_cocoa_experiment(
        data,
        k_c=1,
        k_t=1,
        query_column="query_column",
        target_column="target_column",
        db_config={"tables": TABLES},
        conn=conn,
    )

    assert conn.execute("SELECT 1").fetchone()[0] == 1
    conn.close()

    assert isinstance(result, COCOAResult)
    assert result.k_c == 1
    assert result.k_t == 1
    assert list(result.data["1_1"]) == ["10", "20"]

    assert result.selected_columns == [
        ColumnScore(table_col_id="1_1", correlation=None, is_numeric=None)
    ]

def test_run_cocoa_experiment_opens_and_closes_its_own_connection(tmp_path):
    """
    Tests the `run_cocoa_experiment` function to ensure it correctly establishes
    and terminates its own database connection, processes input data, and returns
    accurate experiment results with no column overlap in an empty index.

    Args:
        tmp_path (Path): Temporary file path used to create and manage test-specific
            database files during the execution of the test.
    """
    db_path = tmp_path / "demo.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE distinct_tokens (tokenized VARCHAR, table_col_id VARCHAR)")
    conn.execute("CREATE TABLE main_tokenized (table_col_id VARCHAR, tokenized VARCHAR, rowid INTEGER)")
    conn.execute("CREATE TABLE max_column (tableid INTEGER, max_colid INTEGER)")
    conn.execute(
        "CREATE TABLE order_index "
        "(table_col_id VARCHAR, is_numeric BOOLEAN, min_index INTEGER, order_list VARCHAR, binary_list VARCHAR)"
    )
    conn.close()

    data = pd.DataFrame({"query_column": ["Apple"], "target_column": [5]})

    result = run_cocoa_experiment(
        data,
        k_c=1,
        k_t=1,
        query_column="query_column",
        target_column="target_column",
        db_config={"connection": {"demo": {"database": str(db_path)}}, "tables": TABLES},
        db_profile="demo",
    )

    # No overlap in this empty index, so enrich() returns the input unchanged.
    assert result.selected_columns == []
    assert "query_column" in result.data.columns
