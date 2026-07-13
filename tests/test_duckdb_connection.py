# DuckDB setup sanity checks (see test_end_to_end.py for pipeline logic).
import json
import duckdb

def test_duckdb_connection():
    """
    Tests the DuckDB in-memory connection and query execution.

    This function establishes an in-memory connection to DuckDB, executes a simple
    SQL query to return a value, and validates that the result matches the expected
    output. After the validation, it closes the connection.

    Raises:
        AssertionError: If the query result does not match the expected value.
    """
    conn = duckdb.connect(":memory:")
    result = conn.execute("SELECT 1").fetchone()[0]
    assert result == 1
    conn.close()

def test_duckdb_config_has_expected_shape():
    """
    Tests whether the DuckDB configuration file matches the expected structure and contains
    all necessary keys for connection profiles and tables.

    This test ensures that the "cocoa_duckdb_config.json" file adheres to the required
    schema by verifying the presence of specific keys in the configuration. It checks the
    existence of "database" keys in different connection profiles and mandatory table keys.

    Raises:
        AssertionError: If any required keys are missing in the connection profiles or
        table definitions.
    """
    with open("config/cocoa_duckdb_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    for profile in ("demo", "real"):
        assert "database" in config["connection"][profile]
    for key in ("dt", "mt", "mc", "oi"):
        assert key in config["tables"]
