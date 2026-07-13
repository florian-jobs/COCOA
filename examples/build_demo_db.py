"""
Builds a tiny demo DuckDB DB at the "demo" connection's path in
config/cocoa_duckdb_config.json (db/cocoa.duckdb): one fake table (id 1),
col 0 = apple/banana/cherry, col 1 = 10/20/30.

Run with (needs -m, project root):
    uv run python -m examples.build_demo_db

Then:
    uv run python run_cocoa.py --input examples/demo_queries.csv --output examples/demo_output.csv \
        --query-column query_column --target-column target_column \
        --k-c 1 --k-t 1 --db-config config/cocoa_duckdb_config.json --db-profile demo
"""
import json
import os
import duckdb

from src.DataAugmentation import create_index

def main():
    with open("config/cocoa_duckdb_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    db_path = config["connection"]["demo"]["database"]
    tables = config["tables"]

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    conn.execute(f"CREATE TABLE {tables['dt']} (tokenized TEXT, table_col_id TEXT NOT NULL)")
    conn.execute(
        f"CREATE TABLE {tables['mt']} (tokenized TEXT, tableid INT NOT NULL, rowid INT NOT NULL, table_col_id TEXT NOT NULL)")
    conn.execute(f"CREATE TABLE {tables['mc']} (tableid INT NOT NULL, max_colid INT NOT NULL, PRIMARY KEY (tableid))")
    conn.execute(f"INSERT INTO {tables['mc']} VALUES (1, 1)")
    conn.execute(
        f"INSERT INTO {tables['dt']} VALUES ('apple', '1_0'), ('banana', '1_0'), ('cherry', '1_0')"
    )
    conn.execute(
        f"""
        INSERT INTO {tables['mt']} VALUES
            ('apple', 1, 0, '1_0'), ('banana', 1, 1, '1_0'), ('cherry', 1, 2, '1_0'),
            ('10', 1, 0, '1_1'), ('20', 1, 1, '1_1'), ('30', 1, 2, '1_1')
        """
    )

    min_index, order_list, binary_list = create_index(["10", "20", "30"])
    conn.execute(
        f"CREATE TABLE {tables['oi']} "
        "(table_col_id TEXT NOT NULL, is_numeric BOOLEAN, min_index INT NOT NULL, order_list TEXT, binary_list TEXT)"
    )
    conn.execute(
        f"INSERT INTO {tables['oi']} VALUES (?, ?, ?, ?, ?)",
        [
            "1_1",
            True,
            int(min_index),
            ",".join(str(x) for x in order_list),
            ",".join(str(x) for x in binary_list),
        ],
    )

    conn.close()
    print(f"Demo database built at {db_path}")

if __name__ == "__main__":
    main()
