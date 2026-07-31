import argparse
import json
import os
from pathlib import Path

import duckdb
import pandas as pd

from src.DataAugmentation import create_index, get_cleaned_text

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {parsed}")
    return parsed

# Decided not to touch DataAugmentation.py thus redeclared here.
def tokenize_cell(value):
    return get_cleaned_text(str(value).lower())

# Decided not to touch DataAugmentation.py thus redeclared here.
def _is_numeric(s):
    if s.lower() == 'nan':
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False

# Decided not to touch DataAugmentation.py thus redeclared here.
def _is_numeric_list(values):
    return all(_is_numeric('nan' if (v is None or v == '') else str(v)) for v in values)

def melt_dataframe(df):
    # Melt first and attach rowid via the index afterwards, so a source column
    # literally named "rowid" can't collide with the rowid column we add.
    col_position = {col: i for i, col in enumerate(df.columns)}
    long_df = df.melt(var_name="colname", value_name="value", ignore_index=False)
    long_df.index.name = "rowid"
    long_df = long_df.reset_index()
    long_df["colid"] = long_df["colname"].map(col_position)
    return long_df

def build_main_tokenized(long_df, tableid):
    # Keep every rowid (even NaN cells as empty tokens) so that rowids stay aligned
    # with the full-column order_index built by create_index() (see build_order_index_rows).
    rows = long_df.copy()
    rows["tokenized"] = rows["value"].fillna("").apply(tokenize_cell)
    rows["tableid"] = tableid
    rows["table_col_id"] = rows["tableid"].astype(str) + "_" + rows["colid"].astype(str)
    return rows[["tokenized", "tableid", "rowid", "table_col_id"]]

def build_order_index_rows(df, tableid):
    # Index the same tokenized text that ends up in main_tokenized (not the raw
    # cell values), matching the original COCOA pipeline where generate_order_index
    # reads FROM main_tokenized.tokenized instead of the source CSVs.
    rows = []
    for colid, colname in enumerate(df.columns):
        tokenized_values = df[colname].fillna("").apply(tokenize_cell).tolist()

        is_numeric = _is_numeric_list(tokenized_values)

        min_index, order_list, binary_list = create_index(tokenized_values)

        rows.append({
            "table_col_id": f"{tableid}_{colid}",
            "is_numeric": is_numeric,
            "min_index": int(min_index),
            "order_list": ",".join(str(x) for x in order_list),
            "binary_list": ",".join(str(x) for x in binary_list),
        })
    return pd.DataFrame(rows)

def main(argv=None):
    # Parser for stating table corpora directory.
    parser = argparse.ArgumentParser(description="Run COCOA indexing.")
    parser.add_argument("--corpora", required=False,
                        help="Directory containing the table corpora. Defaults to dataset/.")
    parser.add_argument("--limit", required=False, type=_non_negative_int,
                        help="limit the number of csv's to process for testing purposes")
    args = parser.parse_args(argv)

    with open(_PROJECT_ROOT / "config" / "cocoa_duckdb_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    db_path = _PROJECT_ROOT / config["connection"]["real"]["database"]
    tables = config["tables"]

    os.makedirs(db_path.parent, exist_ok=True)

    # Build into a temporary file first and only replace the real db once the
    # build succeeds, so a bad run doesn't destroy a previously working index.
    tmp_db_path = db_path.with_name(db_path.name + ".tmp")
    if os.path.exists(tmp_db_path):
        os.remove(tmp_db_path)

    # Obtain all csv paths. If --corpora is specified, use that, else use dataset/.
    corpora_dir = args.corpora if args.corpora is not None else _PROJECT_ROOT / "dataset"
    csv_paths = sorted(os.path.join(root, file)
                       for root, dirs, files in os.walk(corpora_dir)
                       for file in files if
                       file.endswith(".csv"))

    if args.limit is not None:
        csv_paths = csv_paths[:args.limit]

    print(f"Found {len(csv_paths)} csv's")

    conn = duckdb.connect(tmp_db_path)
    try:
        conn.execute(
            f"CREATE TABLE {tables['mt']} (tokenized TEXT, tableid INT NOT NULL, rowid INT NOT NULL, table_col_id TEXT NOT NULL)")
        conn.execute(f"CREATE TABLE {tables['dt']} (tokenized TEXT, table_col_id TEXT NOT NULL)")
        conn.execute(
            f"CREATE TABLE {tables['oi']} (table_col_id TEXT NOT NULL, is_numeric BOOLEAN, min_index INT NOT NULL, order_list TEXT, binary_list TEXT)")
        conn.execute(
            f"CREATE TABLE {tables['mc']} (tableid INT NOT NULL, max_colid INT NOT NULL, PRIMARY KEY (tableid))")

        skipped = []
        for tableid, path in enumerate(csv_paths, start=1):
            filename = os.path.basename(path)
            try:
                df = pd.read_csv(path)
                if df.empty:
                    raise ValueError("no data rows")
                long_df = melt_dataframe(df)
                tmp_tokenized_df = build_main_tokenized(long_df, tableid)
                tmp_order_index_df = build_order_index_rows(df, tableid)
            except Exception as e:
                print(f"Skipping {filename}: {e}")
                skipped.append(filename)
                continue

            conn.register("tmp_tokenized", tmp_tokenized_df)
            conn.execute(f"INSERT INTO {tables['mt']} SELECT * FROM tmp_tokenized")

            conn.register("tmp_order_index", tmp_order_index_df)
            conn.execute(f"INSERT INTO {tables['oi']} SELECT * FROM tmp_order_index")

            conn.unregister("tmp_tokenized")
            conn.unregister("tmp_order_index")

        conn.execute(f"INSERT INTO {tables['dt']} SELECT DISTINCT tokenized, table_col_id FROM {tables['mt']}")
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
    except Exception:
        conn.close()
        os.remove(tmp_db_path)
        raise
    else:
        conn.close()

    if os.path.exists(db_path):
        os.remove(db_path)
    os.replace(tmp_db_path, db_path)

    if skipped:
        print(f"Skipped {len(skipped)} csv's: {', '.join(skipped)}")
    print(f"Real index built at {db_path}")

if __name__ == "__main__":
    main()
