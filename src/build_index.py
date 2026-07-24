import glob
import argparse
import json
import os
import duckdb
import pandas as pd

from src.DataAugmentation import create_index, get_cleaned_text

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
    col_position = {col: i for i, col in enumerate(df.columns)}
    long_df = df.reset_index(names="rowid").melt(id_vars="rowid", var_name="colname", value_name="value")
    long_df["colid"] = long_df["colname"].map(col_position)
    return long_df

def build_main_tokenized(long_df, tableid):
    rows = long_df.dropna(subset=["value"]).copy()
    rows["tokenized"] = rows["value"].apply(tokenize_cell)
    rows["tableid"] = tableid
    rows["table_col_id"] = rows["tableid"].astype(str) + "_" + rows["colid"].astype(str)
    return rows[["tokenized", "tableid", "rowid", "table_col_id"]]

def build_order_index_rows(df, tableid):
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

    conn.close()

if __name__ == "__main__":
    main()
