"""
Builds the offline COCOA index (tokenized cell text + order index) from a
directory of csv's into a local DuckDB file.

This is the "offline phase" of the pipeline: DataAugmentation.py's
COCOAHandler.enrich() is the "online" query phase that reads the index this
module produces. Mirrors the original COCOA index_generation.py pipeline
(tokenize -> order index -> is_numeric), but targets DuckDB instead of
Vertica and builds directly from source csv's instead of an
already-populated main_tokenized table.
"""

import argparse
import contextlib
import json
import os
import re
import threading
import time
from pathlib import Path

import duckdb
import pandas as pd

from src.DataAugmentation import create_index, get_cleaned_text

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _non_negative_int(value: str) -> int:
    """argparse type= helper: parses value as an int, rejecting negatives."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {parsed}")
    return parsed

# tokenize_cell and the _is_numeric* helpers below duplicate private logic that
# also lives in DataAugmentation.py. Redeclared here (not imported) so that
# file doesn't need to change to expose them - see also create_index, which
# IS imported from there since it's already public.
def tokenize_cell(value):
    """Canonical per-cell text used everywhere in the index (main_tokenized and the order index): lowercased and stripped of punctuation/stopwords."""
    return get_cleaned_text(str(value).lower())

# Fast pre-check for "could this even be a number float() would accept" - lets
# us skip the try/except below for cells that obviously aren't numeric, which
# matters at large-corpus scale since raising/catching an exception per cell
# is expensive. This regex only ever rejects; float() below still has the
# final say on whether a value that passes is actually numeric.
_LOOKS_NUMERIC_RE = re.compile(
    r'^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$|^\s*[+-]?inf(inity)?\s*$',
    re.IGNORECASE,
)

def _is_numeric(s):
    """True if float(s) would succeed (treats the literal string "nan" as numeric too, matching the original COCOA behaviour)."""
    if s.lower() == 'nan':
        return True
    if not _LOOKS_NUMERIC_RE.match(s):
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

def _is_numeric_list(values):
    """True if every value in the column is numeric (missing cells count as numeric)."""
    return all(_is_numeric('nan' if (v is None or v == '') else str(v)) for v in values)

def _is_numeric_list_with_header(values):
    """
    True if every value is numeric except for a single non-numeric one - e.g.
    a stray header row baked into the data as its first "cell" instead of a
    real csv header.
    """
    found_heading = False
    for v in values:
        s = 'nan' if (v is None or v == '') else str(v)
        if _is_numeric(s):
            continue
        if not found_heading:
            found_heading = True
        else:
            return False
    return True

def melt_dataframe(df):
    """
    Reshapes a wide table (one row per record) into long format: one row per
    (rowid, column) pair, which the rest of the pipeline builds on. Melts
    first and derives rowid from the index afterwards, so a source column
    literally named "rowid" can't collide with the rowid column we add.
    """
    col_position = {col: i for i, col in enumerate(df.columns)}
    long_df = df.melt(var_name="colname", value_name="value", ignore_index=False)
    long_df.index.name = "rowid"
    long_df = long_df.reset_index()
    long_df["colid"] = long_df["colname"].map(col_position)
    return long_df

def tokenize_long_df(long_df):
    """
    Adds a "tokenized" column to a melted long_df. Done once here and reused
    by both build_main_tokenized() and build_order_index_rows(), instead of
    each re-tokenizing every cell - tokenize_cell is regex-heavy pure Python,
    so at large-corpus scale this roughly halves per-file tokenization cost.
    Keeps every rowid (even NaN cells, tokenized as empty strings) so rowids
    stay aligned with the per-column order index built in build_order_index_rows.
    """
    rows = long_df.copy()
    rows["tokenized"] = rows["value"].fillna("").apply(tokenize_cell)
    return rows

def build_main_tokenized(tokenized_long_df, tableid):
    """Shapes an already-tokenized long_df into the main_tokenized table's exact column layout."""
    rows = tokenized_long_df.copy()
    rows["tableid"] = tableid
    rows["table_col_id"] = rows["tableid"].astype(str) + "_" + rows["colid"].astype(str)
    return rows[["tokenized", "tableid", "rowid", "table_col_id"]]

def build_order_index_rows(tokenized_long_df, tableid, num_columns):
    """
    Builds one order_index row per column of the table: is_numeric plus the
    (min_index, order_list, binary_list) triple from create_index(), computed
    over each column's tokenized values (not the raw cell values) - matching
    the original COCOA pipeline, where index generation reads from
    main_tokenized.tokenized rather than the source csv's.
    """
    by_col = {
        colid: group.sort_values("rowid")["tokenized"].tolist()
        for colid, group in tokenized_long_df.groupby("colid")
    }

    rows = []
    for colid in range(num_columns):
        tokenized_values = by_col.get(colid, [])

        is_numeric = _is_numeric_list(tokenized_values) or _is_numeric_list_with_header(tokenized_values)

        min_index, order_list, binary_list = create_index(tokenized_values)

        rows.append({
            "table_col_id": f"{tableid}_{colid}",
            "is_numeric": is_numeric,
            "min_index": int(min_index),
            "order_list": ",".join(str(x) for x in order_list),
            "binary_list": ",".join(str(x) for x in binary_list),
        })
    return pd.DataFrame(rows)

@contextlib.contextmanager
def _build_lock(lock_path, poll_interval=5.0, stale_after=24 * 3600, heartbeat_interval=60.0):
    """
    Serializes concurrent build_index.main() invocations against the same
    database (e.g. multiple parallel Beluga runs racing to build a missing index),
    so they don't corrupt each other by writing to the same file at once.

    Lock ownership is a plain exclusive-create file (portable, no extra
    dependency). A lock older than `stale_after` is assumed to belong to a
    crashed process and is taken over rather than waited on forever. A
    background heartbeat refreshes the lock's mtime every `heartbeat_interval`
    seconds while held, so a build that legitimately runs longer than
    `stale_after` (realistic for very large corpora) doesn't get its lock
    stolen out from under it.
    """
    fd = None
    stop_heartbeat = threading.Event()
    heartbeat_thread = None
    try:
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, str(os.getpid()).encode())
                break
            except FileExistsError:
                try:
                    age = time.time() - os.path.getmtime(lock_path)
                except FileNotFoundError:
                    continue  # lock disappeared between the check and now, retry
                if age > stale_after:
                    print(f"Lock at {lock_path} is older than {stale_after}s, assuming stale and taking over.")
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(lock_path)
                    continue
                time.sleep(poll_interval)

        def _heartbeat():
            while not stop_heartbeat.wait(heartbeat_interval):
                with contextlib.suppress(FileNotFoundError):
                    os.utime(lock_path, None)

        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()
        yield
    finally:
        stop_heartbeat.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=heartbeat_interval)
        if fd is not None:
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            os.remove(lock_path)

def main(argv=None):
    """CLI entry point: parses --corpora/--limit, then (re)builds the index under a lock so concurrent builds can't collide."""
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

    lock_path = db_path.with_name(db_path.name + ".lock")
    with _build_lock(lock_path):
        _build(db_path, tables, args)

def _build(db_path, tables, args):
    """
    Rebuilds db_path from scratch (mt/dt/oi/mc tables) out of every csv under
    args.corpora (or a --limit-capped subset). Must be called with the build
    lock held.

    Writes go straight to db_path - no tmp-file-then-swap. Each csv is
    committed in its own transaction (see the per-file loop below), so a
    crash mid-build leaves a partial-but-internally-consistent index (some
    tables missing, none half-written) instead of losing all prior progress.
    Tradeoff, accepted for now: db_path existing no longer means "build
    finished successfully" - a crashed build leaves an incomplete-but-present
    db_path behind. Callers that need a hard "fully built" signal should
    check for that separately (e.g. a completion marker) rather than relying
    on existence alone.
    """
    if os.path.exists(db_path):
        os.remove(db_path)

    corpora_dir = args.corpora if args.corpora is not None else _PROJECT_ROOT / "dataset"
    csv_paths = sorted(os.path.join(root, file)
                       for root, dirs, files in os.walk(corpora_dir)
                       for file in files if
                       file.endswith(".csv"))

    if args.limit is not None:
        csv_paths = csv_paths[:args.limit]

    print(f"Found {len(csv_paths)} csv's")

    conn = duckdb.connect(db_path)
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
            registered = []
            try:
                # dtype=str skips pandas' per-column type inference (int/float/datetime
                # sniffing) - we stringify every cell in tokenize_cell anyway, so that
                # inference work is pure overhead. Bonus: it also avoids inference
                # artifacts like "005" -> 5 or "3.140" -> 3.14 changing the token text.
                df = pd.read_csv(path, dtype=str)
                if df.empty:
                    raise ValueError("no data rows")
                long_df = melt_dataframe(df)
                tokenized_long_df = tokenize_long_df(long_df)
                tmp_tokenized_df = build_main_tokenized(tokenized_long_df, tableid)
                tmp_order_index_df = build_order_index_rows(tokenized_long_df, tableid, len(df.columns))

                # Per-file transaction: a bad insert for one csv must not take down
                # the whole build, and must not leave mt/oi out of sync with each other.
                conn.execute("BEGIN TRANSACTION")
                conn.register("tmp_tokenized", tmp_tokenized_df)
                registered.append("tmp_tokenized")
                conn.execute(f"INSERT INTO {tables['mt']} SELECT * FROM tmp_tokenized")

                conn.register("tmp_order_index", tmp_order_index_df)
                registered.append("tmp_order_index")
                conn.execute(f"INSERT INTO {tables['oi']} SELECT * FROM tmp_order_index")
                conn.execute("COMMIT")
            except Exception as e:
                try:
                    conn.execute("ROLLBACK")
                except duckdb.Error:
                    pass
                print(f"Skipping {filename}: {e}")
                skipped.append(filename)
                continue
            finally:
                for name in registered:
                    conn.unregister(name)

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
    finally:
        conn.close()

    if skipped:
        print(f"Skipped {len(skipped)} csv's: {', '.join(skipped)}")
    print(f"Real index built at {db_path}")

if __name__ == "__main__":
    main()
