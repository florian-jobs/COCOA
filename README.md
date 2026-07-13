# COCOA
### COrrelation COefficient-Aware Data Augmentation

Minimal changes to original repository. Dependencies via [uv](https://docs.astral.sh/uv/); backend is [DuckDB](https://duckdb.org/).

## Setup

```
uv sync
```

⚠️ **pandas is pinned to `1.5.3`** — newer pandas changes `groupby()` behavior in a way that breaks `DataAugmentation.py`. Don't bump it without re-running the tests below.

Prefix every command with `uv run` (e.g. `uv run pytest`, `uv run python run_cocoa.py ...`).

## Test

```
uv run pytest -v
```
Covers: duckdb/config sanity checks, helper-function smoke tests, the offline index builder, `interface.py`, and a full synthetic-data pipeline run. CI runs this on every push (`.github/workflows/ci.yml`).

## Demo (fake data)

```
uv run python -m examples.build_demo_db
uv run python run_cocoa.py --input examples/demo_queries.csv --output examples/demo_output.csv --query-column query_column --target-column target_column --k-c 1 --k-t 1 --db-config config/cocoa_duckdb_config.json --db-profile demo
cat examples/demo_output.csv
```
Expected: `apple,5,2.0,10` and `banana,15,1.0,20`.

## Real data 

```
uv run -m examples.build_real_index 
uv run python run_cocoa.py --input <csv> --output <csv> --query-column <col> --target-column <col> --k-c <N> --k-t <N> --db-config <config.json> --db-profile <demo|real>
```
`--k-t`: overlap candidates considered. `--k-c`: top correlating columns joined in.

Or call the library directly — see `run_cocoa.py` for the reference usage of `COCOAHandler` (`src/DataAugmentation.py`).


## For external experiments

`interface.py` is entry point to call into COCOA: either by shelling out to its CLI or by importing `run_cocoa_experiment()` directly. It wraps `COCOAHandler` from `src/DataAugmentation.py` (unchanged).

CLI (same flags as `run_cocoa.py`, plus `--scores-output`; also installed as `cocoa-interface` after `uv sync`):
```
uv run python interface.py \
    --input examples/demo_queries.csv --output examples/demo_output.csv \
    --query-column query_column --target-column target_column \
    --k-c 1 --k-t 1 --db-config config/cocoa_duckdb_config.json --db-profile demo \
    --scores-output examples/demo_scores.json
```

Python:
```python
from interface import run_cocoa_experiment

result = run_cocoa_experiment(
    data, k_c=1, k_t=1,
    query_column="query_column", target_column="target_column",
    db_config="config/cocoa_duckdb_config.json", db_profile="demo",
    # conn=<your own open duckdb.DuckDBPyConnection>,  # optional: manage the connection yourself
)
result.data              # enriched pandas DataFrame, same shape as run_cocoa.py's output
result.selected_columns  # list[ColumnScore]: which external columns were joined in
```

## Database schema

```sql
CREATE TABLE main_tokenized  (tokenized TEXT, tableid INT NOT NULL, rowid INT NOT NULL, table_col_id TEXT NOT NULL);
CREATE TABLE distinct_tokens (tokenized TEXT, table_col_id TEXT NOT NULL);
CREATE TABLE order_index     (table_col_id TEXT NOT NULL, is_numeric BOOLEAN, min_index INT NOT NULL, order_list TEXT, binary_list TEXT);
CREATE TABLE max_column      (tableid INT NOT NULL, max_colid INT NOT NULL, PRIMARY KEY (tableid));
```
- `main_tokenized`: inverted index, token -> table/col/row.
- `distinct_tokens`/`max_column`: derived from `main_tokenized`.
- `order_index`: one row per column, built with `create_index(values)` from `src/DataAugmentation.py`.