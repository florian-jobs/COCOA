# COCOA
### COrrelation COefficient-Aware Data Augmentation

Dependencies via [uv](https://docs.astral.sh/uv/); backend is [DuckDB](https://duckdb.org/). This branch contains only the production/baseline code — tests and demo data live on a separate branch.

## Setup

```
uv sync
```

Prefix every command with `uv run`.

## Build the index

```
uv run python -m src.build_index --corpora <path to corpus directory>
```
Database target (`config/cocoa_duckdb_config.json`, `"real"` profile) is hardcoded, not a flag.

## Run COCOA

```
uv run python interface.py \
    --input <csv path> --output <csv path> \
    --query-column <col> --target-column <col> \
    --k-c <N> --k-t <N> --db-config config/cocoa_duckdb_config.json --db-profile real
```
Or programmatically via `from interface import run_cocoa_experiment`.

## Beluga baseline

```python
from baseline import COCOABaseline

cocoa = COCOABaseline(k_c=5, k_t=20)
result = cocoa.run(config)  # config: beluga.config.schema.Config -> polars.DataFrame
```
Builds the offline index from `config.data_dir`/`config.corpus` if missing (or if `rebuild_index=True`), enriches `config.queries_dir`/`config.base_table`. DB config/profile are hardcoded, not per-instance.
