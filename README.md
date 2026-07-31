# COCOA
### COrrelation COefficient-Aware Data Augmentation

Dependencies via [uv](https://docs.astral.sh/uv/); backend is [DuckDB](https://duckdb.org/). This branch contains only the production/baseline code — tests and demo data live on a separate branch.

## Setup

```
uv sync
```

Prefix every command with `uv run`.

## Beluga baseline (primary entry point)

```python
from baseline import COCOABaseline

cocoa = COCOABaseline(k_c=5, k_t=20)
result = cocoa.run(config)  # config: beluga.config.schema.Config -> polars.DataFrame
```
Builds the offline index from `config.data_dir`/`config.corpus` if missing (or `rebuild_index=True`), then enriches `config.queries_dir`/`config.base_table`. DB config/profile are hardcoded.

`build_index`/`interface.py` below are what `baseline.py` calls internally — only needed manually, e.g. outside of Beluga.

## Manual: build the index

```
uv run python -m src.build_index --corpora <corpus dir> [--limit <N>]
```
`--limit` caps how many CSVs get indexed (for quick tests). DB target (`config/cocoa_duckdb_config.json`, `"real"` profile) is hardcoded.

Order index (`is_numeric`/rank ties) is built from the **tokenized** cell text (same as `main_tokenized`), not the raw CSV values — matching the original COCOA pipeline, where index generation reads from `main_tokenized.tokenized` rather than the source CSVs.

## Manual: run COCOA

```
uv run python interface.py \
    --input <csv> --output <csv> --query-column <col> --target-column <col> \
    --k-c <N> --k-t <N> --db-config config/cocoa_duckdb_config.json --db-profile real
```
Or programmatically via `from interface import run_cocoa_experiment`.

## Testing status

`build_index` + `run_cocoa_experiment` (what `baseline.py` calls internally) tested end-to-end against a real corpus:

```
uv run python test_run_cocoa_experiment.py \
    --corpora ~/data/corpora/open_data/joinable_tables/ --limit 5 \
    --input ~/data/corpora/open_data/joinable_tables/nyc/nyc-finance-39g5-gbp3/table.csv \
    --query_column agency_name --target_column total_current_budget_amount
```

`baseline.py` itself **not tested yet** — no `beluga` access so far.
