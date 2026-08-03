# COCOA
### COrrelation COefficient-Aware Data Augmentation

Deps via [uv](https://docs.astral.sh/uv/), backend is [DuckDB](https://duckdb.org/). This branch has only the production/baseline code — tests and demo data live on a separate branch.

## Setup

```
uv sync
```

Prefix every command with `uv run`.

## Testing on the server

Two scripts, covering the two phases (index build, and the actual COCOA run):

**1. `test_run_cocoa_experiment.py`** — builds the index over a corpus, then runs COCOA end to end on one input table.

```
uv run python test_run_cocoa_experiment.py \
    --corpora ~/data/corpora/open_data/joinable_tables/ --limit 5 \
    --input ~/data/corpora/open_data/joinable_tables/nyc/nyc-finance-39g5-gbp3/table.csv \
    --query_column agency_name --target_column total_current_budget_amount
```

**2. `test_compare_index.py`** — checks that the index-building math (rank/order index, numeric detection) matches the original COCOA implementation, run on the same csv's. No db, no original checkout needed — the original's functions are vendored into the script.

```
uv run python test_compare_index.py --corpora ~/data/corpora/open_data/joinable_tables/ --limit 100
```

Both take `--corpora <dir>` and `--limit <N>` to control how much of a corpus gets used, so start small (`--limit 5`) before pointing them at the full `open_data`/`web_tables` corpora.

Index builds are safe to run unattended on a large corpus: runs are locked so two builds against the same db don't race each other, each csv commits on its own, and a killed/crashed build leaves whatever was already indexed in place instead of wiping it — just rerun to pick back up.

## Beluga baseline

```python
from baseline import COCOABaseline

cocoa = COCOABaseline(k_c=10, k_t=50)
result = cocoa.run(config)  # config: beluga.config.schema.Config -> polars.DataFrame
```

Reads its inputs from the beluga `Config` object rather than hardcoding them:

- `config.target_column_id` — required, raises if not set
- `config.data_dir` / `config.corpus` — which corpus to index. If not set, falls back to beluga's bundled toy corpus
- `config.queries_dir` / `config.base_table` — which base table to query. If not set, falls back to beluga's bundled `beers` table

So any of these can be left out of the config and the baseline still runs, against the packaged defaults, instead of failing. Implemented, but not yet run against a real beluga config — no beluga access so far.

## Manual usage

The pipeline is two independent phases: build the index once (offline), then query it as many times as you want (online). Useful to run separately when you're testing just one of the two.

**Phase 1 — build the index** (`src/build_index.py`): reads a corpus of csv's and writes the DuckDB index (tokenized content + order index) to disk. Only needs to run again if the corpus changes.
```
uv run python -m src.build_index --corpora <corpus dir> [--limit <N>]
```

**Phase 2 — run COCOA** (`src/DataAugmentation.py`, via `interface.py`): reads an existing index and enriches one input table against it. Can be run repeatedly against the same index, e.g. for different query/target columns.
```
uv run python interface.py \
    --input <csv> --output <csv> --query-column <col> --target-column <col> \
    --k-c <N> --k-t <N> --db-config config/cocoa_duckdb_config.json --db-profile real
```
Or programmatically via `from interface import run_cocoa_experiment` (wraps `DataAugmentation.COCOAHandler.enrich`). DB config/profile for both phases is hardcoded in `config/cocoa_duckdb_config.json`.
