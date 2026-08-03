"""
Beluga entry point for the COCOA baseline: wires build_index.py (offline
index build) and interface.run_cocoa_experiment (online query) together
behind beluga's Config object, so beluga can run COCOA without knowing
anything about its internals.
"""

import polars as pl
import warnings
import json
from importlib import resources
from pathlib import Path

from beluga.config.schema import Config
from beluga.online.base_table import read_base_table
from beluga.resources.utils import POLARS_NUMERIC_TYPES

from src import build_index
from interface import run_cocoa_experiment

_PROJECT_ROOT = Path(__file__).resolve().parent
_DB_CONFIG = _PROJECT_ROOT / "config" / "cocoa_duckdb_config.json"
_DB_PROFILE = "real"

class COCOABaseline:
    """
    Runs the full COCOA pipeline (build index if needed, then augment) for beluga.

    :ivar k_c: Number of top-correlating external columns to join into the result.
    :type k_c: int
    :ivar k_t: Number of overlap-candidate columns considered before ranking.
    :type k_t: int
    :ivar rebuild_index: Force a full index rebuild even if one already exists on disk.
    :type rebuild_index: bool
    """

    def __init__(
            self,
            k_c: int = 10,
            k_t: int = 50,
            rebuild_index: bool = False,
    ) -> None:
        self.k_c = k_c
        self.k_t = k_t
        self.rebuild_index = rebuild_index

    def run(
            self,
            config: Config | None = None
    ) -> pl.DataFrame:
        """Runs the pipeline for one beluga Config and returns the augmented base table."""
        config = Config() if config is None else config

        # NB: target_column_id is only validated here (must be set), not actually
        # used to pick the column - which column is join/target is determined
        # positionally below (first/last column of the base table).
        if not config.target_column_id:
            raise ValueError("Value for target_column_id not specified in the configuration file")

        if config.queries_dir is not None:
            table_dir = Path(config.queries_dir) / config.base_table
        else:
            table_dir = resources.files("beluga.data").joinpath(
                "queries/beers")  # to update with a new default base table

        if config.data_dir is not None:
            corpus_dir = Path(config.data_dir) / config.corpus
        else:
            corpus_dir = resources.files("beluga.data").joinpath("corpora/toy")

        # Offline phase: (re)build the DuckDB index if it's missing or a rebuild was requested.
        with open(_DB_CONFIG, "r", encoding="utf-8") as f:
            db_path = _PROJECT_ROOT / json.load(f)["connection"][_DB_PROFILE]["database"]

        if self.rebuild_index or not db_path.exists():
            build_index.main(argv=["--corpora", str(corpus_dir)])

        # Online phase: query the index and join in the best-correlating external columns.
        base_table_df = read_base_table(config.base_table, table_dir, config)

        # Base table convention: join column is always first, target column always last.
        join_column_id = 0
        join_column = base_table_df.columns[join_column_id]
        target_column_id = len(base_table_df.columns) - 1
        target_column = base_table_df.columns[target_column_id]

        if base_table_df.schema[target_column] not in POLARS_NUMERIC_TYPES:
            raise ValueError(f"Target column ({target_column!r}) not numeric")

        data = base_table_df.to_pandas()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            augmented_table = run_cocoa_experiment(
                data,
                k_c=self.k_c,
                k_t=self.k_t,
                query_column=join_column,
                target_column=target_column,
                db_config=_DB_CONFIG,
                db_profile=_DB_PROFILE,
            )

        return pl.from_pandas(augmented_table.data)

# Example usage (kept as reference; needs a real config.yaml with target_column_id set):
"""
from beluga.config.loader import load_config

config = load_config("config.yaml")

cocoa = COCOABaseline()

print(cocoa.run(config))
"""
