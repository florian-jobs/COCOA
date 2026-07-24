import polars as pl
import warnings
import json
import os
from importlib import resources
from pathlib import Path

from beluga.config.schema import Config
from beluga.online.base_table import read_base_table
from beluga.resources.utils import POLARS_NUMERIC_TYPES

from src import build_index
from interface import run_cocoa_experiment

_DB_CONFIG = "config/cocoa_duckdb_config.json"
_DB_PROFILE = "real"

class COCOABaseline:
    def __init__(
            self,
            k_c: int = 5,
            k_t: int = 20,
            db_config: str = "config/cocoa_duckdb_config.json",
            db_profile: str = "real",
            rebuild_index: bool = False,
    ) -> None:
        self.k_c = k_c
        self.k_t = k_t
        self.rebuild_index = rebuild_index

    def run(
            self,
            config: Config | None = None
    ) -> pl.DataFrame:
        config = Config() if config is None else config

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

        # Offline phase.
        with open(_DB_CONFIG, "r", encoding="utf-8") as f:
            db_path = json.load(f)["connection"][_DB_PROFILE]["database"]

        if self.rebuild_index or not os.path.exists(db_path):
            build_index.main(argv=["--corpora", str(corpus_dir)])

        # Online Phase.
        base_table_df = read_base_table(config.base_table, table_dir, config)

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

"""
from beluga.config.loader import load_config

config = load_config("config.yaml")

cocoa = COCOABaseline()

print(cocoa.run(config))
"""
