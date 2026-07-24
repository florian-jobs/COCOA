import polars as pl
import pandas as pd
import warnings
from importlib import resources
from pathlib import Path

# from beluga.config.schema import Config
# from beluga.online.augmentation import join_tables
# from beluga.online.base_table import read_base_table
# from beluga.resources.utils import POLARS_NUMERIC_TYPES

from examples import build_real_index
from interface import run_cocoa_experiment

# Hardcode parameters for testing then do it with initialization like the others.
class COCOABaseline:
    def __init__(
            self,
            k_c: int = 5,
            k_t: int = 20,
            db_config: str = "config/cocoa_duckdb_config.json",
            db_profile: str = "real",
            corpus_dir_tmp: str = "dataset/",
            data_dir_tmp: str = "dataset/movie.csv",
            query_column: str = "color",
            target_column: str = "duration",
    ) -> None:
        self.k_c = k_c
        self.k_t = k_t
        self.db_config = db_config
        self.db_profile = db_profile
        self.corpus_dir_tmp = corpus_dir_tmp
        self.data_dir_tmp = data_dir_tmp
        self.query_column = query_column
        self.target_column = target_column

    def run(
            self,
            # config: Config | None = None
    ) -> pl.DataFrame:
        # NO ACCESS TO THE CONFIG FILE YET.
        # config = Config() if config is None else config

        # if not config.target_column_id:
        #     raise ValueError("Value for target_column_id not specified in the configuration file")
        #
        # if config.queries_dir is not None:
        #     table_dir = Path(config.queries_dir) / config.base_table
        # else:
        #     table_dir = resources.files("beluga.data").joinpath(
        #         "queries/beers")  # to update with a new default base table
        # if config.data_dir is not None:
        #     corpus_dir = Path(config.data_dir) / config.corpus
        # else:
        #     corpus_dir = resources.files("beluga.data").joinpath("corpora/toy")

        # Offline phase.
        build_real_index.main(argv=["--corpora", str(self.corpus_dir_tmp)])

        # NO ACCESS TO THE CONFIG FILE YET.
        # base_table_df = read_base_table(config.base_table, table_dir, config)

        # join_column_id = 0
        # join_column = base_table_df.columns[join_column_id]
        # target_column_id = len(base_table_df.columns) - 1
        # target_column = base_table_df.columns[target_column_id]

        # if base_table_df.schema[target_column] not in POLARS_NUMERIC_TYPES:
        #     raise ValueError(f"Target column ({target_column!r}) not numeric")

        data = pd.read_csv(self.data_dir_tmp)

        augmented_table = run_cocoa_experiment(
            data,
            k_c=self.k_c,
            k_t=self.k_t,
            query_column=self.query_column,
            target_column=self.target_column,
            db_config=self.db_config,
            db_profile=self.db_profile,
        )

        return pl.from_pandas(augmented_table.data)

"""
from beluga.config.loader import load_config

config = load_config("config.yaml")

arda = ARDABaseline()

print(arda.run(config))
"""
