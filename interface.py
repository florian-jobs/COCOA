"""
Online-phase entry point: runs cocoa (DataAugmentation.COCOAHandler.enrich)
against an already-built index (see src/build_index.py for the offline
phase). Exposes both a Python function, run_cocoa_experiment(), and a CLI
wrapper around it, for use outside of the beluga baseline (baseline.py).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from src.DataAugmentation import COCOAHandler

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_DB_CONFIG = _PROJECT_ROOT / "config" / "cocoa_duckdb_config.json"

@dataclass
class COCOAResult:
    """Output of run_cocoa_experiment(): the input data with the top k_c correlated external columns joined in."""
    data: pd.DataFrame
    k_c: int
    k_t: int
    query_column: str
    target_column: str

    def to_csv(self, path: str) -> None:
        self.data.to_csv(path, index=False)

def _load_db_config(db_config: str | dict) -> dict:
    """Accepts either an already-loaded config dict or a path to one, and returns the dict either way."""
    if isinstance(db_config, dict):
        return db_config
    path = Path(db_config)
    path = path if path.is_absolute() else _PROJECT_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))

def _resolve_database_path(database: str) -> str:
    """Resolves a db path from the config relative to the project root, unless it's already absolute."""
    path = Path(database)
    return str(path if path.is_absolute() else _PROJECT_ROOT / path)

def run_cocoa_experiment(
        data: pd.DataFrame,
        *,
        k_c: int,
        k_t: int,
        query_column: str = "query",
        target_column: str = "target",
        db_config: str | dict | None = None,
        db_profile: str = "demo",
        conn: duckdb.DuckDBPyConnection | None = None,
        leaky_features: dict[str, list[int]] | None = None,
        base_table_name: str | None = None,
) -> COCOAResult:
    """
    Enriches `data` with the top k_c external columns (out of k_t overlap
    candidates) that best correlate with target_column, matched via
    query_column against the index described by db_config/db_profile.

    :param data: Input dataframe, must contain query_column and target_column.
    :param k_c: Number of top-correlating external columns to join in.
    :param k_t: Number of overlap candidates to consider before ranking.
    :param query_column: Column in `data` to match against the index.
    :param target_column: Column in `data` to correlate external columns against.
    :param db_config: Path to a DuckDB config json, or an already-loaded dict. Defaults to config/cocoa_duckdb_config.json.
    :param db_profile: Which entry under the config's "connection" block to use.
    :param conn: Reuse an existing DuckDB connection instead of opening (and later closing) a new one.
    :param leaky_features: Optional dict mapping candidate table name -> list of leaky column ids to
        exclude from ranking (same format as arda/qcr's leaky_features.json).
    :param base_table_name: Optional name of the query/base table itself, excluded from ranking if
        the corpus also contains a table by that same name (a self-join in disguise).
    """
    if k_c < 0 or k_t < 0:
        raise ValueError(f"k_c and k_t must be >= 0, got k_c={k_c}, k_t={k_t}")

    config = _load_db_config(db_config if db_config is not None else _DEFAULT_DB_CONFIG)
    tables = config["tables"]

    owns_conn = conn is None
    if owns_conn:
        # Without an explicit memory_limit, DuckDB has no ceiling of its own and just keeps growing
        # until the OS OOM-killer SIGKILLs the process (seen on base tables with a large number of
        # distinct join values, e.g. nyc_street_trees at ~517k rows, where the overlap query's
        # `WHERE tokenized IN (...)` scans/aggregates the full corpus-wide distinct_tokens table).
        # Capping it makes DuckDB spill intermediate results to temp_directory instead of crashing.
        db_path = Path(_resolve_database_path(config["connection"][db_profile]["database"]))
        conn = duckdb.connect(
            str(db_path),
            config={
                "memory_limit": "200GB",
                "temp_directory": str(db_path.with_name(db_path.stem + "_spill")),
            },
        )

    try:
        handler = COCOAHandler(conn, tables)
        enriched = handler.enrich(
            data,
            k_c=k_c,
            k_t=k_t,
            query_column=query_column,
            target_column=target_column,
            leaky_features=leaky_features,
            base_table_name=base_table_name,
        )

        return COCOAResult(
            data=enriched,
            k_c=k_c,
            k_t=k_t,
            query_column=query_column,
            target_column=target_column,
        )
    finally:
        if owns_conn:
            conn.close()

def _non_negative_int(value: str) -> int:
    """argparse type= helper: parses value as an int, rejecting negatives."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {parsed}")
    return parsed

def main() -> None:
    """CLI wrapper: reads --input, runs run_cocoa_experiment(), writes the result to --output."""
    parser = argparse.ArgumentParser(
        description="Run cocoa for an external experiment harness (CLI adapter around run_cocoa_experiment())."
    )
    parser.add_argument("--input", required=True, help="CSV file containing the dataset to augment.")
    parser.add_argument("--output", required=True, help="Path to write the enriched CSV.")
    parser.add_argument("--query-column", required=True,
                        help="Query column in --input to match against the external data.")
    parser.add_argument("--target-column", required=True,
                        help="Target column in --input to correlate external columns against.")
    parser.add_argument("--k-c", type=_non_negative_int, required=True,
                        help="Number of top-correlating columns to join in.")
    parser.add_argument("--k-t", type=_non_negative_int, required=True,
                        help="Number of overlap candidates to consider before ranking.")
    parser.add_argument("--db-config", required=True,
                        help="Path to a DuckDB config JSON (see config/cocoa_duckdb_config.json).")
    parser.add_argument("--db-profile", required=True, choices=["demo", "real"],
                        help="Which connection to use from --db-config's \"connection\" block.")
    parser.add_argument("--leaky-features", required=False,
                        help="Path to a leaky_features.json (table name -> list of leaky column ids) to exclude from ranking.")

    args = parser.parse_args()

    leaky_features = None
    if args.leaky_features:
        leaky_features = json.loads(Path(args.leaky_features).read_text(encoding="utf-8"))

    data = pd.read_csv(args.input)
    result = run_cocoa_experiment(
        data,
        k_c=args.k_c,
        k_t=args.k_t,
        query_column=args.query_column,
        target_column=args.target_column,
        db_config=args.db_config,
        db_profile=args.db_profile,
        leaky_features=leaky_features,
    )

    result.to_csv(args.output)

if __name__ == "__main__":
    main()
