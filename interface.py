from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd

from src.DataAugmentation import COCOAHandler

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_DB_CONFIG = _PROJECT_ROOT / "config" / "cocoa_duckdb_config.json"

@dataclass
class ColumnScore:
    table_col_id: str
    correlation: float | None
    is_numeric: bool | None

@dataclass
class COCOAResult:
    data: pd.DataFrame
    selected_columns: list[ColumnScore]
    k_c: int
    k_t: int
    query_column: str
    target_column: str

    def to_csv(self, path: str) -> None:
        self.data.to_csv(path, index=False)

    def scores_to_json(self, path: str) -> None:
        payload = {
            "k_c": self.k_c,
            "k_t": self.k_t,
            "query_column": self.query_column,
            "target_column": self.target_column,
            "selected_columns": [asdict(c) for c in self.selected_columns],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _load_db_config(db_config: str | dict) -> dict:
    if isinstance(db_config, dict):
        return db_config
    return json.loads(Path(db_config).read_text(encoding="utf-8"))

def _resolve_database_path(database: str) -> str:
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
) -> COCOAResult:
    if k_c < 0 or k_t < 0:
        raise ValueError(f"k_c and k_t must be >= 0, got k_c={k_c}, k_t={k_t}")

    config = _load_db_config(db_config if db_config is not None else _DEFAULT_DB_CONFIG)
    tables = config["tables"]

    owns_conn = conn is None
    if owns_conn:
        conn = duckdb.connect(_resolve_database_path(config["connection"][db_profile]["database"]))

    try:
        handler = COCOAHandler(conn, tables)
        original_columns = set(data.columns)
        enriched = handler.enrich(
            data,
            k_c=k_c,
            k_t=k_t,
            query_column=query_column,
            target_column=target_column,
        )

        cocoa_meta = enriched.attrs.get("cocoa")
        if cocoa_meta is not None:
            selected_columns = [ColumnScore(**c) for c in cocoa_meta["selected_columns"]]
        else:
            new_columns = [
                c for c in enriched.columns
                if c not in original_columns and c != "rank_target"
            ]
            selected_columns = [
                ColumnScore(table_col_id=c, correlation=None, is_numeric=None)
                for c in new_columns
            ]

        return COCOAResult(
            data=enriched,
            selected_columns=selected_columns,
            k_c=k_c,
            k_t=k_t,
            query_column=query_column,
            target_column=target_column,
        )
    finally:
        if owns_conn:
            conn.close()

def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {parsed}")
    return parsed

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run COCOA for an external experiment harness (CLI adapter around run_cocoa_experiment())."
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
    parser.add_argument("--scores-output", default=None,
                        help="Optional path to write the selected external columns as JSON.")

    args = parser.parse_args()

    data = pd.read_csv(args.input)
    result = run_cocoa_experiment(
        data,
        k_c=args.k_c,
        k_t=args.k_t,
        query_column=args.query_column,
        target_column=args.target_column,
        db_config=args.db_config,
        db_profile=args.db_profile,
    )

    # result.to_csv(args.output)
    # if args.scores_output:
    #     result.scores_to_json(args.scores_output)

if __name__ == "__main__":
    main()
