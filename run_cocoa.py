"""
    CLI for running COCOA.
    uv run python run_cocoa.py \
        --input my_queries.csv
        --output enriched.csv \
        --query-column my_query_column
        --target-column my_target_column \
        --k-c 5 --k-t 100 --db-config config/cocoa_duckdb_config.json --db-profile demo

    Main module for running the COCOA data augmentation pipeline. This script reads a dataset
    in CSV format, applies specific augmentation operations using the COCOAHandler, and writes
    the enriched dataset to a new CSV file. The script utilizes a DuckDB-backed configuration
    for managing database connections and external data joining tasks.

    Args:
        --input (str): Path to the CSV file containing the dataset to augment.
            Required.
        --output (str): Path to write the enriched dataset in CSV format.
            Required.
        --query-column (str): Name of the column in the input dataset used to match
            against external data for correlation operations. Required.
        --target-column (str): Name of the column in the input dataset used to correlate
            external columns during data augmentation. Required.
        --k-c (int): Number of top-correlating columns to join from external data.
            Must be a non-negative integer. Required.
        --k-t (int): Number of overlap candidates to consider from external data before
            ranking correlated columns. Must be a non-negative integer. Required.
        --db-config (str): Path to a JSON configuration file defining DuckDB connections
            and table information. Required.
        --db-profile (str): Profile key in the "connection" block of --db-config specifying
            which database connection details to use. Must be either "demo" or "real". Required.
"""
import argparse
import json
import pandas as pd
import duckdb

from src.DataAugmentation import COCOAHandler

def _non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {parsed}")
    return parsed

def main():
    parser = argparse.ArgumentParser(description="Run COCOA data augmentation baseline.")
    parser.add_argument("--input", required=True, help="CSV file containing the dataset to augment.")
    parser.add_argument("--output", required=True, help="Path to write the enriched CSV.")
    parser.add_argument("--query-column", required=True,
                        help="Query column in --input to match against the external data.")
    parser.add_argument("--target-column", required=True,
                        help="Target column in --input to correlate external columns against.")
    parser.add_argument("--k-c", type=_non_negative_int, required=True,
                        help="Number of k top-correlating columns to join in.")
    parser.add_argument("--k-t", type=_non_negative_int, required=True,
                        help="Number of overlap candidates to consider before ranking.")
    parser.add_argument("--db-config", required=True,
                        help="Path to a DuckDB config JSON (see config/cocoa_duckdb_config.json).")
    parser.add_argument("--db-profile", required=True, choices=["demo", "real"],
                        help="Which connection to use from --db-config's \"connection\" block.")

    args = parser.parse_args()

    with open(args.db_config, "r", encoding="utf-8") as f:
        config = json.load(f)

    data = pd.read_csv(args.input)

    conn = duckdb.connect(config["connection"][args.db_profile]["database"])
    try:
        cocoa = COCOAHandler(conn, config["tables"])

        result = cocoa.enrich(
            data=data,
            k_c=args.k_c,
            k_t=args.k_t,
            query_column=args.query_column,
            target_column=args.target_column,
        )

        result.to_csv(args.output, index=False)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
