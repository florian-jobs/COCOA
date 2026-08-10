"""
End-to-end smoke test: builds the index over a corpus, then runs COCOA on one
input table, printing a preview of the result. Exercises both phases
(src/build_index.py and interface.run_cocoa_experiment) together against a
real corpus - see README for example invocations.
"""

import argparse

from interface import run_cocoa_experiment
from src import build_index
import pandas as pd

def main():
    parser = argparse.ArgumentParser("Run Cocoa Baseline Test")
    parser.add_argument("--corpora", required=True, help="Corpora to run")
    parser.add_argument("--limit", required=True, help="Number of csv's to process")
    parser.add_argument("--input", required=True, help="CSV file to run the experiment on, e.g. dataset/movie.csv")
    parser.add_argument("--query_column", required=True, help="Query column to run the experiment on")
    parser.add_argument("--target_column", required=True, help="Target column to run the experiment on")
    parser.add_argument("--workers", required=False, default="1",
                        help="Worker processes for the index build's per-csv parse/tokenize step (see build_index.py --workers)")
    args = parser.parse_args()

    build_index.main(
        argv=["--corpora", args.corpora, "--limit", args.limit, "--workers", args.workers])

    data = pd.read_csv(args.input)

    result = run_cocoa_experiment(
        data=data,
        k_c=10,  # matches COCOABaseline's defaults, see baseline.py
        k_t=50,
        query_column=args.query_column,
        target_column=args.target_column,
        db_config="config/cocoa_duckdb_config.json",
        db_profile="real",
    )

    print(result.data.head())

if __name__ == '__main__':
    main()
