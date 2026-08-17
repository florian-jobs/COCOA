"""
End-to-end smoke test: builds the index over a corpus, then runs COCOA on one
input table, printing a preview of the result. Exercises both phases
(src/build_index.py and interface.run_cocoa_experiment) together against a
real corpus - see README for example invocations.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

from interface import run_cocoa_experiment
from src import build_index
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent
_DB_CONFIG = _PROJECT_ROOT / "config" / "cocoa_duckdb_config.json"
_DB_PROFILE = "real"

def main():
    parser = argparse.ArgumentParser("Run Cocoa Baseline Test")
    parser.add_argument("--corpora", required=True, help="Corpora to run")
    parser.add_argument("--limit", required=False, help="Number of csv's to process")
    parser.add_argument("--input", required=True, help="CSV file to run the experiment on, e.g. dataset/movie.csv")
    parser.add_argument("--query_column", required=True, help="Query column to run the experiment on")
    parser.add_argument("--target_column", required=True, help="Target column to run the experiment on")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Force a full index rebuild even if one already exists on disk (matches baseline.py's rebuild_index)")
    args = parser.parse_args()

    startTime = time.time()
    build_argv = ["--corpora", args.corpora]
    if args.limit is not None:
        build_argv += ["--limit", args.limit]

    with open(_DB_CONFIG, "r", encoding="utf-8") as f:
        db_path = _PROJECT_ROOT / json.load(f)["connection"][_DB_PROFILE]["database"]

    # is_build_complete(), not db_path.exists(): db_path is created (and
    # partially populated) as soon as a build starts, so existence alone
    # can't tell an in-progress or crashed build apart from a finished one.
    if args.rebuild_index or not build_index.is_build_complete(db_path):
        build_index.main(argv=build_argv)
    else:
        print(f"Reusing existing index at {db_path} (pass --rebuild-index to force a rebuild)")

    data = pd.read_csv(args.input)
    print("original dataframe head: " + "\n", data.head(20))
    print("original dataframe tail: " + "\n", data.tail(20))
    print("original dataframe columns: " + "\n", data.columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_cocoa_experiment(
            data=data,
            k_c=10,  # matches COCOABaseline's defaults, see baseline.py
            k_t=50,
            query_column=args.query_column,
            target_column=args.target_column,
            db_config="config/cocoa_duckdb_config.json",
            db_profile="real",
        )
    endTime = time.time()

    print(f"Time taken: {endTime - startTime} seconds")
    print("result dataframe head:" + "\n", result.data.head(20))
    print("result dataframe tail: " + "\n", result.data.tail(20))
    print("result dataframe columns:" + "\n", result.data.columns)

if __name__ == '__main__':
    main()
