import argparse
import glob
import os

# 1. Usage: ls -l ./nyc/ | grep -c ^d
# 2. uv run python -m tests.test_csv_paths_enumeration.py
# expected: equal output size.

def main():
    # Parser for stating table corpora directory.
    parser = argparse.ArgumentParser(description="Run COCOA indexing.")
    parser.add_argument("--corpora", required=False,
                        help="Directory containing the table corpora. Defaults to dataset/.")
    parser.add_argument("--limit", required=False, help="Only process the first n csv files.")
    args = parser.parse_args()

    # Obtain all csv paths. If --corpora is specified, use that, else use dataset/. Possible error source: empty csv's.
    if args.corpora is not None:
        # print(f"Using corpora directory {args.corpora}").
        csv_paths = sorted(os.path.join(root, file)
                           for root, dirs, files in os.walk(args.corpora)
                           for file in files if
                           file.endswith(".csv"))
    else:
        csv_paths = sorted(
            glob.glob(
                os.path.join("dataset", "*.csv")))

    limit = int(args.limit)
    print(csv_paths[0:limit], "\n", len(csv_paths))

if __name__ == "__main__":
    main()
