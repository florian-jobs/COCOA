"""
Compares the index-building logic of the original COCOA (COCOA_Original/index_generation.py)
against this adaptation (src/build_index.py, src/DataAugmentation.py) on the same input data.

Only the pure, DB-free functions are compared (is_numeric_list, is_numeric_list_with_header,
create_order_index / create_index). generate_order_index (Vertica) is out of scope.
"""

import argparse
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_index import tokenize_cell
from src.DataAugmentation import create_index as new_create_index

# ---------------------------------------------------------------------------
# Vendored, byte-for-byte copy of COCOA_Original/index_generation.py's pure
# (DB-free) functions: is_numeric, is_numeric_list, is_numeric_list_with_header,
# create_order_index.
# ---------------------------------------------------------------------------

def orig_is_numeric(s):
    if s.lower() == 'nan':
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False

def orig_is_numeric_list(l):
    for i in np.arange(len(l)):
        if l[i] is None or l[i] == '':
            l[i] = np.nan
        else:
            l[i] = l[i]

    result = [s for s in l if orig_is_numeric(str(s))]
    return len(result) == len(l)

def orig_is_numeric_list_with_header(l):
    for i in np.arange(len(l)):
        if l[i] is None or l[i] == '':
            l[i] = np.nan
        else:
            l[i] = l[i]

    found_heading = False
    heading_index = -1
    ret = []

    for i in range(len(l)):
        if orig_is_numeric(str(l[i])):
            ret += [float(l[i])]
        elif not found_heading:
            found_heading = True
            heading_index = i
        else:
            return False, -1, []

    return True, heading_index, ret

def orig_create_order_index(values):
    rows = np.arange(0, len(values), 1)
    if orig_is_numeric_list(values):
        for i in np.arange(len(values)):
            if values[i] is None or values[i] == '':
                values[i] = np.nan
        values = [float(i) for i in values]
    else:
        for i in np.arange(len(values)):
            if values[i] is None:
                values[i] = ''
        values = [str(i) for i in values]
    ranks = list(pd.Series(values).rank())

    rows_sorted_based_on_ranks = [x for _, x in sorted(zip(ranks, rows))]
    min_index = rows_sorted_based_on_ranks[0]  # starting point in the order index
    order_list = np.empty(len(rows), dtype=int)
    binary_list = np.empty(len(rows), dtype=str)
    sorted_ranks = np.sort(ranks).copy()
    for i in np.arange(len(rows) - 1):
        order_list[i] = rows_sorted_based_on_ranks[i + 1]
        if sorted_ranks[i] == sorted_ranks[i + 1]:
            binary_list[i] = 'F'
        else:
            binary_list[i] = 'T'
    order_list[len(rows) - 1] = -1  # Maximum value
    binary_list[len(rows) - 1] = -1  # Maximum value

    final_order_list = [x for _, x in
                        sorted(zip(rows_sorted_based_on_ranks, order_list))]  # order list in the order index
    final_binary_list = [x for _, x in
                         sorted(zip(rows_sorted_based_on_ranks, binary_list))]  # binary list in the order index

    return min_index, final_order_list, final_binary_list

def compare_column(tokenized_values):
    """Runs one column's tokenized values through both implementations and reports whether their outputs agree."""
    orig_is_numeric = orig_is_numeric_list(list(tokenized_values)) or \
                      orig_is_numeric_list_with_header(list(tokenized_values))[0]
    orig_min_index, orig_order_list, orig_binary_list = orig_create_order_index(list(tokenized_values))

    new_min_index, new_order_list, new_binary_list = new_create_index(list(tokenized_values))

    return {
        "is_numeric_orig": bool(orig_is_numeric),
        "min_index_orig": int(orig_min_index),
        "min_index_new": int(new_min_index),
        "min_index_match": int(orig_min_index) == int(new_min_index),
        "order_list_match": list(orig_order_list) == list(new_order_list),
        "binary_list_match": list(orig_binary_list) == list(new_binary_list),
    }

def main():
    """CLI entry point: compares every column of every csv under --corpora, prints a summary, and lists any mismatches."""
    startTime = time.time()
    print(f'Starting at {startTime}')
    parser = argparse.ArgumentParser("Compare index-building logic: original vs. this adaptation")
    parser.add_argument("--corpora", default="dataset", help="Directory containing csv's to compare")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of csv's to process")
    args = parser.parse_args()

    csv_paths = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(args.corpora)
        for f in files if f.endswith(".csv")
    )
    if args.limit is not None:
        csv_paths = csv_paths[: args.limit]

    rows = []
    for path in csv_paths:
        df = pd.read_csv(path)
        for colname in df.columns:
            tokenized_values = df[colname].fillna("").apply(tokenize_cell).tolist()
            result = compare_column(tokenized_values)
            result["file"] = os.path.basename(path)
            result["column"] = colname
            rows.append(result)

    endTime = time.time()
    print(f'Elapsed: {endTime - startTime:.2f} seconds')

    report = pd.DataFrame(rows)
    all_match = report["min_index_match"] & report["order_list_match"] & report["binary_list_match"]
    mismatches = report[~all_match]

    print(f"Compared {len(report)} columns across {len(csv_paths)} csv's.")
    print(f"Mismatches: {len(mismatches)}")
    if not mismatches.empty:
        print(mismatches[["file", "column", "min_index_match", "order_list_match", "binary_list_match"]]
              .to_string(index=False))

if __name__ == "__main__":
    main()
