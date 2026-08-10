"""
Locates the specific file(s) under --corpora (same --limit as the
test_compare_index.py run that reported the mismatch) containing a column
named --column, and prints enough detail to see exactly why the original
and this adaptation disagree on it.
"""
import argparse
import os

import pandas as pd

from src.build_index import tokenize_cell
from test_compare_index import compare_column, orig_create_order_index
from src.DataAugmentation import create_index as new_create_index

parser = argparse.ArgumentParser()
parser.add_argument("--corpora", required=True)
parser.add_argument("--limit", type=int, default=100)
parser.add_argument("--column", required=True)
args = parser.parse_args()

csv_paths = sorted(
    os.path.join(root, f)
    for root, _, files in os.walk(args.corpora)
    for f in files if f.endswith(".csv")
)[: args.limit]

for path in csv_paths:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        continue
    if args.column not in df.columns:
        continue

    print(f"\n=== {path} ===")
    raw_values = df[args.column].tolist()
    print("num rows:", len(raw_values))
    print("num NaN/missing (raw):", sum(1 for v in raw_values if pd.isna(v)))
    print("first 20 raw values:", raw_values[:20])

    tokenized_values = df[args.column].fillna("").apply(tokenize_cell).tolist()
    print("num empty tokens:", sum(1 for v in tokenized_values if v == ""))
    print("num distinct tokens:", len(set(tokenized_values)))
    print("first 20 tokenized values:", tokenized_values[:20])

    result = compare_column(tokenized_values)
    print("compare_column result:", result)

    orig_min, orig_order, orig_binary = orig_create_order_index(list(tokenized_values))
    new_min, new_order, new_binary = new_create_index(list(tokenized_values))
    print("orig min_index/order_list[:15]/binary_list[:15]:", orig_min, list(orig_order)[:15], list(orig_binary)[:15])
    print("new  min_index/order_list[:15]/binary_list[:15]:", new_min, list(new_order)[:15], list(new_binary)[:15])
