#!/usr/bin/env python3
"""
Inspects the SAGE agricultural disease dataset (Hugging Face or local Parquet directory).
Analyzes schema, column distribution, field coverage, and prints sample formatted prompts.
"""

import os
import sys
import glob
import argparse
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi

from src.utils import load_config, ensure_dirs, save_json
from src.prompts import format_user_prompt, format_assistant_target, is_valid_field


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect SAGE dataset structure and fields.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--data-dir", type=str, default="", help="Local path to SAGE Parquet files.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of sample records to inspect.")
    return parser.parse_args()


def inspect_local_parquets(parquet_files, num_samples):
    print(f"Found {len(parquet_files)} local Parquet shard(s).")
    total_rows = 0
    sample_rows = []
    columns = []

    for idx, p_file in enumerate(parquet_files):
        pf = pq.ParquetFile(p_file)
        meta = pf.metadata
        total_rows += meta.num_rows
        if idx == 0:
            columns = pf.schema.names
            # Read first few rows without image bytes to print schema
            table = pf.read_row_group(0)
            df_sample = table.to_pandas()
            for _, r in df_sample.head(num_samples).iterrows():
                row_dict = r.to_dict()
                if "image" in row_dict:
                    row_dict["image"] = f"<image bytes: {type(r['image'])}>"
                sample_rows.append(row_dict)

    return total_rows, columns, sample_rows


def inspect_hf_dataset(dataset_name, num_samples):
    print(f"Connecting to Hugging Face dataset '{dataset_name}' (streaming mode)...")
    ds = load_dataset(dataset_name, split="train", streaming=True)
    sample_rows = []
    columns = []

    for idx, item in enumerate(ds):
        if idx >= num_samples:
            break
        if idx == 0:
            columns = list(item.keys())
        r = dict(item)
        if "image" in r:
            r["image"] = f"<image object: {type(r['image'])}>"
        sample_rows.append(r)

    return columns, sample_rows


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset_name", "tirtho149/SAGE")
    data_dir = args.data_dir or data_cfg.get("data_dir", "")

    ensure_dirs("artifacts")
    print("=" * 65)
    print(" SAGE DATASET INSPECTION")
    print("=" * 65)

    parquet_files = []
    if data_dir and os.path.exists(data_dir):
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))

    total_rows = "Unknown (Streaming HF)"
    if parquet_files:
        total_rows, columns, sample_rows = inspect_local_parquets(parquet_files, args.num_samples)
    else:
        columns, sample_rows = inspect_hf_dataset(dataset_name, args.num_samples)

    print(f"Dataset Source:     {data_dir if parquet_files else dataset_name}")
    print(f"Total Rows:         {total_rows}")
    print(f"Columns Detected:   {columns}")
    print("=" * 65)

    if sample_rows:
        print("\n--- SAMPLE FORMATTED TRAINING CONVERSATION (Row 0) ---")
        row0 = sample_rows[0]
        print("\n[USER INSTRUCTION]:")
        print(format_user_prompt(row0))
        print("\n[ASSISTANT TARGET (GROUND TRUTH)]:")
        print(format_assistant_target(row0))
        print("-" * 65)

    inspection_data = {
        "dataset_source": data_dir if parquet_files else dataset_name,
        "total_rows": total_rows,
        "columns": columns,
        "sample_preview": sample_rows,
    }

    out_path = os.path.join("artifacts", "dataset_inspection.json")
    save_json(inspection_data, out_path)
    print(f"\nInspection report saved to {out_path}")


if __name__ == "__main__":
    main()
