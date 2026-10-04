#!/usr/bin/env python3
"""
Prepares deterministic, class-balanced manifests and Train / Val / Test splits for SAGE.
Reads only metadata columns (ignoring heavy image bytes during manifest generation)
to maintain a minimal RAM footprint.
"""

import os
import sys
import glob
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from datasets import load_dataset

from src.utils import load_config, ensure_dirs, save_json, seed_everything
from src.prompts import is_valid_field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SAGE.manifest")

# Metadata columns (excluding heavy 'image' bytes)
METADATA_COLUMNS = [
    "crop",
    "disease",
    "canonical_disease",
    "plant_organ",
    "visual_symptoms",
    "pathogen",
    "disease_type",
    "symptom_source",
    "symptom_quote",
    "filename",
    "raw_label",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create balanced train/val/test manifests for SAGE.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--data-dir", type=str, default="", help="Path to local Parquet shards.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Target balanced sample size.")
    parser.add_argument("--min-class-samples", type=int, default=None, help="Keep all if count <= this.")
    parser.add_argument("--max-class-samples", type=int, default=None, help="Cap dominant classes at this.")
    return parser.parse_args()


def load_metadata_from_local_shards(parquet_files: List[str]) -> pd.DataFrame:
    """Reads metadata columns from local Parquet files, recording shard path and row index."""
    logger.info(f"Scanning {len(parquet_files)} local Parquet shard(s)...")
    records = []

    for shard_idx, shard_path in enumerate(parquet_files):
        pf = pq.ParquetFile(shard_path)
        available_cols = [c for c in METADATA_COLUMNS if c in pf.schema.names]
        table = pf.read(columns=available_cols)
        df_shard = table.to_pandas()
        
        df_shard["shard_path"] = shard_path
        df_shard["row_idx_in_shard"] = np.arange(len(df_shard), dtype=np.int32)
        records.append(df_shard)

    full_df = pd.concat(records, ignore_index=True)
    logger.info(f"Loaded {len(full_df):,} total metadata rows across shards.")
    return full_df


def load_metadata_from_hf(dataset_name: str, cache_dir: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Downloads/verifies Parquet shards from Hugging Face Hub using snapshot_download,
    then scans only the metadata columns across shards to build manifests with exact shard pointers.
    """
    from huggingface_hub import snapshot_download
    logger.info(f"Fetching Parquet shard metadata from Hugging Face Hub: '{dataset_name}'...")
    local_dir = snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        allow_patterns=["*.parquet", "**/*.parquet"],
        cache_dir=cache_dir,
    )
    parquet_files = sorted(glob.glob(os.path.join(local_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in downloaded repo '{dataset_name}' at {local_dir}")
    
    logger.info(f"Downloaded/located {len(parquet_files)} Parquet shard(s) at {local_dir}")
    df = load_metadata_from_local_shards(parquet_files)
    return df, parquet_files


def balance_classes(
    df: pd.DataFrame,
    target_column: str,
    min_samples: int,
    max_samples: int,
    max_total_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Performs per-class balanced sampling:
    - Retains all examples if count <= min_samples
    - Caps dominant classes at max_samples
    - Samples deterministically with seed
    """
    logger.info(f"Balancing by '{target_column}' (min_cap={min_samples}, max_cap={max_samples})...")
    
    # Ensure target column is clean
    df[target_column] = df[target_column].fillna("Unknown").astype(str).str.strip()
    
    selected_indices = []
    class_groups = df.groupby(target_column)

    for cls_name, group in class_groups:
        n = len(group)
        if n <= min_samples:
            # Retain all examples for rare classes
            selected_indices.extend(group.index.tolist())
        else:
            sample_size = min(n, max_samples)
            sampled = group.sample(n=sample_size, random_state=seed)
            selected_indices.extend(sampled.index.tolist())

    balanced_df = df.loc[selected_indices].copy()

    # If exceeding max_total_samples, uniformly sample down slightly while preserving classes
    if len(balanced_df) > max_total_samples:
        logger.info(f"Downsampling from {len(balanced_df):,} to target max {max_total_samples:,}...")
        balanced_df = balanced_df.sample(n=max_total_samples, random_state=seed).reset_index(drop=True)

    return balanced_df.reset_index(drop=True)


def create_splits(
    df: pd.DataFrame,
    target_column: str,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates deterministic 90% Train / 5% Val / 5% Test splits stratified by canonical_disease."""
    logger.info(f"Creating Train ({(1-val_frac-test_frac)*100:.0f}%) / Val ({val_frac*100:.0f}%) / Test ({test_frac*100:.0f}%) splits...")
    
    # Identify classes with at least 3 samples for stratified splitting
    counts = df[target_column].value_counts()
    valid_classes = counts[counts >= 3].index
    
    df_strat = df[df[target_column].isin(valid_classes)].copy()
    df_rare = df[~df[target_column].isin(valid_classes)].copy()

    # Stratified split for main classes
    total_eval_frac = val_frac + test_frac
    train_idx, eval_idx = train_test_split(
        df_strat.index,
        test_size=total_eval_frac,
        stratify=df_strat[target_column],
        random_state=seed,
    )
    
    df_train_strat = df_strat.loc[train_idx]
    df_eval = df_strat.loc[eval_idx]

    # Split eval equally into Val and Test
    val_ratio = val_frac / total_eval_frac
    val_idx, test_idx = train_test_split(
        df_eval.index,
        test_size=(1.0 - val_ratio),
        stratify=df_eval[target_column],
        random_state=seed,
    )
    
    df_val = df_eval.loc[val_idx].copy()
    df_test = df_eval.loc[test_idx].copy()

    # Rare classes: put into train to maximize disease coverage
    df_train = pd.concat([df_train_strat, df_rare], ignore_index=True)

    # Shuffle deterministically
    df_train = df_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_val = df_val.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_test = df_test.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df_train, df_val, df_test


def compute_statistics_report(
    total_source_rows: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    output_dir: str = "artifacts",
) -> Dict[str, Any]:
    """Generates dataset statistics JSON and class distribution CSV."""
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    class_counts = combined_df[target_column].value_counts()
    
    stats = {
        "total_source_rows": total_source_rows,
        "selected_total_rows": len(combined_df),
        "selected_train_rows": len(train_df),
        "selected_val_rows": len(val_df),
        "selected_test_rows": len(test_df),
        "num_canonical_diseases": int(combined_df[target_column].nunique()),
        "num_crops": int(combined_df["crop"].nunique()) if "crop" in combined_df else 0,
        "median_samples_per_class": float(class_counts.median()),
        "min_samples_per_class": int(class_counts.min()),
        "max_samples_per_class": int(class_counts.max()),
        "rows_with_visual_symptoms": int(combined_df["visual_symptoms"].apply(is_valid_field).sum()),
        "rows_with_plant_organ": int(combined_df["plant_organ"].apply(is_valid_field).sum()),
        "rows_with_pathogen": int(combined_df["pathogen"].apply(is_valid_field).sum()),
        "rows_with_disease_type": int(combined_df["disease_type"].apply(is_valid_field).sum()),
    }

    # Save JSON report
    save_json(stats, os.path.join(output_dir, "data_statistics.json"))

    # Save class distribution CSV
    dist_df = pd.DataFrame({
        "canonical_disease": class_counts.index,
        "total_samples": class_counts.values,
        "train_samples": train_df[target_column].value_counts().reindex(class_counts.index, fill_value=0).values,
        "val_samples": val_df[target_column].value_counts().reindex(class_counts.index, fill_value=0).values,
        "test_samples": test_df[target_column].value_counts().reindex(class_counts.index, fill_value=0).values,
    })
    dist_df.to_csv(os.path.join(output_dir, "class_distribution.csv"), index=False)

    return stats


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = config.get("seed", 42)
    seed_everything(seed)

    data_cfg = config.get("data", {})
    target_column = data_cfg.get("target_column", "canonical_disease")
    min_class_samples = args.min_class_samples or data_cfg.get("min_class_samples", 20)
    max_class_samples = args.max_class_samples or data_cfg.get("max_class_samples", 400)
    max_train_samples = args.max_train_samples or data_cfg.get("max_train_samples", 200000)
    val_frac = float(data_cfg.get("validation_fraction", 0.05))
    test_frac = float(data_cfg.get("test_fraction", 0.05))
    data_dir = args.data_dir or data_cfg.get("data_dir", "")
    manifest_dir = data_cfg.get("manifest_dir", "artifacts/data")

    ensure_dirs("artifacts", manifest_dir)

    print("=" * 65)
    print(" SAGE DATASET MANIFEST GENERATION & BALANCING")
    print("=" * 65)

    # 1. Locate and read metadata
    parquet_files = []
    if data_dir and os.path.exists(data_dir):
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))

    if parquet_files:
        df_meta = load_metadata_from_local_shards(parquet_files)
    else:
        df_meta, parquet_files = load_metadata_from_hf(data_cfg.get("dataset_name", "tirtho149/SAGE"))

    total_source = len(df_meta)

    # 2. Balance classes
    balanced_df = balance_classes(
        df=df_meta,
        target_column=target_column,
        min_samples=min_class_samples,
        max_samples=max_class_samples,
        max_total_samples=int(max_train_samples / (1.0 - val_frac - test_frac)),
        seed=seed,
    )

    # 3. Create Train / Val / Test splits
    train_df, val_df, test_df = create_splits(
        df=balanced_df,
        target_column=target_column,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    # 4. Save Manifest Parquets (without duplicate image bytes!)
    # Save both in artifacts/data/ and artifacts/ for standard compatibility
    for out_folder in [manifest_dir, "artifacts"]:
        train_df.to_parquet(os.path.join(out_folder, "train_manifest.parquet"), index=False)
        val_df.to_parquet(os.path.join(out_folder, "val_manifest.parquet"), index=False)
        test_df.to_parquet(os.path.join(out_folder, "test_manifest.parquet"), index=False)

    # 5. Compute and save statistics report
    stats = compute_statistics_report(
        total_source_rows=total_source,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_column=target_column,
        output_dir="artifacts",
    )

    print("\n" + "=" * 65)
    print(" SAGE DATASET REPORT")
    print("=" * 65)
    print(f" Total Source Rows:          {stats['total_source_rows']:,}")
    print(f" Selected Training Rows:     {stats['selected_train_rows']:,}")
    print(f" Selected Validation Rows:   {stats['selected_val_rows']:,}")
    print(f" Selected Test Rows:         {stats['selected_test_rows']:,}")
    print(f" Number of Canonical Diseases: {stats['num_canonical_diseases']}")
    print(f" Number of Crops:            {stats['num_crops']}")
    print(f" Median Samples/Class:       {stats['median_samples_per_class']}")
    print(f" Min Samples/Class:          {stats['min_samples_per_class']}")
    print(f" Max Samples/Class:          {stats['max_samples_per_class']}")
    print(f" Rows with Visual Symptoms:  {stats['rows_with_visual_symptoms']:,}")
    print(f" Rows with Plant Organ:      {stats['rows_with_plant_organ']:,}")
    print(f" Rows with Pathogen:         {stats['rows_with_pathogen']:,}")
    print(f" Rows with Disease Type:     {stats['rows_with_disease_type']:,}")
    print("=" * 65)
    print(f"Saved manifests to '{manifest_dir}/' and 'artifacts/'")


if __name__ == "__main__":
    main()
