#!/usr/bin/env python3
"""
High-Speed Offline Image Preprocessor for SAGE.
Extracts and pre-resizes raw images from Parquet shards into local JPEG files (448x448).
Generates an updated manifest pointing directly to local images, eliminating runtime
Parquet decompression and speeding up training by ~3x.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from src.dataset import decode_image
from src.utils import ensure_dirs, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SAGE.preprocess")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and cache SAGE images to disk.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="artifacts/data/train_manifest.parquet",
        help="Path to manifest parquet file to preprocess.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default="artifacts/data/val_manifest.parquet",
        help="Optional path to validation manifest to also preprocess.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/preprocessed_images",
        help="Destination directory for resized images.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples to process (e.g. 15000).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=448,
        help="Target square image dimension (e.g. 448 for 448x448).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG compression quality (1-100). 92 gives ~50KB per image with crisp details.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel worker threads.",
    )
    return parser.parse_args()


def process_and_save_image(
    img_raw: Any,
    dest_path: str,
    target_size: int = 448,
    quality: int = 92,
) -> bool:
    """Decodes, resizes, and saves image to JPEG."""
    try:
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 500:
            return True  # Already cached

        img = decode_image(img_raw)
        # High quality bilinear/bicubic resize to target dimension
        img = img.resize((target_size, target_size), resample=Image.Resampling.BICUBIC)
        img.save(dest_path, format="JPEG", quality=quality, optimize=True)
        return True
    except Exception as e:
        logger.debug(f"Failed to process image for {dest_path}: {e}")
        return False


def preprocess_manifest(
    manifest_path: str,
    output_dir: str,
    split_name: str,
    target_size: int = 448,
    quality: int = 92,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
) -> str:
    """
    Extracts images from Parquet shards grouped by shard to minimize I/O overhead,
    saves them to output_dir, and writes an updated parquet manifest with 'image_path'.
    """
    if not os.path.exists(manifest_path):
        logger.warning(f"Manifest path not found: {manifest_path}. Skipping.")
        return manifest_path

    logger.info(f"Loading manifest from {manifest_path}...")
    df = pd.read_parquet(manifest_path)
    total_original = len(df)

    if max_samples and total_original > max_samples:
        logger.info(f"Subsetting manifest from {total_original:,} to {max_samples:,} samples...")
        target_col = "canonical_disease" if "canonical_disease" in df.columns else df.columns[0]
        # Stratified sampling across disease classes
        class_groups = df.groupby(target_col)
        sampled_parts = []
        for _, group in class_groups:
            n_keep = max(1, int(round(len(group) / total_original * max_samples)))
            sampled_parts.append(group.sample(n=min(n_keep, len(group)), random_state=42))
        df = pd.concat(sampled_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
        if len(df) > max_samples:
            df = df.iloc[:max_samples].copy()
        logger.info(f"Subset ready: {len(df):,} samples retained.")

    split_dir = os.path.join(output_dir, split_name)
    ensure_dirs(split_dir)

    # Prepare image path mappings
    image_paths: List[Optional[str]] = [None] * len(df)
    
    # Check if images are stored in shards or directly in manifest
    has_shard_path = "shard_path" in df.columns or "shard_id" in df.columns
    shard_col = "shard_path" if "shard_path" in df.columns else ("shard_id" if "shard_id" in df.columns else None)
    row_idx_col = "row_idx_in_shard" if "row_idx_in_shard" in df.columns else None

    # Group by shard to read each large Parquet file only ONCE
    if has_shard_path and shard_col:
        logger.info(f"Extracting images from shards grouped by file (split={split_name})...")
        shard_groups = df.groupby(shard_col)
        progress = tqdm(total=len(df), desc=f"Preprocessing {split_name}", dynamic_ncols=True)

        for shard_ref, group in shard_groups:
            shard_path = str(shard_ref)
            if not os.path.exists(shard_path):
                # Try basename match if relative
                base_name = os.path.basename(shard_path)
                if os.path.exists(base_name):
                    shard_path = base_name

            if not os.path.exists(shard_path):
                logger.warning(f"Shard file not found: {shard_path}. Skipping group of {len(group)} rows.")
                progress.update(len(group))
                continue

            try:
                # Read only image column for this shard
                table = pq.read_table(shard_path, columns=["image"])
                image_col_data = table["image"]

                def process_row(args):
                    df_idx, row_in_shard = args
                    dest_file = os.path.join(split_dir, f"{df_idx:06d}.jpg")
                    try:
                        raw_cell = image_col_data[int(row_in_shard)].as_py()
                        success = process_and_save_image(raw_cell, dest_file, target_size, quality)
                        return df_idx, dest_file if success else None
                    except Exception:
                        return df_idx, None

                row_args = []
                for df_idx, row in group.iterrows():
                    r_idx = row[row_idx_col] if row_idx_col else df_idx
                    row_args.append((df_idx, r_idx))

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    for df_idx, path_result in executor.map(process_row, row_args):
                        image_paths[df_idx] = path_result
                        progress.update(1)

            except Exception as e:
                logger.error(f"Error processing shard {shard_path}: {e}")
                progress.update(len(group))

        progress.close()

    elif "image" in df.columns:
        # Images directly inside manifest rows
        logger.info(f"Extracting images directly from manifest rows (split={split_name})...")
        progress = tqdm(total=len(df), desc=f"Preprocessing {split_name}", dynamic_ncols=True)

        def process_direct_row(df_idx):
            dest_file = os.path.join(split_dir, f"{df_idx:06d}.jpg")
            raw_cell = df.at[df_idx, "image"]
            success = process_and_save_image(raw_cell, dest_file, target_size, quality)
            return df_idx, dest_file if success else None

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_direct_row, i) for i in range(len(df))]
            for future in as_completed(futures):
                idx, res_path = future.result()
                image_paths[idx] = res_path
                progress.update(1)

        progress.close()

    # Assign image_path column
    df["image_path"] = image_paths
    valid_count = df["image_path"].notna().sum()
    logger.info(f"Successfully preprocessed {valid_count:,}/{len(df):,} images for {split_name}.")

    # Save output manifest
    manifest_dir = os.path.dirname(manifest_path)
    base_stem = Path(manifest_path).stem
    out_manifest_path = os.path.join(manifest_dir, f"{base_stem}_preprocessed.parquet")
    df.to_parquet(out_manifest_path, index=False)
    logger.info(f"Saved updated manifest to {out_manifest_path}")

    # Calculate total size on disk
    total_bytes = sum(
        os.path.getsize(p) for p in image_paths if p and os.path.exists(p)
    )
    logger.info(f"Disk space used: {total_bytes / (1024 ** 2):.1f} MB in '{split_dir}'")
    return out_manifest_path


def main():
    args = parse_args()
    logger.info("=" * 65)
    logger.info(" SAGE HIGH-SPEED OFFLINE IMAGE PREPROCESSOR")
    logger.info("=" * 65)
    logger.info(f" Target Resolution:  {args.target_size}x{args.target_size} px")
    logger.info(f" JPEG Quality:       {args.jpeg_quality}")
    logger.info(f" Output Directory:   {args.output_dir}")
    logger.info(f" Max Samples:        {args.max_samples or 'ALL'}")
    logger.info("=" * 65)

    # 1. Preprocess Training Manifest
    train_prep = preprocess_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        split_name="train",
        target_size=args.target_size,
        quality=args.jpeg_quality,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )

    # 2. Preprocess Validation Manifest (if exists)
    if args.val_manifest and os.path.exists(args.val_manifest):
        preprocess_manifest(
            manifest_path=args.val_manifest,
            output_dir=args.output_dir,
            split_name="val",
            target_size=args.target_size,
            quality=args.jpeg_quality,
            max_samples=None,
            num_workers=args.num_workers,
        )

    logger.info("Preprocessing complete! You can now start training with maximum throughput.")


if __name__ == "__main__":
    main()
