#!/usr/bin/env python3
"""
Standalone evaluation script for SAGE checkpoints on Validation or Held-out Test manifests.
Generates complete diagnostic classification reports, crop breakdowns, confusion matrices,
and failure analysis datasets.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import load_config, seed_everything, ensure_dirs
from src.model import setup_model_and_lora
from src.dataset import SAGEManifestDataset, QwenDataCollator, BadSampleLogger
from src.evaluation import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SAGE.eval")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAGE Qwen2.5-VL checkpoint.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best", help="Path to LoRA checkpoint.")
    parser.add_argument("--manifest", type=str, default=None, help="Direct path to manifest parquet file.")
    parser.add_argument("--split", type=str, choices=["test", "val", "train"], default="test", help="Dataset split to evaluate.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of evaluation samples.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generative evaluation.")
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics", help="Directory to save metric artifacts.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = config.get("seed", 42)
    seed_everything(seed)

    data_cfg = config.get("data", {})
    manifest_dir = data_cfg.get("manifest_dir", "artifacts/data")

    # Locate target manifest
    manifest_path = args.manifest
    if not manifest_path:
        manifest_path = os.path.join(manifest_dir, f"{args.split}_manifest.parquet")
        if not os.path.exists(manifest_path):
            manifest_path = os.path.join("artifacts", f"{args.split}_manifest.parquet")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")

    logger.info(f"Loading evaluation dataset from {manifest_path}...")
    df = pd.read_parquet(manifest_path)
    logger.info(f"Evaluating {len(df):,} samples...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Model with LoRA Checkpoint
    logger.info(f"Loading model with adapter weights from {args.checkpoint}...")
    model, processor = setup_model_and_lora(
        config=config,
        is_training=False,
        adapter_checkpoint_path=args.checkpoint,
    )

    bad_logger = BadSampleLogger("artifacts/bad_samples.csv")
    dataset = SAGEManifestDataset(
        manifest_df=df,
        data_dir=data_cfg.get("data_dir"),
        bad_sample_logger=bad_logger,
    )
    collator = QwenDataCollator(processor=processor, is_training=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    metrics_dir = args.output_dir
    failures_dir = os.path.join(metrics_dir, "failures")
    ensure_dirs(metrics_dir, failures_dir)

    metrics, _ = evaluate_model(
        model=model,
        processor=processor,
        dataloader=dataloader,
        device=device,
        max_samples=args.max_samples,
        metrics_dir=metrics_dir,
        failures_dir=failures_dir,
    )

    print("\n" + "=" * 65)
    print(" SAGE EVALUATION RESULTS")
    print("=" * 65)
    print(f" Split:                     {args.split.upper()}")
    print(f" Exact Match Accuracy:      {metrics.get('exact_match_accuracy', 0.0):.4f}")
    print(f" Normalized Accuracy:       {metrics.get('normalized_accuracy', 0.0):.4f}")
    print(f" Macro Precision:           {metrics.get('macro_precision', 0.0):.4f}")
    print(f" Macro Recall:              {metrics.get('macro_recall', 0.0):.4f}")
    print(f" Macro F1 (Primary):        {metrics.get('macro_f1', 0.0):.4f}")
    print(f" Weighted F1:               {metrics.get('weighted_f1', 0.0):.4f}")
    print("=" * 65)
    print(f"Artifacts saved in {metrics_dir}/")


if __name__ == "__main__":
    main()
