#!/usr/bin/env python3
"""
Main training entrypoint for SAGE Qwen2.5-VL-3B LoRA Fine-Tuning.
Supports 10-hour compute budget enforcement, step-wise resumable checkpointing,
and complete capstone review artifact generation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Mitigate CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils import load_config, seed_everything, ensure_dirs, save_json
from src.model import setup_model_and_lora
from src.dataset import SAGEManifestDataset, QwenDataCollator, BadSampleLogger
from src.trainer import SAGETrainer
from src.evaluation import evaluate_model, generate_final_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SAGE.train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL with LoRA on SAGE dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from.")
    parser.add_argument("--smoke-test", action="store_true", help="Run 100-sample smoke test and exit.")
    parser.add_argument("--data-dir", type=str, default=None, help="Override path to SAGE Parquet shards.")
    parser.add_argument("--model-path", type=str, default=None, help="Override path to base model weights.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size per device.")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation steps.")
    parser.add_argument("--quantization", type=str, default=None, choices=["none", "4bit", "8bit"], help="Quantization mode (use 4bit if low VRAM).")
    parser.add_argument("--max-pixels", type=int, default=None, help="Max image pixel resolution for vision encoder.")
    parser.add_argument("--max-hours", type=float, default=None, help="Override maximum training budget (hours).")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Stratified subset of training manifest (e.g. 15000).")
    parser.add_argument("--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing (faster when VRAM allows).")
    return parser.parse_args()


def main():
    args = parse_args()

    # If smoke test flag is passed, run smoke test suite directly
    if args.smoke_test:
        from scripts.smoke_test import run_smoke_test
        run_smoke_test(args.config)
        return

    # Load configuration
    overrides = {}
    if args.data_dir:
        overrides["data.data_dir"] = args.data_dir
    if args.model_path:
        overrides["model.path"] = args.model_path
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    if args.batch_size:
        overrides["training.batch_size"] = args.batch_size
    if args.grad_accum:
        overrides["training.gradient_accumulation_steps"] = args.grad_accum
    if args.quantization:
        overrides["model.quantization"] = args.quantization
    if args.max_pixels:
        overrides["model.max_pixels"] = args.max_pixels
    if args.max_hours:
        overrides["training.max_training_hours"] = args.max_hours
    if args.no_grad_ckpt:
        overrides["training.gradient_checkpointing"] = False

    config = load_config(args.config, overrides=overrides)
    seed = config.get("seed", 42)
    seed_everything(seed)

    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    paths_cfg = config.get("paths", {})

    manifest_dir = data_cfg.get("manifest_dir", "artifacts/data")
    train_manifest_path = os.path.join(manifest_dir, "train_manifest.parquet")
    val_manifest_path = os.path.join(manifest_dir, "val_manifest.parquet")
    test_manifest_path = os.path.join(manifest_dir, "test_manifest.parquet")

    # Fallback to root artifacts if not in artifacts/data
    if not os.path.exists(train_manifest_path) and os.path.exists("artifacts/train_manifest.parquet"):
        train_manifest_path = "artifacts/train_manifest.parquet"
        val_manifest_path = "artifacts/val_manifest.parquet"
        test_manifest_path = "artifacts/test_manifest.parquet"

    # Verify manifest existence
    if not os.path.exists(train_manifest_path):
        raise FileNotFoundError(
            f"Train manifest not found at '{train_manifest_path}'. "
            "Please run 'python scripts/prepare_manifest.py' first to build the balanced split manifests."
        )

    logger.info(f"Loading training manifest from {train_manifest_path}...")
    train_df = pd.read_parquet(train_manifest_path)
    val_df = pd.read_parquet(val_manifest_path)

    # Stratified subset for compute budget fitting
    max_train_samples = args.max_train_samples
    if max_train_samples and len(train_df) > max_train_samples:
        target_col = data_cfg.get("target_column", "canonical_disease")
        logger.info(f"Subsetting training manifest from {len(train_df):,} to {max_train_samples:,} stratified samples...")
        # Per-class proportional sampling
        class_counts = train_df[target_col].value_counts()
        sampled_parts = []
        rng = np.random.RandomState(seed)
        for cls, group in train_df.groupby(target_col):
            n_keep = max(1, int(round(len(group) / len(train_df) * max_train_samples)))
            sampled_parts.append(group.sample(n=min(n_keep, len(group)), random_state=rng.randint(0, 9999)))
        train_df = pd.concat(sampled_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        # Trim to exact target if slightly over
        if len(train_df) > max_train_samples:
            train_df = train_df.sample(n=max_train_samples, random_state=seed).reset_index(drop=True)
        logger.info(f"Subset complete: {len(train_df):,} training samples retained across {train_df[target_col].nunique()} disease classes.")

    logger.info(f"Training on {len(train_df):,} samples | Validation on {len(val_df):,} samples.")

    # Model & Processor Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_adapter_dir = args.resume if args.resume else None
    model, processor = setup_model_and_lora(
        config=config,
        is_training=True,
        adapter_checkpoint_path=resume_adapter_dir,
    )

    # Datasets & Collators
    bad_logger = BadSampleLogger("artifacts/bad_samples.csv")
    train_dataset = SAGEManifestDataset(
        manifest_df=train_df,
        data_dir=data_cfg.get("data_dir"),
        bad_sample_logger=bad_logger,
    )
    val_dataset = SAGEManifestDataset(
        manifest_df=val_df,
        data_dir=data_cfg.get("data_dir"),
        bad_sample_logger=bad_logger,
    )

    train_collator = QwenDataCollator(processor=processor, is_training=True)
    val_collator = QwenDataCollator(processor=processor, is_training=False)

    num_workers = int(data_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        collate_fn=train_collator,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("eval_batch_size", 8)),
        shuffle=False,
        collate_fn=val_collator,
        num_workers=min(2, num_workers),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )

    # Initialize Trainer
    trainer = SAGETrainer(
        model=model,
        processor=processor,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        device=device,
        resume_checkpoint_dir=args.resume,
    )

    # Launch Training
    train_stats = trainer.train()

    # Final Comprehensive Evaluation
    logger.info("Running post-training validation evaluation on best checkpoint...")
    best_checkpoint_dir = os.path.join(paths_cfg.get("checkpoints_dir", "checkpoints"), "best")
    if os.path.exists(best_checkpoint_dir):
        logger.info(f"Loading best checkpoint from '{best_checkpoint_dir}' for final evaluation...")
        best_model, _ = setup_model_and_lora(
            config=config,
            is_training=False,
            adapter_checkpoint_path=best_checkpoint_dir,
        )
    else:
        best_model = model

    val_metrics, _ = evaluate_model(
        model=best_model,
        processor=processor,
        dataloader=val_loader,
        device=device,
        max_samples=None,
        metrics_dir="artifacts/metrics",
        failures_dir="artifacts/failures",
    )

    # Evaluate on Test Manifest if present
    test_metrics = {}
    if os.path.exists(test_manifest_path):
        logger.info("Evaluating on held-out test manifest...")
        test_df = pd.read_parquet(test_manifest_path)
        test_dataset = SAGEManifestDataset(
            manifest_df=test_df,
            data_dir=data_cfg.get("data_dir"),
            bad_sample_logger=bad_logger,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(train_cfg.get("eval_batch_size", 8)),
            shuffle=False,
            collate_fn=val_collator,
            num_workers=min(2, num_workers),
        )
        test_metrics, _ = evaluate_model(
            model=best_model,
            processor=processor,
            dataloader=test_loader,
            device=device,
            max_samples=None,
            metrics_dir="artifacts/metrics/test",
            failures_dir="artifacts/failures/test",
        )
        train_stats["total_test_samples"] = len(test_df)

    # Generate Final Capstone Report
    generate_final_report(
        config=config,
        train_stats=train_stats,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        output_dir="artifacts",
    )

    # Flush bad samples log
    bad_logger.flush()
    print(f"\nTraining completed. Total bad/corrupted samples intercepted: {bad_logger.bad_samples_count}")
    print("=" * 65)
    print(" ALL TRAINING & EVALUATION ARTIFACTS READY FOR REVIEW")
    print("=" * 65)


if __name__ == "__main__":
    main()
