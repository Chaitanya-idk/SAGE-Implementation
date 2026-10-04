#!/usr/bin/env python3
"""
Batch qualitative demonstration script for SAGE Qwen2.5-VL.
Runs inference across 10-20 held-out test examples and saves structured
predictions, ground truths, and visual symptoms to artifacts/qualitative_predictions.json.
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

from src.utils import load_config, ensure_dirs, save_json
from src.model import setup_model_and_lora
from src.dataset import SAGEManifestDataset, QwenDataCollator
from src.prompts import parse_generated_response, normalize_disease_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SAGE.demo")


def parse_args():
    parser = argparse.ArgumentParser(description="Run qualitative batch inference demo.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best", help="Path to LoRA checkpoint.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest parquet.")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of qualitative samples to test.")
    parser.add_argument("--output", type=str, default="artifacts/qualitative_predictions.json", help="Output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    manifest_dir = data_cfg.get("manifest_dir", "artifacts/data")

    # Locate test manifest
    manifest_path = args.manifest
    if not manifest_path:
        manifest_path = os.path.join(manifest_dir, "test_manifest.parquet")
        if not os.path.exists(manifest_path):
            manifest_path = os.path.join("artifacts", "test_manifest.parquet")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Test manifest not found at {manifest_path}")

    df = pd.read_parquet(manifest_path)
    sample_df = df.sample(n=min(args.num_samples, len(df)), random_state=42).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = setup_model_and_lora(
        config=config,
        is_training=False,
        adapter_checkpoint_path=args.checkpoint,
    )
    model.eval()

    dataset = SAGEManifestDataset(manifest_df=sample_df, data_dir=data_cfg.get("data_dir"))
    collator = QwenDataCollator(processor=processor, is_training=False)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collator)

    results = []
    print(f"\nRunning qualitative batch demo on {len(sample_df)} held-out samples...")

    with torch.no_grad():
        for batch in dataloader:
            metadata_list = batch.get("metadata", [])
            model_inputs = {}
            for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
                if k in batch and isinstance(batch[k], torch.Tensor):
                    model_inputs[k] = batch[k].to(device)

            generated_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=False)
            input_ids = model_inputs["input_ids"]
            trimmed = [out[len(inp):] for inp, out in zip(input_ids, generated_ids)]
            texts = processor.batch_decode(trimmed, skip_special_tokens=True)

            for meta, gen_text in zip(metadata_list, texts):
                parsed = parse_generated_response(gen_text)
                gt = str(meta.get("canonical_disease") or meta.get("disease", "Unknown"))
                pred = parsed.get("diagnosis", "")
                is_correct = (normalize_disease_name(gt) == normalize_disease_name(pred))

                item = {
                    "filename": meta.get("filename", ""),
                    "crop": meta.get("crop", ""),
                    "ground_truth_disease": gt,
                    "predicted_disease": pred,
                    "is_correct": is_correct,
                    "disease_type": parsed.get("disease_type", ""),
                    "plant_organ": parsed.get("plant_organ", "") or meta.get("plant_organ", ""),
                    "visual_symptoms": parsed.get("visual_symptoms", "") or meta.get("visual_symptoms", ""),
                    "pathogen": parsed.get("pathogen", "") or meta.get("pathogen", ""),
                    "raw_output": gen_text.strip(),
                }
                results.append(item)

    ensure_dirs(os.path.dirname(args.output) or ".")
    save_json(results, args.output)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Crop':<12} | {'Ground Truth':<25} | {'Predicted':<25} | {'Match'}")
    print("-" * 80)
    for r in results:
        status = "✓ Correct" if r["is_correct"] else "✗ Mismatch"
        print(f"{r['crop'][:12]:<12} | {r['ground_truth_disease'][:25]:<25} | {r['predicted_disease'][:25]:<25} | {status}")
    print("=" * 80)
    print(f"Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
