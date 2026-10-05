"""
Generative evaluation engine, failure analysis, and final report generation for SAGE Qwen2.5-VL.
Parses structured model diagnoses, calculates comprehensive metrics, and creates capstone-ready reports.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.prompts import parse_generated_response, normalize_disease_name
from src.metrics import compute_diagnostic_metrics
from src.utils import ensure_dirs, save_json

logger = logging.getLogger("SAGE.evaluation")


def evaluate_model(
    model: torch.nn.Module,
    processor: Any,
    dataloader: Any,
    device: torch.device,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 128,
    metrics_dir: str = "artifacts/metrics",
    failures_dir: str = "artifacts/failures",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates Qwen2.5-VL generatively across the evaluation dataloader.
    Parses outputs, computes full metrics, saves failure cases, and plots error samples.
    """
    ensure_dirs(metrics_dir, failures_dir)
    model.eval()

    all_predictions: List[str] = []
    all_ground_truths: List[str] = []
    all_crops: List[str] = []
    failure_records: List[Dict[str, Any]] = []
    evaluation_records: List[Dict[str, Any]] = []

    sample_count = 0
    progress_bar = tqdm(dataloader, desc="Generative Evaluation", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            metadata_list = batch.get("metadata", [])
            batch_images = batch.get("images", [])

            # Move inputs to target device
            model_inputs = {}
            for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
                if k in batch and isinstance(batch[k], torch.Tensor):
                    model_inputs[k] = batch[k].to(device)

            # Generate structured response
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            # Extract generated tokens (trim prompt input_ids)
            input_ids = model_inputs["input_ids"]
            trimmed_generated_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            generated_texts = processor.batch_decode(
                trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for i, text in enumerate(generated_texts):
                meta = metadata_list[i] if i < len(metadata_list) else {}
                parsed = parse_generated_response(text)
                
                gt_disease = str(meta.get("canonical_disease") or meta.get("disease", "Unknown"))
                pred_disease = parsed.get("diagnosis", "")
                crop = str(meta.get("crop", "Unknown"))

                all_predictions.append(pred_disease)
                all_ground_truths.append(gt_disease)
                all_crops.append(crop)

                norm_pred = normalize_disease_name(pred_disease)
                norm_gt = normalize_disease_name(gt_disease)
                is_correct = (norm_pred == norm_gt) and (len(norm_pred) > 0)

                record = {
                    "filename": meta.get("filename", f"sample_{sample_count}"),
                    "crop": crop,
                    "ground_truth": gt_disease,
                    "predicted": pred_disease,
                    "predicted_type": parsed.get("disease_type", ""),
                    "predicted_organ": parsed.get("plant_organ", ""),
                    "predicted_symptoms": parsed.get("visual_symptoms", ""),
                    "predicted_pathogen": parsed.get("pathogen", ""),
                    "plant_organ": meta.get("plant_organ", ""),
                    "visual_symptoms": meta.get("visual_symptoms", ""),
                    "pathogen": meta.get("pathogen", ""),
                    "is_correct": is_correct,
                    "raw_output": text,
                }
                evaluation_records.append(record)

                if not is_correct:
                    failure_records.append(record)

                sample_count += 1
                if max_samples and sample_count >= max_samples:
                    break

            if max_samples and sample_count >= max_samples:
                break

    # Calculate complete diagnostic metrics
    metrics = compute_diagnostic_metrics(
        predictions=all_predictions,
        ground_truths=all_ground_truths,
        crops=all_crops,
        output_dir=metrics_dir,
    )

    # Save failures CSV
    failures_csv_path = os.path.join(failures_dir, "failures.csv")
    if failure_records:
        df_fail = pd.DataFrame(failure_records)
        df_fail.to_csv(failures_csv_path, index=False)
        # Also copy to root artifacts for direct review
        df_fail.to_csv(os.path.join("artifacts", "failures.csv"), index=False)
        print(f"Logged {len(failure_records)} failure cases to {failures_csv_path}")

    return metrics, evaluation_records


def generate_final_report(
    config: Dict[str, Any],
    train_stats: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]] = None,
    output_dir: str = "artifacts",
) -> None:
    """
    Generates capstone-ready final report in JSON and GitHub-flavored Markdown.
    """
    ensure_dirs(output_dir)
    test_metrics = test_metrics or {}

    report_data = {
        "project": "SAGE — Scalable Agentic Grounded Evaluation for Crop Disease Diagnosis",
        "model": config.get("model", {}).get("name", "Qwen2.5-VL-3B-Instruct"),
        "lora_config": config.get("lora", {}),
        "dataset_name": config.get("data", {}).get("dataset_name", "tirtho149/SAGE"),
        "training_statistics": train_stats,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    # Save JSON
    save_json(report_data, os.path.join(output_dir, "final_report.json"))

    # Generate Markdown
    md_content = f"""# SAGE Qwen2.5-VL-3B LoRA Training — Final Capstone Report

## 1. Executive Summary & Model Overview
- **Base Vision-Language Model**: `{config.get('model', {}).get('name', 'Qwen2.5-VL-3B-Instruct')}`
- **Training Paradigm**: Parameter-Efficient Fine-Tuning (PEFT / LoRA)
- **Primary Learning Target**: `canonical_disease` (SAGE Canonical Taxonomy)
- **Total Training Samples**: `{train_stats.get('total_train_samples', 'N/A'):,}`
- **Total Validation Samples**: `{train_stats.get('total_val_samples', 'N/A'):,}`
- **Total Test Samples**: `{train_stats.get('total_test_samples', 'N/A'):,}`

---

## 2. LoRA Architecture
- **LoRA Rank (r)**: `{config.get('lora', {}).get('r', 32)}`
- **LoRA Alpha**: `{config.get('lora', {}).get('alpha', 64)}`
- **LoRA Dropout**: `{config.get('lora', {}).get('dropout', 0.05)}`
- **Target Modules**: `{", ".join(config.get('lora', {}).get('target_modules', []))}`
- **Trainable Parameters**: `{train_stats.get('trainable_parameters', 'N/A') if isinstance(train_stats.get('trainable_parameters'), str) else f"{train_stats.get('trainable_parameters', 0):,}"}`
- **Total Parameters**: `{train_stats.get('total_parameters', 'N/A') if isinstance(train_stats.get('total_parameters'), str) else f"{train_stats.get('total_parameters', 0):,}"}`
- **Trainable Percentage**: `{train_stats.get('trainable_pct', 'N/A')}%`

---

## 3. Training Runtime & Budget Compliance
- **Total Elapsed Training Time**: `{train_stats.get('elapsed_time_str', 'N/A')}`
- **Epochs Completed**: `{train_stats.get('epochs_completed', 'N/A')}` / `{config.get('training', {}).get('epochs', 2)}`
- **Best Validation Epoch**: `{train_stats.get('best_epoch', 'N/A')}`
- **Best Validation Step**: `{train_stats.get('best_step', 'N/A')}`
- **Best Validation Macro F1**: `{train_stats.get('best_macro_f1', 'N/A')}`

---

## 4. Diagnostic Performance (Validation & Test)

| Metric | Validation Set | Final Test Set |
| :--- | :---: | :---: |
| **Exact Match Accuracy** | {val_metrics.get('exact_match_accuracy', 'N/A')} | {test_metrics.get('exact_match_accuracy', 'N/A')} |
| **Normalized Disease Accuracy** | {val_metrics.get('normalized_accuracy', 'N/A')} | {test_metrics.get('normalized_accuracy', 'N/A')} |
| **Macro Precision** | {val_metrics.get('macro_precision', 'N/A')} | {test_metrics.get('macro_precision', 'N/A')} |
| **Macro Recall** | {val_metrics.get('macro_recall', 'N/A')} | {test_metrics.get('macro_recall', 'N/A')} |
| **Macro F1 Score (Primary)** | **{val_metrics.get('macro_f1', 'N/A')}** | **{test_metrics.get('macro_f1', 'N/A')}** |
| **Weighted F1 Score** | {val_metrics.get('weighted_f1', 'N/A')} | {test_metrics.get('weighted_f1', 'N/A')} |

---

## 5. Capstone Review Artifacts
- Per-class classification metrics: `artifacts/metrics/classification_report.csv`
- Crop-level accuracy & F1: `artifacts/metrics/crop_metrics.csv`
- Top error confusions: `artifacts/metrics/top_confusions.csv`
- Error analysis records: `artifacts/failures.csv`
- Confusion matrix plot: `artifacts/metrics/confusion_matrix.png`
"""
    with open(os.path.join(output_dir, "final_report.md"), "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Generated final reports at {output_dir}/final_report.json and final_report.md")
