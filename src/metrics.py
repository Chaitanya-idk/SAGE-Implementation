"""
Comprehensive diagnostic evaluation metrics for SAGE:
Calculates Exact Match, Normalized Accuracy, Macro/Weighted F1, Recall, Precision,
Crop-level breakdowns, Top Confusions, and generates Confusion Matrix plots.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.prompts import normalize_disease_name
from src.utils import ensure_dirs, save_json


def compute_diagnostic_metrics(
    predictions: List[str],
    ground_truths: List[str],
    crops: Optional[List[str]] = None,
    output_dir: str = "artifacts/metrics",
) -> Dict[str, Any]:
    """
    Computes full suite of classification and diagnostic metrics:
    - Exact match accuracy
    - Normalized disease accuracy
    - Macro precision, recall, F1
    - Weighted precision, recall, F1
    - Per-class classification report CSV
    - Crop-level metrics CSV
    - Top confusions CSV
    - Confusion matrix PNG
    """
    ensure_dirs(output_dir)

    # Clean and normalize lists
    norm_preds = [normalize_disease_name(p) for p in predictions]
    norm_gts = [normalize_disease_name(gt) for gt in ground_truths]

    # Exact string match (raw)
    exact_acc = float(np.mean([p.strip().lower() == gt.strip().lower() for p, gt in zip(predictions, ground_truths)]))
    
    # Normalized match
    norm_acc = float(accuracy_score(norm_gts, norm_preds))

    # All unique classes across GT and Pred
    all_classes = sorted(list(set(norm_gts) | set(norm_preds)))

    # Macro & Weighted metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        norm_gts, norm_preds, labels=all_classes, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        norm_gts, norm_preds, labels=all_classes, average="weighted", zero_division=0
    )

    metrics_summary = {
        "total_samples": len(predictions),
        "num_classes_in_gt": len(set(norm_gts)),
        "exact_match_accuracy": round(exact_acc, 4),
        "normalized_accuracy": round(norm_acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(p_weighted), 4),
        "weighted_recall": round(float(r_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
    }

    # 1. Per-class report CSV
    report_dict = classification_report(
        norm_gts, norm_preds, labels=all_classes, output_dict=True, zero_division=0
    )
    class_rows = []
    for cls_name, vals in report_dict.items():
        if cls_name not in ("accuracy", "macro avg", "weighted avg"):
            class_rows.append({
                "Class": cls_name,
                "Support": int(vals.get("support", 0)),
                "Precision": round(vals.get("precision", 0.0), 4),
                "Recall": round(vals.get("recall", 0.0), 4),
                "F1": round(vals.get("f1-score", 0.0), 4),
            })
    class_df = pd.DataFrame(class_rows).sort_values(by="Support", ascending=False)
    class_df.to_csv(os.path.join(output_dir, "classification_report.csv"), index=False)

    # 2. Crop-level performance breakdown
    if crops and len(crops) == len(ground_truths):
        crop_rows = []
        df_crop = pd.DataFrame({"crop": crops, "gt": norm_gts, "pred": norm_preds})
        for crop_name, group in df_crop.groupby("crop"):
            if len(group) == 0:
                continue
            c_acc = accuracy_score(group["gt"], group["pred"])
            _, _, c_f1, _ = precision_recall_fscore_support(
                group["gt"], group["pred"], average="macro", zero_division=0
            )
            crop_rows.append({
                "Crop": crop_name,
                "Samples": len(group),
                "Accuracy": round(float(c_acc), 4),
                "Macro F1": round(float(c_f1), 4),
            })
        crop_metrics_df = pd.DataFrame(crop_rows).sort_values(by="Samples", ascending=False)
        crop_metrics_df.to_csv(os.path.join(output_dir, "crop_metrics.csv"), index=False)

    # 3. Top confusion pairs
    confusions = [
        (gt, pred) for gt, pred in zip(norm_gts, norm_preds) if gt != pred
    ]
    conf_counts = Counter(confusions).most_common(50)
    conf_rows = [
        {"Ground Truth": k[0], "Predicted": k[1], "Count": count}
        for k, count in conf_counts
    ]
    pd.DataFrame(conf_rows).to_csv(os.path.join(output_dir, "top_confusions.csv"), index=False)

    # 4. Confusion Matrix Plot (top 20 most frequent GT classes for readability)
    try:
        top_classes = class_df.head(20)["Class"].tolist()
        if len(top_classes) >= 2:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(norm_gts, norm_preds, labels=top_classes)
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=[c[:15] for c in top_classes],
                yticklabels=[c[:15] for c in top_classes],
            )
            plt.title("Top-20 Disease Classes Confusion Matrix", fontsize=14, pad=12)
            plt.xlabel("Predicted Disease", fontsize=12)
            plt.ylabel("Ground Truth Disease", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
            plt.close()
    except Exception as e:
        print(f"Warning: Could not plot confusion matrix: {e}")

    # Save metrics JSON
    save_json(metrics_summary, os.path.join(output_dir, "metrics.json"))
    return metrics_summary
