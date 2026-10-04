"""
Utility functions for SAGE Qwen2.5-VL training pipeline.
Provides hardware detection, seeding, budget monitoring, configuration loading,
and safe filesystem helpers.
"""

import os
import sys
import json
import time
import random
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch

logger = logging.getLogger("SAGE.utils")


def seed_everything(seed: int = 42) -> None:
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Loads YAML configuration file and applies optional key-value overrides."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                # Support dot-notation nested overrides e.g. "training.epochs"
                keys = k.split(".")
                curr = config
                for subkey in keys[:-1]:
                    curr = curr.setdefault(subkey, {})
                curr[keys[-1]] = v
                
    return config


def ensure_dirs(*dirs: str) -> None:
    """Ensures directories exist on disk."""
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Safely saves data to JSON with directory creation."""
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str) -> Any:
    """Loads data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_device_info() -> Dict[str, Any]:
    """Detects available GPU hardware, VRAM, and bfloat16 support."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": "CPU",
        "total_memory_gb": 0.0,
        "bf16_supported": False,
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory
        info["total_memory_gb"] = round(total_mem / (1024 ** 3), 2)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def resolve_dtype(dtype_str: str, bf16_supported: bool) -> torch.dtype:
    """Resolves string representation of dtype to torch.dtype."""
    dtype_str = dtype_str.lower().strip()
    if dtype_str == "auto":
        return torch.bfloat16 if bf16_supported else torch.float16
    elif dtype_str in ("bfloat16", "bf16"):
        return torch.bfloat16
    elif dtype_str in ("float16", "fp16"):
        return torch.float16
    elif dtype_str in ("float32", "fp32"):
        return torch.float32
    else:
        logger.warning(f"Unrecognized dtype '{dtype_str}', defaulting to float16/bfloat16 auto-detection.")
        return torch.bfloat16 if bf16_supported else torch.float16


class BudgetManager:
    """
    Monitors training execution time against a hard wall-clock budget (e.g. 9.0 hours).
    Provides safe early-stopping signals and formatted throughput/ETA metrics.
    """

    def __init__(self, max_hours: float = 9.0, start_time: Optional[float] = None):
        self.max_seconds = max_hours * 3600.0
        self.start_time = start_time if start_time is not None else time.time()
        self.budget_exceeded = False

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def elapsed_hours(self) -> float:
        return self.elapsed_seconds() / 3600.0

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed_seconds())

    def remaining_hours(self) -> float:
        return self.remaining_seconds() / 3600.0

    def is_budget_exceeded(self) -> bool:
        if self.elapsed_seconds() >= self.max_seconds:
            self.budget_exceeded = True
            return True
        return False

    def format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes:02d}m"
        else:
            return f"{minutes:02d}m {secs:02d}s"

    def get_progress_status(self, current_step: int, total_steps: int) -> Dict[str, Any]:
        elapsed = self.elapsed_seconds()
        remaining = self.remaining_seconds()
        pct = (current_step / max(total_steps, 1)) * 100.0
        steps_per_sec = current_step / max(elapsed, 1e-5)
        eta_sec = (total_steps - current_step) / max(steps_per_sec, 1e-5)
        
        return {
            "elapsed_str": self.format_time(elapsed),
            "remaining_budget_str": self.format_time(remaining),
            "eta_str": self.format_time(eta_sec),
            "pct": pct,
            "steps_per_sec": round(steps_per_sec, 3),
            "budget_exceeded": self.is_budget_exceeded(),
        }
