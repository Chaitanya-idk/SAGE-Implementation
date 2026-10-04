"""
High-performance, shard-aware dataset loader and batch collator for SAGE Qwen2.5-VL fine-tuning.
Provides memory-efficient Parquet reading, image decoding with error recovery,
and label masking for causal multimodal training.
"""

import io
import os
import time
import logging
import base64
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader

from src.prompts import build_conversation, is_valid_field, format_assistant_target

# Enable loading of truncated/progressive images safely
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger("SAGE.dataset")


def decode_image(image_input: Any) -> Image.Image:
    """
    Decodes an image from multiple possible formats:
    - PIL.Image.Image
    - dict with 'bytes' or 'path' (standard Hugging Face Image feature)
    - raw bytes (JPEG/PNG/WEBP)
    - local filepath string
    - base64 string
    Returns RGB PIL.Image.
    """
    if image_input is None:
        raise ValueError("Image input is None")

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, dict):
        if "bytes" in image_input and image_input["bytes"] is not None:
            return Image.open(io.BytesIO(image_input["bytes"])).convert("RGB")
        elif "path" in image_input and image_input["path"] is not None:
            return Image.open(image_input["path"]).convert("RGB")
        else:
            raise ValueError(f"Unrecognized image dict structure: {list(image_input.keys())}")

    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert("RGB")

    if isinstance(image_input, str):
        if os.path.exists(image_input):
            return Image.open(image_input).convert("RGB")
        # Check if base64 encoded
        if image_input.startswith("data:image") or len(image_input) > 256:
            try:
                base64_data = re.sub(r"^data:image/.+;base64,", "", image_input)
                img_bytes = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                pass
        raise ValueError(f"Image path does not exist: {image_input}")

    raise TypeError(f"Unsupported image type: {type(image_input)}")


class BadSampleLogger:
    """Logs corrupted images or malformed rows without crashing the training run."""

    def __init__(self, log_path: str = "artifacts/bad_samples.csv"):
        self.log_path = log_path
        self.bad_samples_count = 0
        self.records: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        # Initialize file with header if not exists
        if not os.path.exists(log_path):
            pd.DataFrame(columns=["timestamp", "filename", "shard_id", "row_idx", "error"]).to_csv(
                log_path, index=False
            )

    def log(self, filename: str, error_msg: str, shard_id: Any = None, row_idx: Any = None) -> None:
        self.bad_samples_count += 1
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": str(filename),
            "shard_id": str(shard_id),
            "row_idx": str(row_idx),
            "error": str(error_msg),
        }
        self.records.append(record)
        if len(self.records) >= 10:
            self.flush()

    def flush(self) -> None:
        if self.records:
            df = pd.DataFrame(self.records)
            df.to_csv(self.log_path, mode="a", header=False, index=False)
            self.records.clear()


class SAGEManifestDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset reading from SAGE Parquet manifest and shards.
    Does NOT keep all 20GB in RAM. Reads records on-demand or from shard caches.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        data_dir: Optional[str] = None,
        bad_sample_logger: Optional[BadSampleLogger] = None,
        fallback_image: Optional[Image.Image] = None,
    ):
        self.manifest = manifest_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.bad_logger = bad_sample_logger or BadSampleLogger()
        self.fallback_image = fallback_image or Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        # Shard caching mechanism: cache the most recently opened shard table to avoid re-opening
        self._cached_shard_path: Optional[str] = None
        self._cached_shard_table: Optional[Any] = None

    def __len__(self) -> int:
        return len(self.manifest)

    def _resolve_shard_path(self, shard_ref: str) -> str:
        """Resolves local or relative path for a shard file."""
        if os.path.isabs(shard_ref) and os.path.exists(shard_ref):
            return shard_ref
        if self.data_dir:
            candidate = os.path.join(self.data_dir, shard_ref)
            if os.path.exists(candidate):
                return candidate
            candidate_base = os.path.join(self.data_dir, os.path.basename(shard_ref))
            if os.path.exists(candidate_base):
                return candidate_base
        if os.path.exists(shard_ref):
            return shard_ref
        return shard_ref

    def _read_image_from_shard(self, shard_path: str, row_idx: int) -> Image.Image:
        """Reads image from parquet shard at specific row index."""
        resolved = self._resolve_shard_path(shard_path)
        
        # Read single cell via pyarrow or cached table
        if self._cached_shard_path != resolved:
            self._cached_shard_path = resolved
            # Read only the 'image' column to minimize RAM footprint
            self._cached_shard_table = pq.read_table(resolved, columns=["image"])
            
        cell = self._cached_shard_table["image"][row_idx].as_py()
        return decode_image(cell)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.manifest.iloc[idx].to_dict()
        filename = row.get("filename", f"row_{idx}")
        shard_path = row.get("shard_path") or row.get("shard_id")
        row_idx_in_shard = row.get("row_idx_in_shard", idx)
        
        img = None
        # Option A: Image is stored directly in manifest row (e.g. small subset/smoke test)
        if "image" in row and row["image"] is not None:
            try:
                img = decode_image(row["image"])
            except Exception as e:
                self.bad_logger.log(filename, f"Error decoding row image: {e}", shard_path, row_idx_in_shard)
                img = self.fallback_image
                
        # Option B: Image stored in parquet shard file
        elif shard_path and os.path.exists(self._resolve_shard_path(str(shard_path))):
            try:
                img = self._read_image_from_shard(str(shard_path), int(row_idx_in_shard))
            except Exception as e:
                self.bad_logger.log(filename, f"Error reading from shard: {e}", shard_path, row_idx_in_shard)
                img = self.fallback_image
                
        # Option C: Direct image file path
        elif "image_path" in row and os.path.exists(str(row["image_path"])):
            try:
                img = Image.open(str(row["image_path"])).convert("RGB")
            except Exception as e:
                self.bad_logger.log(filename, f"Error opening image path: {e}", shard_path, row_idx_in_shard)
                img = self.fallback_image
        else:
            # Fallback for missing image reference
            self.bad_logger.log(filename, "No image source found in row", shard_path, row_idx_in_shard)
            img = self.fallback_image
            
        return {
            "image": img,
            "metadata": row,
            "filename": filename,
        }


class QwenDataCollator:
    """
    Collator for Qwen2.5-VL that:
    1. Formats multi-turn chat templates with image tokens.
    2. Tokenizes conversation with processor.
    3. Builds attention masks and pixel value tensors.
    4. Masks instruction prompt tokens in `labels` with -100 so loss is computed ONLY on the assistant's diagnosis!
    """

    def __init__(
        self,
        processor: Any,
        is_training: bool = True,
        max_length: int = 1024,
    ):
        self.processor = processor
        self.is_training = is_training
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        images: List[Image.Image] = []
        target_texts: List[str] = []
        raw_rows: List[Dict[str, Any]] = []

        for item in batch:
            img = item["image"]
            meta = item["metadata"]
            raw_rows.append(meta)
            images.append(img)
            
            if self.is_training:
                # Include assistant target in the conversation
                conv = build_conversation(meta, img, include_target=True)
                # Render using Qwen's chat template
                formatted_text = self.processor.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False
                )
                texts.append(formatted_text)
                target_texts.append(format_assistant_target(meta))
            else:
                # Evaluation mode: prompt only (with generation prompt)
                conv = build_conversation(meta, img, include_target=False)
                formatted_text = self.processor.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )
                texts.append(formatted_text)
                target_texts.append(format_assistant_target(meta))

        # Process inputs using Qwen2.5-VL processor
        try:
            from qwen_vl_utils import process_vision_info
            # If Qwen vision utils is present, extract vision inputs cleanly
            convs = [build_conversation(meta, img, include_target=self.is_training) for meta, img in zip(raw_rows, images)]
            image_inputs, video_inputs = process_vision_info(convs)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        except Exception:
            # Fallback direct processor call
            inputs = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt"
            )

        if self.is_training:
            # Mask user instruction tokens in labels with -100
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            
            # Find assistant header token start in each sequence
            # In Qwen2.5-VL chat template: "<|im_start|>assistant\n" marks assistant turn
            assistant_token_str = "<|im_start|>assistant\n"
            assistant_token_ids = self.processor.tokenizer.encode(assistant_token_str, add_special_tokens=False)
            
            for i in range(len(input_ids)):
                seq = input_ids[i].tolist()
                # Find occurrence of assistant token sequence
                found_idx = -1
                m = len(assistant_token_ids)
                for j in range(len(seq) - m + 1):
                    if seq[j:j+m] == assistant_token_ids:
                        found_idx = j + m
                        break
                        
                if found_idx != -1:
                    # Mask everything before the assistant response
                    labels[i, :found_idx] = -100
                else:
                    # If not explicitly found, mask user prompt length heuristically
                    user_conv = build_conversation(raw_rows[i], images[i], include_target=False)
                    user_text = self.processor.apply_chat_template(user_conv, tokenize=False, add_generation_prompt=True)
                    user_tokens = self.processor.tokenizer.encode(user_text, add_special_tokens=False)
                    cutoff = min(len(user_tokens), len(seq))
                    labels[i, :cutoff] = -100
                    
            # Mask padding tokens
            if "attention_mask" in inputs:
                labels[inputs["attention_mask"] == 0] = -100
                
            inputs["labels"] = labels

        # Attach original metadata for evaluation/tracking
        inputs["metadata"] = raw_rows
        inputs["target_texts"] = target_texts
        return inputs


class PipelineProfiler:
    """Tracks time spent loading data vs GPU compute to detect data bottlenecks."""

    def __init__(self):
        self.data_wait_time = 0.0
        self.gpu_compute_time = 0.0
        self.last_timestamp = time.time()
        self.state = "data" # "data" or "compute"

    def mark_data_ready(self) -> None:
        now = time.time()
        self.data_wait_time += (now - self.last_timestamp)
        self.last_timestamp = now
        self.state = "compute"

    def mark_compute_finished(self) -> None:
        now = time.time()
        self.gpu_compute_time += (now - self.last_timestamp)
        self.last_timestamp = now
        self.state = "data"

    def get_wait_percentage(self) -> float:
        total = self.data_wait_time + self.gpu_compute_time
        if total <= 1e-6:
            return 0.0
        return round((self.data_wait_time / total) * 100.0, 1)

    def summary(self) -> Dict[str, Any]:
        return {
            "data_wait_sec": round(self.data_wait_time, 2),
            "gpu_compute_sec": round(self.gpu_compute_time, 2),
            "data_wait_pct": self.get_wait_percentage(),
        }
