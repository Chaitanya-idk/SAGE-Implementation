"""
Model initialization, hardware-aware quantization, and PEFT LoRA configuration
for Qwen2.5-VL-3B-Instruct.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

from src.utils import get_device_info, resolve_dtype

# Compatibility patch for pre-installed older torchao in Kaggle/cloud environments
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

try:
    import peft.tuners.lora.torchao as _peft_torchao
    _peft_torchao.is_torchao_available = lambda: False
except Exception:
    pass

logger = logging.getLogger("SAGE.model")


def print_model_parameters(model: torch.nn.Module, header: str = "LoRA Parameter Summary") -> Dict[str, Any]:
    """Calculates and logs total, trainable, and frozen parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    print("=" * 65)
    print(f" {header}")
    print("=" * 65)
    print(f" Total Parameters:     {total_params:>15,}")
    print(f" Trainable Parameters: {trainable_params:>15,}")
    print(f" Trainable %:          {trainable_pct:>14.4f}%")
    print("=" * 65)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_pct": round(trainable_pct, 4),
    }


def load_processor(
    model_ref: str,
    min_pixels: int = 3136,
    max_pixels: int = 1003520,
    trust_remote_code: bool = True,
) -> Qwen2_5_VLProcessor:
    """Loads and configures official Qwen2.5-VL processor."""
    logger.info(f"Loading Qwen2.5-VL processor from '{model_ref}'...")
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_ref,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=trust_remote_code,
    )
    return processor


def setup_model_and_lora(
    config: Dict[str, Any],
    is_training: bool = True,
    adapter_checkpoint_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, Qwen2_5_VLProcessor]:
    """
    Initializes Qwen2.5-VL-3B with:
    - Auto-detected GPU precision (bf16/fp16)
    - Optional 4-bit/8-bit quantization
    - PEFT LoRA adapter
    - Gradient checkpointing
    """
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})

    # Determine model path / identifier
    model_path = model_cfg.get("path")
    model_name = model_cfg.get("name", "Qwen/Qwen2.5-VL-3B-Instruct")
    model_ref = model_path if (model_path and os.path.exists(model_path)) else model_name

    # Detect hardware & precision
    dev_info = get_device_info()
    target_dtype = resolve_dtype(model_cfg.get("dtype", "auto"), dev_info["bf16_supported"])
    quant_mode = model_cfg.get("quantization", "none").lower()

    print("\n" + "=" * 65)
    print(" SAGE HARDWARE & MODEL CONFIGURATION")
    print("=" * 65)
    print(f" Target Model Ref:      {model_ref}")
    print(f" GPU Device:            {dev_info['device_name']}")
    print(f" Available VRAM:        {dev_info['total_memory_gb']} GB")
    print(f" Compute Dtype:         {str(target_dtype).replace('torch.', '')}")
    print(f" Quantization Mode:     {quant_mode.upper()}")
    print("=" * 65 + "\n")

    # Configure quantization if requested
    bnb_config = None
    if quant_mode in ("4bit", "4-bit", "nf4"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=target_dtype,
        )
    elif quant_mode in ("8bit", "8-bit"):
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load base model
    logger.info(f"Loading base model from {model_ref}...")
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": target_dtype,
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
        "low_cpu_mem_usage": True,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    else:
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_ref, **model_kwargs)

    # Load processor
    processor = load_processor(
        model_ref=model_ref,
        min_pixels=model_cfg.get("min_pixels", 3136),
        max_pixels=model_cfg.get("max_pixels", 1003520),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    # Gradient checkpointing
    if is_training and train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if quant_mode in ("4bit", "4-bit", "nf4", "8bit"):
            model = prepare_model_for_kbit_training(model)
        else:
            model.enable_input_require_grads()

    # Apply or resume LoRA
    if adapter_checkpoint_path and os.path.exists(adapter_checkpoint_path):
        logger.info(f"Loading LoRA adapter weights from checkpoint: {adapter_checkpoint_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_checkpoint_path,
            is_trainable=is_training,
        )
    elif is_training:
        logger.info("Initializing fresh PEFT LoRA adapter configuration...")
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 32),
            lora_alpha=lora_cfg.get("alpha", 64),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, peft_config)

    print_model_parameters(model)
    return model, processor
