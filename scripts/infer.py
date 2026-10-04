#!/usr/bin/env python3
"""
Qualitative single-image inference script for SAGE Qwen2.5-VL.
Produces structured crop disease diagnosis directly from an input agricultural image:
- Crop
- Diagnosis (canonical disease)
- Disease Type
- Plant Organ
- Visual Symptoms
- Pathogen
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import torch

from src.utils import load_config
from src.model import setup_model_and_lora
from src.prompts import build_conversation, parse_generated_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SAGE.infer")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAGE diagnostic inference on a single crop image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input crop image.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best", help="Path to LoRA checkpoint.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration YAML.")
    parser.add_argument("--crop", type=str, default="", help="Optional known crop name (e.g. Tomato, Soybean).")
    parser.add_argument("--organ", type=str, default="", help="Optional known plant organ (e.g. Leaf, Fruit).")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and processor with trained LoRA weights
    logger.info(f"Loading Qwen2.5-VL with adapter from '{args.checkpoint}'...")
    model, processor = setup_model_and_lora(
        config=config,
        is_training=False,
        adapter_checkpoint_path=args.checkpoint,
    )
    model.eval()

    # Load and decode image
    image = Image.open(args.image).convert("RGB")

    # Build prompt
    row_meta = {}
    if args.crop:
        row_meta["crop"] = args.crop
    if args.organ:
        row_meta["plant_organ"] = args.organ

    conv = build_conversation(row_meta, image, include_target=False)
    text_prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    # Process inputs
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info([conv])
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
    except Exception:
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

    model_inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        input_len = model_inputs["input_ids"].shape[1]
        generated_tokens = output_ids[:, input_len:]
        output_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    # Parse structured diagnostic fields
    parsed = parse_generated_response(output_text)

    print("\n" + "=" * 65)
    print(" SAGE MULTIMODAL AGRICULTURAL DIAGNOSIS")
    print("=" * 65)
    print(f" Input Image:       {args.image}")
    print(f" Crop:              {parsed['crop'] or args.crop or 'Unspecified'}")
    print(f" Diagnosis:         {parsed['diagnosis']}")
    print(f" Disease Type:      {parsed['disease_type'] or 'N/A'}")
    print(f" Plant Organ:       {parsed['plant_organ'] or args.organ or 'N/A'}")
    print(f" Visual Symptoms:   {parsed['visual_symptoms'] or 'N/A'}")
    print(f" Pathogen:          {parsed['pathogen'] or 'N/A'}")
    print("-" * 65)
    print(" Raw Output:")
    print(output_text.strip())
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
