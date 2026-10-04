"""
SAGE: Scalable Agentic Grounded Evaluation for Crop Disease Diagnosis
Qwen2.5-VL-3B LoRA Production Pipeline Package
"""

__version__ = "1.0.0"

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
