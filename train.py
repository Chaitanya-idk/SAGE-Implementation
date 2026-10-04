#!/usr/bin/env python3
"""
Convenience entrypoint forwarding to scripts/train.py.
Allows running `python train.py --config configs/config.yaml` directly from repository root.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.train import main

if __name__ == "__main__":
    main()
