#!/usr/bin/env python3
"""
Run the best fine-tuned checkpoint against the test set.

Usage:
    python test_best.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

BEST_CHECKPOINT = Path("outputs/finetune_social_media/best_model.pth")
MODEL_DIR       = BEST_CHECKPOINT.parent
ARCH_CONFIG     = Path("models/wavlm/config.yaml")

REAL_DIR   = Path("ai_audio_dataset/testing/real")
FAKE_DIR   = Path("ai_audio_dataset/testing/fake")
OUTPUT_JSON = Path("outputs/finetune_social_media/test_results.json")


def main():
    if not BEST_CHECKPOINT.exists():
        sys.exit(f"Checkpoint not found: {BEST_CHECKPOINT}")

    # infer.py needs config.yaml alongside the .pth
    config_dst = MODEL_DIR / "config.yaml"
    if not config_dst.exists():
        if not ARCH_CONFIG.exists():
            sys.exit(f"Architecture config not found: {ARCH_CONFIG}")
        shutil.copy(ARCH_CONFIG, config_dst)
        print(f"Copied {ARCH_CONFIG} → {config_dst}")

    cmd = [
        sys.executable, "infer.py",
        "--model-dir",  str(MODEL_DIR),
        "--real-dir",   str(REAL_DIR),
        "--fake-dir",   str(FAKE_DIR),
        "--output",     str(OUTPUT_JSON),
        "--batch-size", "16",
        "--max-len",    "7",
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
