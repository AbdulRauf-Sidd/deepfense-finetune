#!/usr/bin/env python3
"""
Run the last fine-tuned checkpoint against the test set.

Usage:
    python test_last.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

LAST_CHECKPOINT = Path("outputs/finetune_social_media/last_model.pth")
MODEL_DIR       = LAST_CHECKPOINT.parent
ARCH_CONFIG     = Path("models/wavlm/config.yaml")

REAL_DIR    = Path("ai_audio_dataset/testing/real")
FAKE_DIR    = Path("ai_audio_dataset/testing/fake")
OUTPUT_JSON = Path("outputs/finetune_social_media/test_results_last.json")


def main():
    if not LAST_CHECKPOINT.exists():
        sys.exit(f"Checkpoint not found: {LAST_CHECKPOINT}")

    config_dst = MODEL_DIR / "config.yaml"
    if not config_dst.exists():
        if not ARCH_CONFIG.exists():
            sys.exit(f"Architecture config not found: {ARCH_CONFIG}")
        shutil.copy(ARCH_CONFIG, config_dst)
        print(f"Copied {ARCH_CONFIG} → {config_dst}")

    cmd = [
        sys.executable, "infer.py",
        "--model-dir",   str(MODEL_DIR),
        "--checkpoint",  str(LAST_CHECKPOINT),
        "--real-dir",    str(REAL_DIR),
        "--fake-dir",    str(FAKE_DIR),
        "--output",      str(OUTPUT_JSON),
        "--batch-size",  "16",
        "--max-len",     "7",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
