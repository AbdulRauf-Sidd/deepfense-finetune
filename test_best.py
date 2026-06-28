#!/usr/bin/env python3
"""
Run the best fine-tuned checkpoint against the test set.

Usage:
    python test_best.py                    # defaults to outputs/unfreeze0
    python test_best.py outputs/unfreeze2
    python test_best.py outputs/unfreeze4
"""

import subprocess
import sys
from pathlib import Path

REAL_DIR = Path("ai_audio_dataset/testing/real")
FAKE_DIR = Path("ai_audio_dataset/testing/fake")


def main():
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/unfreeze0")
    checkpoint = model_dir / "best_model.pth"

    if not checkpoint.exists():
        sys.exit(f"Checkpoint not found: {checkpoint}")

    cmd = [
        sys.executable, "infer.py",
        "--model-dir",  str(model_dir),
        "--real-dir",   str(REAL_DIR),
        "--fake-dir",   str(FAKE_DIR),
        "--output",     str(model_dir / "test_results.json"),
        "--batch-size", "16",
        "--max-len",    "7",
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
