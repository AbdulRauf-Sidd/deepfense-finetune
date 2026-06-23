#!/usr/bin/env python3
"""
Download dataset and model weights from S3.

Usage:
    python download_from_s3.py

Requires: pip install boto3
IAM role on EC2 must have s3:GetObject / s3:ListBucket on ai-audio-dataset.
"""

import boto3
import os
import sys
from pathlib import Path

BUCKET = "ai-audio-dataset"

DATASET_PREFIXES = [
    "train/real/",
    "train/fake/",
    "validation/real/",
    "validation/fake/",
    "testing/real/",
    "testing/fake/",
]

MODEL_KEY       = "codecfake_wavlm_nes2net_best_model.pth"
MODEL_LOCAL     = Path("models/wavlm/codecfake_wavlm_nes2net_best_model.pth")
DATASET_LOCAL   = Path("ai_audio_dataset")


def download_file(s3, bucket, key, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        print(f"  skip (exists): {local_path}")
        return
    print(f"  {key}  →  {local_path}")
    s3.download_file(bucket, key, str(local_path))


def sync_prefix(s3, bucket, prefix, local_dir):
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    keys = [
        obj["Key"]
        for page in pages
        for obj in page.get("Contents", [])
    ]

    if not keys:
        print(f"  WARNING: no objects found under s3://{bucket}/{prefix}")
        return

    print(f"\ns3://{bucket}/{prefix}  ({len(keys)} files)")
    for key in keys:
        relative  = key[len(prefix):]          # strip the prefix
        local_path = local_dir / prefix / relative
        download_file(s3, bucket, key, local_path)


def main():
    s3 = boto3.client("s3")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nDownloading model weights...")
    download_file(s3, BUCKET, MODEL_KEY, MODEL_LOCAL)

    # ── Dataset ───────────────────────────────────────────────────────────────
    for prefix in DATASET_PREFIXES:
        sync_prefix(s3, BUCKET, prefix, DATASET_LOCAL)

    print("\nDone.")


if __name__ == "__main__":
    main()
