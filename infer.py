"""
Deepfense Inference Script
Runs a trained model on two folders (real / fake audio) and reports per-file
scores plus aggregate detection metrics.

Usage:
    python infer.py \
        --model-dir  models/sample_path \
        --real-dir   /path/to/real/audio \
        --fake-dir   /path/to/fake/audio \
        [--output    results.json] \
        [--batch-size 8] \
        [--sr 16000] \
        [--max-len 4]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from deepfense.models import *  # registers all model types
from deepfense.training.evaluations import *  # registers all metrics
from deepfense.utils.registry import build_detector, METRIC_REGISTRY
from deepfense.training.evaluations.evaluator import Evaluator
from deepfense.training.evaluations.utils import _metric_get_1d_scores

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aiff", ".aif"}

# Label convention: real=1 (bonafide), fake=0 (spoof)
REAL_LABEL = 1
FAKE_LABEL = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AMSoftmaxHead(nn.Module):
    """Minimal AM-Softmax head used to load finetuned checkpoints."""
    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))


def find_audio_files(folder: Path) -> list[Path]:
    files = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS
    )
    if not files:
        log.warning(f"No audio files found in {folder}")
    return files


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    return x.astype(np.float32)


def _write_wavlm_large_stub(path: Path):
    """Write a config-only WavLM-Large stub so the unil loader builds the right arch."""
    stub = {
        "cfg": {
            "extractor_mode": "layer_norm",
            "encoder_layers": 24,
            "encoder_embed_dim": 1024,
            "encoder_ffn_embed_dim": 4096,
            "encoder_attention_heads": 16,
            "layer_norm_first": True,
            "relative_position_embedding": True,
            "num_buckets": 320,
            "max_distance": 800,
            "gru_rel_pos": True,
            "conv_pos": 128,
            "conv_pos_groups": 16,
            "conv_bias": False,
            "feature_grad_mult": 1.0,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "encoder_layerdrop": 0.0,
            "dropout_input": 0.0,
            "dropout_features": 0.0,
            "mask_length": 10,
            "mask_prob": 0.65,
            "mask_selection": "static",
            "mask_other": 0,
            "no_mask_overlap": False,
            "mask_min_space": 1,
            "mask_channel_length": 10,
            "mask_channel_prob": 0.0,
            "mask_channel_selection": "static",
            "mask_channel_other": 0,
            "no_mask_channel_overlap": False,
            "mask_channel_min_space": 1,
            "normalize": False,
            "activation_fn": "gelu",
            "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        },
        "model": {},  # empty — all weights come from the training checkpoint
    }
    torch.save(stub, path)
    log.info(f"Wrote WavLM-Large arch stub to {path}")


def load_model(model_dir: Path, device: torch.device, wavlm_path: str | None = None):
    config_path = model_dir / "config.yaml"

    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    # Find checkpoint: accept any .pth in the model dir
    pth_files = list(model_dir.glob("*.pth"))
    if not pth_files:
        sys.exit(f"No .pth checkpoint found in {model_dir}")
    ckpt_path = pth_files[0]
    if len(pth_files) > 1:
        log.warning(f"Multiple .pth files found; using {ckpt_path.name}")

    cfg = OmegaConf.load(config_path)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    # Patch WavLM frontend path so it doesn't rely on the original cluster path
    frontend_cfg = model_cfg.get("frontend", {})
    if frontend_cfg.get("type") == "wavlm":
        args = frontend_cfg.setdefault("args", {})
        if wavlm_path:
            args["ckpt_path"] = wavlm_path
            args["source"] = "unil"
            log.info(f"WavLM: using local checkpoint {wavlm_path}")
        else:
            # Build a minimal config-only stub so the unil loader can construct
            # the WavLM-Large architecture without the full 1.3 GB base weights.
            # All actual weights come from the training checkpoint below.
            stub_path = model_dir / "_wavlm_large_stub.pt"
            if not stub_path.exists():
                _write_wavlm_large_stub(stub_path)
            args["ckpt_path"] = str(stub_path)
            args["source"] = "unil"
            log.info("WavLM: using architecture stub (weights loaded from training checkpoint)")

    model = build_detector(cfg.model.type, model_cfg)
    model.to(device)

    state = torch.load(ckpt_path, map_location=device)
    incompatible = model.load_state_dict(state["model_state"], strict=False)
    if incompatible.missing_keys:
        log.info(f"  missing keys (loaded from pretrained frontend): {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        log.warning(f"  unexpected keys in checkpoint: {incompatible.unexpected_keys}")
    model.eval()
    log.info(f"Loaded model weights from {ckpt_path.name}")

    loss_fn = None
    if "loss_fn" in state:
        try:
            lf_state = state["loss_fn"]
            num_classes, in_features = lf_state["weight"].shape
            loss_fn = _AMSoftmaxHead(in_features, num_classes).to(device)
            loss_fn.load_state_dict(lf_state)
            loss_fn.eval()
            log.info("Loaded finetuned AM-Softmax head from checkpoint")
        except Exception as e:
            log.warning(f"Could not load AM-Softmax head: {e} — using model's built-in classifier")

    return model, loss_fn


def pad_batch(waveforms: list[np.ndarray], max_len_samples: int | None):
    """Pad / truncate to equal length, return (tensor, mask)."""
    if max_len_samples is not None:
        waveforms = [w[:max_len_samples] for w in waveforms]
    max_len = max(len(w) for w in waveforms)
    padded, masks = [], []
    for w in waveforms:
        n = len(w)
        pad = max_len - n
        padded.append(np.concatenate([w, np.zeros(pad, dtype=np.float32)]))
        masks.append(np.array([1.0] * n + [0.0] * pad, dtype=np.float32))
    return (
        torch.from_numpy(np.stack(padded)),   # (B, T)
        torch.from_numpy(np.stack(masks)),    # (B, T)
    )


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(
    model,
    files: list[Path],
    labels: list[int],
    device: torch.device,
    batch_size: int,
    target_sr: int,
    max_len_samples: int | None,
    loss_fn=None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Returns:
        per_file  – list of dicts with file-level results
        all_labels – np.ndarray shape (N,)
        all_scores – np.ndarray shape (N,) or (N,C), raw model scores
    """
    per_file   = []
    all_labels = []
    all_scores = []

    # Load all waveforms first (with tqdm progress)
    log.info("Loading audio files …")
    waveforms = []
    skipped   = []
    for f in tqdm(files, desc="Loading", unit="file"):
        try:
            waveforms.append(load_audio(f, target_sr))
        except Exception as e:
            log.warning(f"Skipping {f.name}: {e}")
            skipped.append(str(f))
            waveforms.append(None)

    log.info("Running inference …")
    n = len(files)
    for start in tqdm(range(0, n, batch_size), desc="Inference", unit="batch"):
        end   = min(start + batch_size, n)
        batch_files  = files[start:end]
        batch_waves  = waveforms[start:end]
        batch_labels = labels[start:end]

        # Filter out failed loads
        valid = [(f, w, l) for f, w, l in zip(batch_files, batch_waves, batch_labels) if w is not None]
        if not valid:
            continue
        vfiles, vwaves, vlabels = zip(*valid)

        x, mask = pad_batch(list(vwaves), max_len_samples)
        x    = x.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            out = model(x, mask=mask)

        if loss_fn is not None:
            # Finetuned checkpoint: use the AM-Softmax head.
            # Training convention was real=0, fake=1.
            # Swap columns so index REAL_LABEL=1 holds the real score/prob,
            # making the rest of this loop work unchanged.
            features = out["embeddings"]
            cosine   = loss_fn(features).float().detach()
            probs_ft = F.softmax(cosine, dim=-1).cpu().numpy()
            scores   = cosine.cpu().numpy()[:, [1, 0]]   # col 1 = real logit
            probs    = probs_ft[:, [1, 0]]                # col 1 = P(real)
        else:
            scores = out["scores"]          # (B,) or (B,C)
            probs  = out["probs"]           # (B,) or (B,C) – post-softmax/sigmoid
            if torch.is_tensor(scores):
                scores = scores.detach().cpu().numpy()
            if torch.is_tensor(probs):
                probs = probs.detach().cpu().numpy()

        # Build 1-D detection score (higher → more bonafide)
        dummy_params = {"bonafide_label": REAL_LABEL, "loss": "crossentropy"}
        scores_1d = _metric_get_1d_scores(scores, dummy_params)

        for i, (fpath, label) in enumerate(zip(vfiles, vlabels)):
            s1d   = float(scores_1d[i])
            p_raw = probs[i]

            # Probability of real (bonafide) class
            if np.ndim(p_raw) == 0 or (hasattr(p_raw, "__len__") and len(p_raw) == 1):
                p_real = float(np.squeeze(p_raw))
                p_fake = 1.0 - p_real
            elif len(p_raw) >= 2:
                p_real = float(p_raw[REAL_LABEL])
                p_fake = float(p_raw[FAKE_LABEL])
            else:
                p_real = float(np.squeeze(p_raw))
                p_fake = 1.0 - p_real

            predicted_label = REAL_LABEL if s1d > 0 else FAKE_LABEL
            correct = int(predicted_label == label)

            per_file.append({
                "file":            str(fpath),
                "true_label":      "real" if label == REAL_LABEL else "fake",
                "predicted_label": "real" if predicted_label == REAL_LABEL else "fake",
                "correct":         bool(correct),
                "score":           round(s1d, 6),
                "p_real":          round(p_real, 6),
                "p_fake":          round(p_fake, 6),
            })
            all_labels.append(label)
            all_scores.append(scores[i])

    all_labels = np.array(all_labels, dtype=int)
    if all_scores:
        all_scores = np.stack(all_scores)
    else:
        all_scores = np.array([])

    if skipped:
        log.warning(f"{len(skipped)} file(s) were skipped due to load errors.")

    return per_file, all_labels, all_scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

ALL_METRICS = {
    "EER":     {},
    "minDCF":  {"Pspoof": 0.5},
    "actDCF":  {"Pspoof": 0.5},
    "CLLR":    {"bonafide_label": REAL_LABEL},
    "EER_CI":  {"bonafide_label": REAL_LABEL},
    "F1_SCORE":{"f1_average": "macro"},
    "ACC":     {},
}


def compute_all_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    evaluator = Evaluator(ALL_METRICS)
    return evaluator.evaluate(labels, scores)


def per_class_metrics(
    per_file: list[dict],
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """Compute metrics separately for real and fake subsets."""
    results = {}
    for cls_name, cls_label in [("real", REAL_LABEL), ("fake", FAKE_LABEL)]:
        mask = labels == cls_label
        if mask.sum() == 0:
            continue
        n_correct = sum(1 for r in per_file if r["true_label"] == cls_name and r["correct"])
        n_total   = mask.sum()
        results[f"{cls_name}_accuracy"] = round(n_correct / n_total, 6)
        results[f"{cls_name}_count"]    = int(n_total)
    return results


def confusion_matrix_counts(per_file: list[dict]) -> dict:
    tp = sum(1 for r in per_file if r["true_label"] == "real" and r["predicted_label"] == "real")
    tn = sum(1 for r in per_file if r["true_label"] == "fake" and r["predicted_label"] == "fake")
    fp = sum(1 for r in per_file if r["true_label"] == "fake" and r["predicted_label"] == "real")
    fn = sum(1 for r in per_file if r["true_label"] == "real" and r["predicted_label"] == "fake")
    n  = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": round(precision, 6),
        "recall":    round(recall, 6),
        "f1":        round(f1, 6),
        "fpr":       round(fpr, 6),
        "fnr":       round(fnr, 6),
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(summary: dict):
    SEP = "=" * 62
    print(f"\n{SEP}")
    print("  DEEPFENSE INFERENCE SUMMARY")
    print(SEP)

    ds = summary["dataset"]
    print(f"  Real files   : {ds['n_real']}")
    print(f"  Fake files   : {ds['n_fake']}")
    print(f"  Total files  : {ds['n_total']}")
    print(f"  Inference time: {summary['inference_time_s']:.1f}s")

    print(f"\n--- Aggregate Metrics ---")
    for k, v in summary["metrics"].items():
        if isinstance(v, float):
            print(f"  {k:<20} {v:.4f}")
        else:
            print(f"  {k:<20} {v}")

    print(f"\n--- Per-Class Accuracy ---")
    pc = summary["per_class"]
    print(f"  Real accuracy : {pc.get('real_accuracy', 'N/A')}")
    print(f"  Fake accuracy : {pc.get('fake_accuracy', 'N/A')}")

    print(f"\n--- Confusion Matrix ---")
    cm = summary["confusion_matrix"]
    print(f"             Pred Real   Pred Fake")
    print(f"  True Real     {cm['TP']:<10} {cm['FN']}")
    print(f"  True Fake     {cm['FP']:<10} {cm['TN']}")
    print(f"\n  Precision : {cm['precision']:.4f}")
    print(f"  Recall    : {cm['recall']:.4f}")
    print(f"  F1        : {cm['f1']:.4f}")
    print(f"  FPR       : {cm['fpr']:.4f}  (fake accepted as real)")
    print(f"  FNR       : {cm['fnr']:.4f}  (real rejected as fake)")
    print(SEP + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Deepfense audio-folder inference")
    p.add_argument("--model-dir",  required=True,  help="Folder with config.yaml + *.pth checkpoint")
    p.add_argument("--real-dir",   required=True,  help="Folder of real (bonafide) audio files")
    p.add_argument("--fake-dir",   required=True,  help="Folder of fake (spoof) audio files")
    p.add_argument("--output",     default="results.json", help="Path to write JSON output")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sr",         type=int, default=16000, help="Target sample rate (Hz)")
    p.add_argument("--max-len",    type=float, default=None,
                   help="Max clip length in seconds (truncate longer files). Default: no limit.")
    p.add_argument("--device",     default=None, help="cuda / cpu (auto-detected if omitted)")
    p.add_argument("--wavlm-path", default=None,
                   help="Path to local WavLM-Large.pt. If omitted, an architecture stub is "
                        "auto-generated and all weights are loaded from the training checkpoint.")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info(f"Device: {device}")

    model_dir = Path(args.model_dir)
    real_dir  = Path(args.real_dir)
    fake_dir  = Path(args.fake_dir)

    for d in (real_dir, fake_dir):
        if not d.is_dir():
            sys.exit(f"Directory not found: {d}")

    max_len_samples = int(args.max_len * args.sr) if args.max_len else None

    # Collect files + labels
    real_files = find_audio_files(real_dir)
    fake_files = find_audio_files(fake_dir)

    if not real_files and not fake_files:
        sys.exit("No audio files found in either folder.")

    all_files  = real_files + fake_files
    all_labels = [REAL_LABEL] * len(real_files) + [FAKE_LABEL] * len(fake_files)

    log.info(f"Real files: {len(real_files)}  |  Fake files: {len(fake_files)}")

    # Load model
    model, loss_fn = load_model(model_dir, device, wavlm_path=args.wavlm_path)

    # Inference
    t0 = time.time()
    per_file, labels, scores = run_inference(
        model, all_files, all_labels, device,
        args.batch_size, args.sr, max_len_samples,
        loss_fn=loss_fn,
    )
    elapsed = time.time() - t0

    if len(labels) == 0:
        sys.exit("No files were successfully processed.")

    # Metrics
    log.info("Computing metrics …")
    metrics = compute_all_metrics(labels, scores)
    # Round floats
    metrics = {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}

    per_class = per_class_metrics(per_file, labels, scores)
    cm        = confusion_matrix_counts(per_file)

    # Score distribution stats
    score_1d = _metric_get_1d_scores(
        scores, {"bonafide_label": REAL_LABEL, "loss": "crossentropy"}
    )
    real_mask = labels == REAL_LABEL
    fake_mask = labels == FAKE_LABEL

    score_stats = {
        "all":  {"mean": float(score_1d.mean()), "std": float(score_1d.std()),
                 "min": float(score_1d.min()),   "max": float(score_1d.max())},
        "real": {"mean": float(score_1d[real_mask].mean()), "std": float(score_1d[real_mask].std())}
               if real_mask.any() else {},
        "fake": {"mean": float(score_1d[fake_mask].mean()), "std": float(score_1d[fake_mask].std())}
               if fake_mask.any() else {},
    }

    # Assemble output
    summary = {
        "dataset": {
            "n_real":  len(real_files),
            "n_fake":  len(fake_files),
            "n_total": len(per_file),
            "real_dir": str(real_dir),
            "fake_dir": str(fake_dir),
        },
        "model_dir":        str(model_dir),
        "inference_time_s": round(elapsed, 2),
        "metrics":          metrics,
        "per_class":        per_class,
        "confusion_matrix": cm,
        "score_stats":      score_stats,
        "per_file":         per_file,
    }

    # Console summary
    print_summary(summary)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
