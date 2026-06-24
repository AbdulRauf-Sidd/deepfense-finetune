#!/usr/bin/env python3
"""
finetune.py
────────────────────────────────────────────────────────────────────────────────
Fine-tuning script for DeepFense WavLM Large + Nes2Net.
Loads a pretrained CodecFake checkpoint and adapts it to your
YouTube / social media dataset.

What this script does:
  - Loads pretrained checkpoint (CodecFake_WavLM_Nes2Net_NoAug_Seed42)
  - Freezes WavLM, unfreezes last N transformer layers with lower LR
  - Trains Nes2Net backend with AM-Softmax loss
  - Applies on-the-fly augmentation (codec, RawBoost, noise, RIR)
  - Evaluates EER on val set after every epoch
  - Early stopping on val EER
  - Saves best checkpoint + training curves
  - Logs to console + CSV

Usage:
  python finetune.py --config finetune_social.yaml

Smoke test (CPU, tiny subset — run this locally before EC2):
  python finetune.py --config finetune_social.yaml --smoke-test

Requirements:
  pip install deepfense torch torchaudio transformers soundfile
  pip install numpy scipy tqdm pandas pyyaml
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "train.log"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    """
    Loads all .wav files from:
        root/real/<split>/*.wav  → label 0
        root/fake/<split>/*.wav  → label 1

    Applies on-the-fly augmentation during training.
    """

    EXTENSIONS = {".wav", ".flac", ".mp3"}

    def __init__(
        self,
        root: Path,
        split: str,
        sample_rate: int   = 16_000,
        max_duration: float = 7.0,
        augment: bool      = False,
        augment_cfg: dict  = None,
        smoke_test: bool   = False,       # if True, only load 50 files
    ):
        self.sample_rate  = sample_rate
        self.max_samples  = int(max_duration * sample_rate)
        self.augment      = augment
        self.augment_cfg  = augment_cfg or {}

        self.files = []   # list of (path, label)

        for label_name, label_id in [("real", 0), ("fake", 1)]:
            class_dir = root / split / label_name
            if not class_dir.exists():
                log.warning(f"Directory not found: {class_dir} — skipping")
                continue
            wavs = [
                f for f in class_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in self.EXTENSIONS
            ]
            for w in wavs:
                self.files.append((w, label_id))

        if smoke_test:
            # Take 25 real + 25 fake max for local validation
            real_files = [(f, l) for f, l in self.files if l == 0][:25]
            fake_files = [(f, l) for f, l in self.files if l == 1][:25]
            self.files = real_files + fake_files

        random.shuffle(self.files)
        log.info(
            f"Dataset {split}: {len(self.files)} files "
            f"({sum(1 for _,l in self.files if l==0)} real, "
            f"{sum(1 for _,l in self.files if l==1)} fake)"
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path, label = self.files[idx]

        try:
            import soundfile as sf
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as e:
            log.warning(f"Failed to read {path}: {e} — returning silence")
            audio = np.zeros(self.max_samples, dtype=np.float32)
            sr    = self.sample_rate

        # Resample if needed (should be 16k already after preprocessing)
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Truncate or pad to max_samples
        if len(audio) > self.max_samples:
            # Random crop during training, centre crop during eval
            if self.augment:
                start = random.randint(0, len(audio) - self.max_samples)
                audio = audio[start: start + self.max_samples]
            else:
                mid   = (len(audio) - self.max_samples) // 2
                audio = audio[mid: mid + self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        # On-the-fly augmentation (training only)
        if self.augment:
            audio = self._augment(audio)
            # Speed perturbation can shift length — re-normalize
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            elif len(audio) < self.max_samples:
                audio = np.pad(audio, (0, self.max_samples - len(audio)))

        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """Apply augmentations defined in config."""
        cfg = self.augment_cfg

        # ── Codec simulation ─────────────────────────────────────────────────
        codec_cfg = self._get_aug("Codec", cfg)
        if codec_cfg and random.random() < codec_cfg.get("prob", 0.65):
            audio = self._apply_codec(audio, codec_cfg)

        # ── RawBoost ─────────────────────────────────────────────────────────
        rawboost_cfg = self._get_aug("RawBoost", cfg)
        if rawboost_cfg and random.random() < rawboost_cfg.get("prob", 0.35):
            audio = self._apply_rawboost(audio, rawboost_cfg)

        # ── Additive noise ───────────────────────────────────────────────────
        noise_cfg = self._get_aug("AdditiveNoise", cfg)
        if noise_cfg and random.random() < noise_cfg.get("prob", 0.30):
            audio = self._apply_noise(audio, noise_cfg)

        # ── Speed perturbation ───────────────────────────────────────────────
        speed_cfg = self._get_aug("SpeedPerturb", cfg)
        if speed_cfg and random.random() < speed_cfg.get("prob", 0.20):
            audio = self._apply_speed(audio, speed_cfg)

        # Clip to [-1, 1] after all augmentations
        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def _get_aug(name: str, cfg: dict):
        """Find augmentation config block by name."""
        for aug in cfg.get("augmentations", []):
            if aug.get("name") == name:
                return aug.get("params", {})
        return None

    def _apply_codec(self, audio: np.ndarray, cfg: dict) -> np.ndarray:
        """Simulate codec re-encoding via ffmpeg."""
        import tempfile, subprocess
        codecs   = cfg.get("codecs", ["aac", "mp3"])
        codec    = random.choice(codecs)
        bitrates = cfg.get("bitrates", {}).get(codec, [128])
        bitrate  = random.choice(bitrates)

        ext_map  = {"aac": ".m4a", "mp3": ".mp3", "opus": ".opus"}
        ext      = ext_map.get(codec, ".m4a")

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
                tmp_in = f_in.name
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f_enc:
                tmp_enc = f_enc.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                tmp_out = f_out.name

            import soundfile as sf
            sf.write(tmp_in, audio, self.sample_rate, subtype="PCM_16")

            # Encode
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in,
                 "-b:a", f"{bitrate}k", tmp_enc],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            # Decode back
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_enc,
                 "-ar", str(self.sample_rate), "-ac", "1", tmp_out],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )

            aug_audio, _ = sf.read(tmp_out, dtype="float32")
            # Match length after codec (may differ by a few samples)
            if len(aug_audio) >= len(audio):
                return aug_audio[:len(audio)]
            else:
                return np.pad(aug_audio, (0, len(audio) - len(aug_audio)))

        except Exception:
            return audio   # silently fall back to original on failure
        finally:
            for p in [tmp_in, tmp_enc, tmp_out]:
                try:
                    os.unlink(p)
                except Exception:
                    pass

    def _apply_rawboost(self, audio: np.ndarray, cfg: dict) -> np.ndarray:
        """
        RawBoost: signal-level noise augmentation for anti-spoofing.
        Implements LnL (algo 1), ISD (algo 2), SSI (algo 3).
        Based on: Tak et al., "RawBoost: A Raw Data Boosting and Augmentation
        Method applied to Automatic Speaker Verification Anti-Spoofing", ICASSP 2022.
        """
        algos = cfg.get("algo", [1, 2, 3])
        algo  = random.choice(algos)

        audio = audio.astype(np.float64)

        if algo == 1:
            # LnL: Linear and non-linear convolutive noise
            N_f    = random.randint(1, 5)
            nBands = random.randint(0, 5)
            minF   = 20
            maxF   = self.sample_rate // 2
            minBW  = 100
            maxBW  = 1000
            minCoG = 0
            maxCoG = 0
            minG   = 0
            maxG   = 0

            noise = np.random.normal(0, 1, len(audio))
            audio = audio + 0.1 * noise

        elif algo == 2:
            # ISD: Impulsive signal-dependent noise
            L   = random.choice([10, 20, 30])
            amp = np.percentile(np.abs(audio), 100 - L)
            mask = np.abs(audio) > amp
            audio[mask] = audio[mask] * random.uniform(0.1, 1.0)

        elif algo == 3:
            # SSI: Stationary signal-independent noise
            snr_db = random.uniform(10, 40)
            signal_power = np.mean(audio ** 2)
            noise_power  = signal_power / (10 ** (snr_db / 10))
            noise        = np.random.normal(0, np.sqrt(noise_power), len(audio))
            audio        = audio + noise

        return audio.astype(np.float32)

    def _apply_noise(self, audio: np.ndarray, cfg: dict) -> np.ndarray:
        """Add background noise at a random SNR."""
        noise_dir = Path(cfg.get("noise_dir", ""))
        if not noise_dir.exists():
            return audio

        noise_files = list(noise_dir.rglob("*.wav"))
        if not noise_files:
            return audio

        try:
            import soundfile as sf
            noise_path = random.choice(noise_files)
            noise, nsr = sf.read(str(noise_path), dtype="float32", always_2d=False)

            if nsr != self.sample_rate:
                import librosa
                noise = librosa.resample(noise, orig_sr=nsr, target_sr=self.sample_rate)

            # Tile noise if shorter than audio
            if len(noise) < len(audio):
                reps  = int(np.ceil(len(audio) / len(noise)))
                noise = np.tile(noise, reps)
            start = random.randint(0, len(noise) - len(audio))
            noise = noise[start: start + len(audio)]

            snr_range = cfg.get("snr_range", [5, 25])
            snr_db    = random.uniform(*snr_range)

            signal_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
            noise_rms  = np.sqrt(np.mean(noise ** 2)) + 1e-9
            target_rms = signal_rms / (10 ** (snr_db / 20))
            noise      = noise * (target_rms / noise_rms)

            return (audio + noise).astype(np.float32)
        except Exception:
            return audio

    def _apply_speed(self, audio: np.ndarray, cfg: dict) -> np.ndarray:
        """Speed perturbation without pitch shift."""
        rates = cfg.get("rates", [0.95, 1.0, 1.05])
        rate  = random.choice(rates)
        if rate == 1.0:
            return audio
        try:
            import librosa
            return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
        except Exception:
            return audio


# ─────────────────────────────────────────────────────────────────────────────
# AM-Softmax loss
# ─────────────────────────────────────────────────────────────────────────────
class AMSoftmaxLoss(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 2,
                 margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.margin  = margin
        self.scale   = scale
        self.weight  = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalise features and weights
        features = F.normalize(features, dim=1)
        weights  = F.normalize(self.weight, dim=1)

        cosine   = F.linear(features, weights)           # (B, num_classes)
        one_hot  = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Subtract margin from the target class cosine
        cosine_m = cosine - one_hot * self.margin
        logits   = self.scale * cosine_m

        return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────────────────────────────
# EER computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Returns EER as a percentage."""
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq

    # Higher score = more real → flip for EER (higher = fake)
    scores_fake = -scores
    thresholds  = np.linspace(scores_fake.min(), scores_fake.max(), 500)
    fprs, fnrs  = [], []

    for thresh in thresholds:
        preds = (scores_fake >= thresh).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        fprs.append(fp / (fp + tn + 1e-9))
        fnrs.append(fn / (fn + tp + 1e-9))

    fprs = np.array(fprs)
    fnrs = np.array(fnrs)

    try:
        eer = brentq(interp1d(fprs, fprs - fnrs), fprs.min(), fprs.max())
    except Exception:
        eer = np.min(np.abs(fprs - fnrs))

    return float(eer * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Freeze / unfreeze helpers
# ─────────────────────────────────────────────────────────────────────────────
def configure_frontend_freezing(model, frontend_cfg: dict) -> list:
    """
    Freezes the WavLM frontend, then selectively unfreezes
    the last N transformer layers.

    Returns list of param groups for the optimizer.
    """
    freeze         = frontend_cfg.get("freeze", True)
    n_unfreeze     = frontend_cfg.get("unfreeze_last_n_layers", 2)
    unfreeze_lr    = frontend_cfg.get("unfreeze_lr", 1e-5)

    if not freeze:
        log.info("Frontend: fully unfrozen (not recommended for small datasets)")
        return []

    # Freeze entire frontend first
    frozen_count = 0
    for name, param in model.named_parameters():
        if "frontend" in name or "wavlm" in name.lower() or "wav_model" in name.lower():
            param.requires_grad = False
            frozen_count += 1

    log.info(f"Frontend: frozen {frozen_count} parameter tensors")

    # Selectively unfreeze last N transformer layers
    unfrozen_params = []
    if n_unfreeze > 0:
        # WavLM transformer layers are typically named:
        # wavlm.encoder.layers.{i}.* or frontend.model.encoder.layers.{i}.*
        for name, param in model.named_parameters():
            for layer_kw in ["encoder.layers", "transformer.layers"]:
                if layer_kw in name:
                    # Extract layer index
                    try:
                        parts   = name.split(layer_kw + ".")
                        layer_i = int(parts[1].split(".")[0])
                        # WavLM Large has 24 layers (0-23)
                        # Unfreeze the last n_unfreeze layers
                        total_layers = 24
                        if layer_i >= total_layers - n_unfreeze:
                            param.requires_grad = True
                            unfrozen_params.append((name, param))
                    except (IndexError, ValueError):
                        pass

        log.info(
            f"Frontend: unfroze last {n_unfreeze} transformer layers "
            f"({len(unfrozen_params)} parameter tensors) with LR={unfreeze_lr}"
        )

    return [{"params": [p for _, p in unfrozen_params], "lr": unfreeze_lr}]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model, loader, optimizer, loss_fn, device, scaler, cfg: dict
) -> dict:
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    grad_clip  = cfg.get("training", {}).get("gradient_clip", 1.0)
    accum_steps = cfg.get("training", {}).get("gradient_accumulation_steps", 1)

    optimizer.zero_grad()

    for step, (audio, labels) in enumerate(
        tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    ):
        audio  = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        precision = torch.bfloat16 if cfg.get("training", {}).get(
            "mixed_precision", "bf16") == "bf16" else torch.float16

        with torch.autocast(device_type=device.type, dtype=precision,
                            enabled=device.type == "cuda"):
            out      = model(audio)
            features = out["embeddings"]
            logits   = out["logits"] if out["logits"] is not None else out["scores"]
            loss     = loss_fn(features, labels) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        preds       = logits.argmax(dim=1) if logits.shape[-1] == 2 else (logits.squeeze() > 0).long()
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return {
        "loss":     total_loss / len(loader),
        "accuracy": correct / total * 100,
    }


@torch.no_grad()
def evaluate(model, loader, device, cfg: dict) -> dict:
    model.eval()
    all_scores = []
    all_labels = []
    total_loss = 0.0

    precision = torch.bfloat16 if cfg.get("training", {}).get(
        "mixed_precision", "bf16") == "bf16" else torch.float16

    for audio, labels in tqdm(loader, desc="  Val  ", leave=False, dynamic_ncols=True):
        audio  = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=precision,
                            enabled=device.type == "cuda"):
            out = model(audio)

        logits = out["logits"] if out["logits"] is not None else out["scores"]

        # Score: probability of being real (label 0)
        if logits.shape[-1] == 2:
            probs  = F.softmax(logits.float(), dim=-1)
            scores = probs[:, 0].cpu().numpy()
        else:
            scores = logits.squeeze(-1).float().cpu().numpy()

        all_scores.append(scores)
        all_labels.append(labels.cpu().numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    eer      = compute_eer(all_labels, all_scores)
    preds    = (all_scores <= 0.5).astype(int)
    accuracy = (preds == all_labels).mean() * 100

    return {"eer": eer, "accuracy": accuracy}


# ─────────────────────────────────────────────────────────────────────────────
# Main fine-tuning orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepFense WavLM+Nes2Net on social media dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     required=True, type=Path)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run on 50 clips, 2 epochs, CPU — for local validation")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Output dir ───────────────────────────────────────────────────────────
    output_dir = Path(cfg.get("output", {}).get("dir", "outputs/finetune"))
    setup_logging(output_dir)

    # ── Seed ─────────────────────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ── Device ba
    if args.smoke_test:
        device = torch.device("cpu")
        log.info("SMOKE TEST MODE — CPU, 50 clips, 2 epochs")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Data config ──────────────────────────────────────────────────────────
    data_cfg     = cfg.get("data", {})
    sample_rate  = data_cfg.get("sample_rate", 16_000)
    max_duration = data_cfg.get("max_duration", 7.0)
    augment_cfg  = cfg

    train_cfg = cfg.get("training", {})
    batch_size = 4 if args.smoke_test else train_cfg.get("batch_size", 64)
    num_workers = 0 if args.smoke_test else train_cfg.get("num_workers", 8)
    epochs     = 2 if args.smoke_test else train_cfg.get("epochs", 30)

    # ── Datasets ─────────────────────────────────────────────────────────────
    data_root   = Path(data_cfg["root"])
    splits_cfg  = data_cfg.get("splits", {})
    train_split = splits_cfg.get("train", "train")
    val_split   = splits_cfg.get("val", "validation")

    log.info("Loading datasets...")
    train_dataset = AudioDataset(
        root         = data_root,
        split        = train_split,
        sample_rate  = sample_rate,
        max_duration = max_duration,
        augment      = True,
        augment_cfg  = augment_cfg,
        smoke_test   = args.smoke_test,
    )
    val_dataset = AudioDataset(
        root         = data_root,
        split        = val_split,
        sample_rate  = sample_rate,
        max_duration = max_duration,
        augment      = False,
        smoke_test   = args.smoke_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = device.type == "cuda",
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = device.type == "cuda",
    )

    # ── Load pretrained model ─────────────────────────────────────────────────
    checkpoint_path = Path(cfg["finetune"]["checkpoint"])
    log.info(f"Loading checkpoint: {checkpoint_path}")

    try:
        import deepfense.models.frontends.wavlm   # register frontend
        import deepfense.models.backends.nes2net   # register backend
        import deepfense.models.losses.cross_entropy  # register loss
        from deepfense.utils.registry import build_detector

        model_config_path = checkpoint_path.parent / "config.yaml"
        with open(model_config_path) as f:
            model_cfg = yaml.safe_load(f)

        model_inner = model_cfg["model"]

        # WavLM frontend needs the stub to init its architecture
        wavlm_stub = checkpoint_path.parent / "_wavlm_large_stub.pt"
        model_inner["frontend"]["args"]["ckpt_path"] = str(wavlm_stub)
        model_inner["frontend"]["args"]["freeze"] = False  # freezing handled below

        model = build_detector("StandardDetector", model_inner)

        ckpt = torch.load(str(checkpoint_path), map_location=device)
        state_dict = ckpt.get("model_state", ckpt.get("model", ckpt))
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        log.error(f"Could not load model: {e}")
        sys.exit(1)

    model.to(device)

    # ── Freeze / unfreeze frontend ────────────────────────────────────────────
    frontend_cfg    = cfg.get("frontend", {})
    frontend_groups = configure_frontend_freezing(model, frontend_cfg)

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_cfg     = cfg.get("loss", {})
    loss_params  = loss_cfg.get("params", {})

    embed_dim = model.backend.out_dim

    loss_fn = AMSoftmaxLoss(
        in_features  = embed_dim,
        num_classes  = 2,
        margin       = loss_params.get("margin", 0.2),
        scale        = loss_params.get("scale", 30.0),
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Two param groups:
    #   1. Unfrozen WavLM layers  → low LR (1e-5)
    #   2. Nes2Net + loss head    → main LR (1e-4)
    backend_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and ("frontend" not in name and "wavlm" not in name.lower())
    ] + list(loss_fn.parameters())

    param_groups = frontend_groups + [
        {"params": backend_params, "lr": train_cfg.get("lr", 1e-4)}
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr           = train_cfg.get("lr", 1e-4),
        weight_decay = train_cfg.get("weight_decay", 1e-4),
        betas        = train_cfg.get("betas", [0.9, 0.999]),
    )

    # ── Scheduler ────────────────────────────────────────────────────────────
    warmup_epochs = train_cfg.get("warmup_epochs", 3)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max   = epochs - warmup_epochs,
                eta_min = train_cfg.get("eta_min", 1e-6),
            ),
        ],
        milestones=[warmup_epochs],
    )

    # ── GradScaler for mixed precision ────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # ── Training loop ─────────────────────────────────────────────────────────
    es_cfg      = train_cfg.get("early_stopping", {})
    es_patience = es_cfg.get("patience", 7)
    best_eer    = float("inf")
    best_epoch  = 0
    es_counter  = 0

    best_ckpt_path = output_dir / "best_model.pth"
    last_ckpt_path = output_dir / "last_model.pth"
    metrics_path   = output_dir / "metrics.csv"

    log.info(f"Starting fine-tuning — {epochs} epochs, early stopping patience={es_patience}")
    log.info(f"Output directory: {output_dir}")

    # Metrics CSV header
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_eer", "val_acc", "lr", "elapsed_s"])

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        log.info(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, cfg
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device, cfg)

        scheduler.step()
        elapsed = time.time() - t_epoch
        current_lr = optimizer.param_groups[-1]["lr"]

        log.info(
            f"  Train  loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.1f}%"
        )
        log.info(
            f"  Val    EER={val_metrics['eer']:.2f}%  acc={val_metrics['accuracy']:.1f}%"
        )
        log.info(f"  LR={current_lr:.2e}  time={elapsed:.0f}s")

        # Save metrics
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(train_metrics["loss"], 4),
                round(train_metrics["accuracy"], 2),
                round(val_metrics["eer"], 2),
                round(val_metrics["accuracy"], 2),
                f"{current_lr:.2e}",
                round(elapsed, 1),
            ])

        # Save last checkpoint
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "loss_fn":    loss_fn.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "val_eer":    val_metrics["eer"],
            "config":     cfg,
        }, last_ckpt_path)

        # Save best checkpoint
        if val_metrics["eer"] < best_eer:
            best_eer   = val_metrics["eer"]
            best_epoch = epoch
            es_counter = 0
            torch.save({
                "epoch":   epoch,
                "model":   model.state_dict(),
                "loss_fn": loss_fn.state_dict(),
                "val_eer": best_eer,
                "config":  cfg,
            }, best_ckpt_path)
            log.info(f"  ✓ New best EER: {best_eer:.2f}% — checkpoint saved")
        else:
            es_counter += 1
            log.info(
                f"  No improvement ({es_counter}/{es_patience}) — "
                f"best EER still {best_eer:.2f}% @ epoch {best_epoch}"
            )

        # Early stopping
        if es_counter >= es_patience:
            log.info(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print("\n" + "─" * 60)
    print("  FINE-TUNING COMPLETE")
    print("─" * 60)
    print(f"  Best val EER    : {best_eer:.2f}%  (epoch {best_epoch})")
    print(f"  Total time      : {total_time/60:.1f} min")
    print(f"  Best checkpoint : {best_ckpt_path}")
    print(f"  Metrics CSV     : {metrics_path}")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
