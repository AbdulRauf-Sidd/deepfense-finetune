"""
Audio preprocessing: silence trimming and chunking.

Usage:
    python preprocess.py \
        --real-dir  data/real \
        --fake-dir  data/fake \
        --output-dir data/processed \
        [--chunk-duration 5.0] \
        [--sr 16000] \
        [--top-db 30]
"""

import argparse
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aiff", ".aif"}


def _find_audio_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def _load(path: Path, sr: int) -> np.ndarray:
    x, orig_sr = librosa.load(str(path), sr=None, mono=True)
    if orig_sr != sr:
        x = librosa.resample(x, orig_sr=orig_sr, target_sr=sr)
    return x.astype(np.float32)


def _save(samples: np.ndarray, path: Path, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr, subtype="PCM_16")


def trim_and_chunk(
    real_dir: str | Path,
    fake_dir: str | Path,
    output_dir: str | Path,
    chunk_duration: float = 5.0,
    sr: int = 16000,
    top_db: float = 30.0,
):
    """
    Trim silence from each audio file in real_dir and fake_dir, then split
    files longer than chunk_duration seconds into fixed-length chunks.

    Processed files are written to:
        output_dir/real/<stem>[_chunk<N>].wav
        output_dir/fake/<stem>[_chunk<N>].wav

    Files shorter than or equal to chunk_duration after trimming are saved
    as a single file (no suffix appended).

    Args:
        real_dir:       Folder containing real (bonafide) audio files.
        fake_dir:       Folder containing fake (spoof) audio files.
        output_dir:     Root folder where processed files are written.
        chunk_duration: Maximum chunk length in seconds (default 5.0).
        sr:             Target sample rate for loading and saving (default 16000).
        top_db:         Threshold (dB below peak) for silence trimming (default 30).
    """
    real_dir   = Path(real_dir)
    fake_dir   = Path(fake_dir)
    output_dir = Path(output_dir)

    chunk_samples = int(chunk_duration * sr)
    splits = [("real", real_dir), ("fake", fake_dir)]

    total_in = total_out = skipped = 0

    for label, src_dir in splits:
        files = _find_audio_files(src_dir)
        if not files:
            log.warning(f"No audio files found in {src_dir}")
            continue

        out_subdir = output_dir / label
        log.info(f"Processing {len(files)} {label} file(s) → {out_subdir}")

        for path in tqdm(files, desc=label, unit="file"):
            try:
                x = _load(path, sr)
            except Exception as e:
                log.warning(f"Skipping {path.name}: {e}")
                skipped += 1
                continue

            # Trim leading/trailing silence
            x_trimmed, _ = librosa.effects.trim(x, top_db=top_db)
            total_in += 1

            if len(x_trimmed) == 0:
                log.warning(f"Skipping {path.name}: empty after trimming")
                skipped += 1
                continue

            stem = path.stem

            if len(x_trimmed) <= chunk_samples:
                _save(x_trimmed, out_subdir / f"{stem}.wav", sr)
                total_out += 1
            else:
                n_chunks = (len(x_trimmed) + chunk_samples - 1) // chunk_samples
                for i in range(n_chunks):
                    chunk = x_trimmed[i * chunk_samples : (i + 1) * chunk_samples]
                    _save(chunk, out_subdir / f"{stem}_chunk{i:03d}.wav", sr)
                    total_out += 1

    log.info(
        f"Done. {total_in} file(s) processed → {total_out} output file(s) written"
        + (f", {skipped} skipped" if skipped else "")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Trim silence and chunk audio files")
    p.add_argument("--real-dir",       required=True)
    p.add_argument("--fake-dir",       required=True)
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--chunk-duration", type=float, default=5.0,
                   help="Max chunk length in seconds (default: 5.0)")
    p.add_argument("--sr",             type=int,   default=16000,
                   help="Target sample rate (default: 16000)")
    p.add_argument("--top-db",         type=float, default=30.0,
                   help="Silence threshold in dB below peak (default: 30)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trim_and_chunk(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        output_dir=args.output_dir,
        chunk_duration=args.chunk_duration,
        sr=args.sr,
        top_db=args.top_db,
    )
