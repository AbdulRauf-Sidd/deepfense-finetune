import subprocess
import os
from pathlib import Path

def convert_to_wav_mono_16k(input_path, output_path=None):
    """
    Converts any audio/video file to 16kHz mono WAV.
    Works on mp4, mp3, m4a, aac, ogg, flac, webm, etc.
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix('.wav')
    
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-ar', '16000',      # resample to 16kHz
        '-ac', '1',          # mix down to mono
        '-sample_fmt', 's16', # 16-bit PCM
        '-y',                 # overwrite if exists
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error converting {input_path}: {result.stderr}")
        return None
    
    return output_path


def batch_convert(input_dir, output_dir=None, extensions=None):
    """
    Recursively converts all audio files in a directory.
    """
    if extensions is None:
        extensions = ['.mp4', '.mp3', '.m4a', '.aac', 
                      '.ogg', '.flac', '.webm', '.opus', '.wav']
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = [f for f in input_dir.rglob('*') 
             if f.suffix.lower() in extensions]
    
    print(f"Found {len(files)} files to convert...")
    
    for i, f in enumerate(files):
        # Preserve subdirectory structure
        relative = f.relative_to(input_dir)
        out_path = output_dir / relative.with_suffix('.wav')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = convert_to_wav_mono_16k(f, out_path)
        
        if result:
            print(f"[{i+1}/{len(files)}] {f.name} → {out_path.name}")
        else:
            print(f"[{i+1}/{len(files)}] FAILED: {f.name}")


# Usage
batch_convert('data/processed/real/', 'data/mono/real/')
batch_convert('data/processed/fake/', 'data/mono/fake/')