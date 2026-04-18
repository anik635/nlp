"""
Data Loader: LJ Speech Dataset Helper
=======================================
The LJ Speech Dataset contains:
  - 13,100 audio clips (24 hours total, all 22,050 Hz mono .wav)
  - metadata.csv -- pipe-delimited, 3 columns:
        ID | Transcription (raw) | Transcription (normalised)

Column meanings:
  1. ID              - filename without extension (e.g. LJ001-0001)
  2. raw_text        - exactly what was read, including abbreviations
  3. normalised_text - cleaned version (numbers expanded, etc.)
                       This is our "Ground Truth" for WER evaluation.
"""

import csv
import os
from pathlib import Path


def find_dataset(search_roots=None):
    """Locate the LJ Speech dataset root directory."""
    candidates = list(search_roots or []) + [
        r"d:\project\LJSpeech-1.1",
        os.path.join(os.getcwd(), "LJSpeech-1.1"),
    ]
    for c in candidates:
        p = Path(c)
        if p.is_dir() and (p / "metadata.csv").exists():
            return p
    return None


def load_metadata(dataset_root, max_samples: int = 100) -> list:
    """
    Parse metadata.csv and return up to max_samples records.

    Returns
    -------
    List of dicts:
        id           - LJ file ID  (e.g. "LJ001-0001")
        wav_path     - absolute path to the .wav file
        raw_text     - original reading text
        ground_truth - normalised text (used for WER evaluation)
        exists       - bool: whether the .wav file exists on disk
    """
    root  = Path(dataset_root)
    meta  = root / "metadata.csv"
    wavs  = root / "wavs"
    records = []

    with open(meta, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            if len(row) < 2:
                continue

            file_id    = row[0].strip()
            raw_text   = row[1].strip() if len(row) > 1 else ""
            normalised = row[2].strip() if len(row) > 2 else raw_text
            wav_path   = wavs / f"{file_id}.wav"

            records.append({
                "id":           file_id,
                "wav_path":     str(wav_path),
                "raw_text":     raw_text,
                "ground_truth": normalised,
                "exists":       wav_path.exists(),
            })

    return records
