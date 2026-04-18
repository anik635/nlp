"""
Block A: The Ear (Automatic Speech Recognition)
================================================
Uses OpenAI Whisper to transcribe .wav audio files into raw text.

Audio loading strategy (no ffmpeg, no torchcodec):
---------------------------------------------------
We use Python's built-in 'wave' module + numpy to read PCM bytes, convert
to float32, collapse to mono, and resample to 16 kHz with np.interp.
Whisper accepts a numpy float32 array directly alongside file paths.
"""

import wave
import whisper
import torch
import numpy as np
import time

_model_cache: dict = {}
WHISPER_SAMPLE_RATE = 16_000


def load_model(model_size: str = "base") -> whisper.Whisper:
    """Load (or return cached) Whisper model."""
    if model_size not in _model_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache[model_size] = whisper.load_model(model_size, device=device)
    return _model_cache[model_size]


def _load_audio_numpy(audio_path: str) -> np.ndarray:
    """
    Load a WAV file using stdlib 'wave' module. Returns 16 kHz mono float32.

    Steps:
      1. wave.open() reads the container metadata and raw PCM bytes.
      2. Convert PCM int16 -> float32 in [-1, 1]  (divide by 32768).
      3. Collapse stereo to mono by averaging channels.
      4. Resample from native rate (LJ Speech = 22050 Hz) to 16000 Hz
         using NumPy linear interpolation (np.interp).
    """
    with wave.open(audio_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        framerate  = wf.getframerate()
        n_frames   = wf.getnframes()
        raw_bytes  = wf.readframes(n_frames)

    # PCM bytes -> float32 normalised to [-1, 1]
    if sampwidth == 1:
        raw   = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        audio = (raw - 128.0) / 128.0
    elif sampwidth == 2:
        raw   = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        audio = raw / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.float32).copy()
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    # Stereo -> mono
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Resample to 16 kHz via linear interpolation if needed
    if framerate != WHISPER_SAMPLE_RATE:
        orig_len   = len(audio)
        target_len = int(orig_len * WHISPER_SAMPLE_RATE / framerate)
        audio      = np.interp(
            np.linspace(0, orig_len - 1, target_len),
            np.arange(orig_len),
            audio,
        ).astype(np.float32)

    return audio.astype(np.float32)


def transcribe(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcribe audio and return a result dict.

    How Whisper works internally:
      1. Float32 array padded/trimmed to 30 s (480 000 samples @ 16 kHz).
      2. Short-Time Fourier Transform -> 80-channel log-Mel spectrogram.
      3. CNN + Transformer Encoder reads spectral patterns into hidden states.
      4. Transformer Decoder generates BPE text tokens autoregressively.

    Returns
    -------
    dict keys: text, language, duration_s, device, segments
    """
    model  = load_model(model_size)
    device = next(model.parameters()).device.type

    audio_np = _load_audio_numpy(audio_path)

    t0      = time.perf_counter()
    result  = model.transcribe(audio_np, fp16=torch.cuda.is_available())
    elapsed = time.perf_counter() - t0

    return {
        "text":       result["text"].strip(),
        "language":   result.get("language", "en"),
        "duration_s": round(elapsed, 2),
        "device":     device,
        "segments":   result.get("segments", []),
    }
