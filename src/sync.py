"""
src/sync.py
-----------
Frame synchronisation between two camera feeds.

Strategy (in order of preference):
  1. Audio cross-correlation  — sub-frame accurate, requires audio tracks
  2. Visual event alignment   — finds the largest motion event (kick-off)
                                and aligns on that frame index

Both methods return a single integer `frame_offset`:
  positive → Camera R starts `frame_offset` frames LATER than Camera L
  negative → Camera R starts `frame_offset` frames EARLIER than Camera L

Usage:
    offset = compute_sync_offset(left_path, right_path, method="audio")
    # Then in the stitching loop:
    frame_r_idx = frame_l_idx + offset
"""

import cv2
import numpy as np
from pathlib import Path

try:
    import scipy.signal as signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from utils.video_io import VideoReader


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def compute_sync_offset(left_path: str,
                         right_path: str,
                         method: str = "auto",
                         verbose: bool = True) -> int:
    """
    Compute the frame offset between left and right camera videos.

    Args:
        left_path:  Path to left camera MP4.
        right_path: Path to right camera MP4.
        method:     "audio" | "visual" | "auto"
                    "auto" tries audio first, falls back to visual.
        verbose:    Print progress messages.

    Returns:
        Integer frame offset (add to left frame index to get right frame index).
    """
    if method == "auto":
        if SCIPY_AVAILABLE:
            try:
                offset = _audio_sync(left_path, right_path, verbose)
                if verbose:
                    print(f"[sync] Audio cross-correlation → offset = {offset} frames")
                return offset
            except Exception as e:
                if verbose:
                    print(f"[sync] Audio sync failed ({e}), falling back to visual")
        method = "visual"

    if method == "audio":
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for audio sync. pip install scipy")
        offset = _audio_sync(left_path, right_path, verbose)
        if verbose:
            print(f"[sync] Audio cross-correlation → offset = {offset} frames")
        return offset

    if method == "visual":
        offset = _visual_sync(left_path, right_path, verbose)
        if verbose:
            print(f"[sync] Visual event alignment → offset = {offset} frames")
        return offset

    raise ValueError(f"Unknown sync method: {method!r}. Use 'audio', 'visual', or 'auto'.")


# ─────────────────────────────────────────────
# Audio synchronisation
# ─────────────────────────────────────────────

def _extract_audio_signal(video_path: str,
                            sample_rate: int = 8000,
                            mono: bool = True) -> np.ndarray:
    """
    Extract audio from a video file as a 1D numpy array using OpenCV's
    VideoCapture audio backend (or fallback via raw PCM decode with ffmpeg subprocess).

    Returns:
        1D float32 numpy array of audio samples, or raises RuntimeError if unavailable.
    """
    # Try reading audio via ffmpeg subprocess (most reliable cross-platform)
    import subprocess
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",                        # no video
        "-acodec", "pcm_s16le",       # raw PCM 16-bit
        "-ar", str(sample_rate),      # resample to target rate
        "-ac", "1",                   # mono
        "-f", "s16le",                # raw format
        "pipe:1",                     # write to stdout
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120
    )
    if result.returncode != 0 or len(result.stdout) < 100:
        raise RuntimeError("ffmpeg audio extraction failed or no audio track")

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
    samples /= 32768.0  # Normalise to [-1, 1]
    return samples


def _audio_sync(left_path: str, right_path: str, verbose: bool) -> int:
    """
    Synchronise using audio cross-correlation.
    Returns frame offset (right relative to left).
    """
    if verbose:
        print("[sync] Extracting audio tracks...")

    reader_l = VideoReader(left_path)
    fps = reader_l.fps
    reader_l.close()

    audio_l = _extract_audio_signal(left_path)
    audio_r = _extract_audio_signal(right_path)

    if verbose:
        print(f"[sync] Audio lengths: L={len(audio_l)} R={len(audio_r)} samples @ 8kHz")

    # Trim to first 5 minutes (400K samples at 8kHz) for speed
    max_samples = 8000 * 300
    audio_l = audio_l[:max_samples]
    audio_r = audio_r[:max_samples]

    # Cross-correlate
    corr = signal.correlate(audio_l, audio_r, mode="full")
    lags = signal.correlation_lags(len(audio_l), len(audio_r), mode="full")

    # Peak lag in samples → convert to frames
    peak_sample_lag = lags[np.argmax(np.abs(corr))]
    frame_offset = int(round(peak_sample_lag / (8000 / fps)))

    return frame_offset


# ─────────────────────────────────────────────
# Visual synchronisation (fallback)
# ─────────────────────────────────────────────

def _compute_motion_curve(video_path: str,
                            sample_every: int = 5,
                            max_frames: int = 3000) -> np.ndarray:
    """
    Compute a 1D motion signal: mean absolute frame difference over time.
    Sampled every `sample_every` frames for speed.
    """
    reader = VideoReader(video_path)
    curve = []
    prev_gray = None

    for idx, frame in reader.read_frames(start=0, end=max_frames, step=sample_every):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)

        if prev_gray is not None:
            diff = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
            curve.append(float(diff.mean()))
        else:
            curve.append(0.0)

        prev_gray = gray

    reader.close()
    return np.array(curve, dtype=np.float32)


def _visual_sync(left_path: str, right_path: str, verbose: bool) -> int:
    """
    Synchronise by finding the largest common motion event (kick-off)
    in both cameras' motion curves and aligning them.
    Returns frame offset.
    """
    if verbose:
        print("[sync] Computing motion curves (this may take ~30s)...")

    sample_every = 5
    curve_l = _compute_motion_curve(left_path,  sample_every=sample_every)
    curve_r = _compute_motion_curve(right_path, sample_every=sample_every)

    if verbose:
        print(f"[sync] Motion curve lengths: L={len(curve_l)} R={len(curve_r)}")

    # Smooth curves to suppress noise
    kernel = np.ones(5) / 5
    curve_l = np.convolve(curve_l, kernel, mode="same")
    curve_r = np.convolve(curve_r, kernel, mode="same")

    # Find peak in each (likely kick-off moment)
    peak_l = int(np.argmax(curve_l))
    peak_r = int(np.argmax(curve_r))

    # Convert sampled indices back to real frame indices
    frame_peak_l = peak_l * sample_every
    frame_peak_r = peak_r * sample_every
    frame_offset = frame_peak_r - frame_peak_l

    if verbose:
        print(f"[sync] Peak motion: L@frame {frame_peak_l}, R@frame {frame_peak_r}")

    return frame_offset
