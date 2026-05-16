"""
utils/colour_match.py
---------------------
Histogram-based colour and exposure normalisation.
Matches Camera R's colour profile to Camera L in the overlap region
so seams don't show colour discontinuities.
"""

import cv2
import numpy as np


def match_histograms(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Adjust `source` so its per-channel histogram matches `reference`.
    Both images should be BGR uint8.

    Uses CDF-based histogram matching (same principle as scikit-image's
    match_histograms, implemented directly to avoid the extra dependency).

    Args:
        source:    Image whose colours will be adjusted.
        reference: Image whose colour distribution will be used as target.

    Returns:
        Colour-matched version of source (uint8 BGR).
    """
    matched = np.empty_like(source)
    for c in range(3):
        src_channel = source[:, :, c].ravel()
        ref_channel = reference[:, :, c].ravel()

        # Build CDFs
        src_hist, bins = np.histogram(src_channel, bins=256, range=(0, 256))
        ref_hist, _    = np.histogram(ref_channel, bins=256, range=(0, 256))

        src_cdf = src_hist.cumsum().astype(np.float64)
        ref_cdf = ref_hist.cumsum().astype(np.float64)
        src_cdf /= src_cdf[-1]
        ref_cdf /= ref_cdf[-1]

        # Build lookup table: for each src value, find the ref value
        # whose CDF is closest
        lut = np.zeros(256, dtype=np.uint8)
        ref_idx = 0
        for src_val in range(256):
            while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_val]:
                ref_idx += 1
            lut[src_val] = ref_idx

        matched[:, :, c] = lut[source[:, :, c]]

    return matched


def compute_overlap_stats(frame_l: np.ndarray,
                           frame_r: np.ndarray,
                           overlap_frac: float = 0.25) -> dict:
    """
    Compute mean and std of each channel in the overlap region of both frames.
    Useful for diagnosing exposure mismatch before correction.

    Args:
        frame_l:      Left camera frame (BGR).
        frame_r:      Right camera frame (BGR).
        overlap_frac: Fraction of width that constitutes the overlap zone.

    Returns:
        Dict with 'left_mean', 'right_mean', 'left_std', 'right_std' per channel.
    """
    h, w = frame_l.shape[:2]
    ol_w = int(w * overlap_frac)

    left_roi  = frame_l[:, w - ol_w:, :]
    right_roi = frame_r[:, :ol_w, :]

    def channel_stats(img):
        return {
            "mean": [float(img[:, :, c].mean()) for c in range(3)],
            "std":  [float(img[:, :, c].std())  for c in range(3)],
        }

    return {
        "left":  channel_stats(left_roi),
        "right": channel_stats(right_roi),
    }


def normalise_exposure(frame_l: np.ndarray,
                        frame_r: np.ndarray,
                        overlap_frac: float = 0.25) -> np.ndarray:
    """
    Scale Camera R's brightness to match Camera L using mean/std
    of the overlap region. Faster and lighter than full histogram matching;
    good for per-frame use in the stitching loop.

    Args:
        frame_l:      Left camera frame (BGR) — used as reference.
        frame_r:      Right camera frame (BGR) — will be adjusted.
        overlap_frac: Fraction of width used to compute overlap stats.

    Returns:
        Adjusted version of frame_r.
    """
    h, w = frame_l.shape[:2]
    ol_w = int(w * overlap_frac)

    left_roi  = frame_l[:, w - ol_w:, :].astype(np.float32)
    right_roi = frame_r[:, :ol_w, :].astype(np.float32)

    result = frame_r.astype(np.float32)

    for c in range(3):
        src_mean = right_roi[:, :, c].mean() + 1e-6
        ref_mean = left_roi[:, :, c].mean()
        src_std  = right_roi[:, :, c].std()  + 1e-6
        ref_std  = left_roi[:, :, c].std()

        # Linear scaling: shift mean, match std
        scale = ref_std / src_std
        shift = ref_mean - scale * src_mean
        result[:, :, c] = result[:, :, c] * scale + shift

    return np.clip(result, 0, 255).astype(np.uint8)
