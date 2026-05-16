"""
src/blend.py
------------
Multi-band Laplacian pyramid blending for seamless panoramic stitching.

Replaces linear alpha blending with a pyramid-based approach that blends
low frequencies (colour, brightness) over a wide zone and high frequencies
(sharp edges) only at the true seam — eliminating visible seam lines and
minimising ghosting from moving objects.

Reference: Burt & Adelson (1983), "A Multiresolution Spline With Application
           to Image Mosaics."
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# Gaussian / Laplacian Pyramid helpers
# ─────────────────────────────────────────────

def _build_gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    """Return a list of `levels` Gaussian pyramid images (float32)."""
    pyr = [img.astype(np.float32)]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _build_laplacian_pyramid(img: np.ndarray, levels: int) -> list:
    """Return a list of `levels` Laplacian pyramid images (float32)."""
    gauss = _build_gaussian_pyramid(img, levels)
    lap = []
    for i in range(levels - 1):
        size = (gauss[i].shape[1], gauss[i].shape[0])
        up = cv2.pyrUp(gauss[i + 1], dstsize=size)
        lap.append(gauss[i] - up)
    lap.append(gauss[-1])          # coarsest level is kept as-is
    return lap


def _reconstruct_from_laplacian(pyr: list) -> np.ndarray:
    """Collapse a Laplacian pyramid back to a full-resolution image."""
    img = pyr[-1].copy()
    for level in reversed(pyr[:-1]):
        size = (level.shape[1], level.shape[0])
        img = cv2.pyrUp(img, dstsize=size) + level
    return img


# ─────────────────────────────────────────────
# Blend mask construction
# ─────────────────────────────────────────────

def make_blend_mask(width: int,
                     height: int,
                     seam_x: int,
                     blend_width: int) -> np.ndarray:
    """
    Create a smooth horizontal alpha ramp centred at `seam_x`.

    Left of (seam_x - blend_width/2) → alpha = 1.0 (fully left image)
    Right of (seam_x + blend_width/2) → alpha = 0.0 (fully right image)
    In between → smooth cosine ramp

    Args:
        width:       Frame width (pixels).
        height:      Frame height (pixels).
        seam_x:      X coordinate of the seam centre.
        blend_width: Width of the transition zone in pixels.

    Returns:
        (H, W, 1) float32 array with values in [0, 1].
    """
    mask = np.zeros((height, width), dtype=np.float32)
    left_edge  = seam_x - blend_width // 2
    right_edge = seam_x + blend_width // 2

    mask[:, :left_edge]  = 1.0
    mask[:, right_edge:] = 0.0

    # Cosine ramp in the blend zone for smooth falloff
    if right_edge > left_edge:
        ramp_len = right_edge - left_edge
        xs = np.linspace(0, np.pi, ramp_len, dtype=np.float32)
        ramp = 0.5 * (1.0 + np.cos(xs))              # 1 → 0 cosine
        mask[:, left_edge:right_edge] = ramp[np.newaxis, :]

    return mask[:, :, np.newaxis]                     # (H, W, 1) for broadcasting


# ─────────────────────────────────────────────
# Motion mask (anti-ghosting)
# ─────────────────────────────────────────────

def compute_motion_mask(frame: np.ndarray,
                          prev_frame: np.ndarray,
                          threshold: float = 25.0,
                          dilate_px: int = 20) -> np.ndarray:
    """
    Detect moving pixels between consecutive frames.
    Returns a binary mask (0 = moving, 1 = static) to use as a blend weight
    modifier — suppresses ghosting of players crossing the seam.

    Args:
        frame:      Current frame (BGR).
        prev_frame: Previous frame (BGR).
        threshold:  Pixel difference threshold (0–255).
        dilate_px:  Dilation radius in pixels (expands detected motion region).

    Returns:
        (H, W, 1) float32 mask: 0.0 where motion detected, 1.0 elsewhere.
    """
    gray_curr = cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = np.abs(gray_curr - gray_prev)
    motion_binary = (diff > threshold).astype(np.uint8)

    # Dilate to cover player edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)
    )
    motion_dilated = cv2.dilate(motion_binary, kernel)

    # Invert: 1 = static (blend normally), 0 = moving (skip smooth blend)
    static_mask = (1 - motion_dilated).astype(np.float32)
    return static_mask[:, :, np.newaxis]


# ─────────────────────────────────────────────
# Main blending function
# ─────────────────────────────────────────────

def multiband_blend(img_l: np.ndarray,
                     img_r: np.ndarray,
                     seam_x: int,
                     blend_width: int = 120,
                     levels: int = 5,
                     motion_mask: np.ndarray = None) -> np.ndarray:
    """
    Blend img_l and img_r using a Laplacian pyramid blend in the overlap zone.

    Args:
        img_l:        Left warped frame (BGR, same canvas size as img_r).
        img_r:        Right warped frame (BGR, same canvas size as img_l).
        seam_x:       X coordinate of the seam centre.
        blend_width:  Width of the blending transition zone (pixels).
        levels:       Number of pyramid levels (5 is a good default).
        motion_mask:  Optional (H, W, 1) float32 mask from compute_motion_mask.
                      Where 0 (motion), blend_width is reduced to 1px to avoid
                      ghosting. Where 1 (static), normal multi-band blend.

    Returns:
        Blended panoramic frame (BGR uint8, same size as inputs).
    """
    h, w = img_l.shape[:2]

    # Smooth alpha mask
    alpha = make_blend_mask(w, h, seam_x, blend_width)         # (H, W, 1)

    # If motion mask provided, narrow blending where players are present
    if motion_mask is not None:
        # In moving regions: use hard cut (alpha unchanged but width → 1)
        hard_alpha = make_blend_mask(w, h, seam_x, 1)          # hard cut
        alpha = alpha * motion_mask + hard_alpha * (1.0 - motion_mask)

    # Build Laplacian pyramids for both images
    lap_l = _build_laplacian_pyramid(img_l.astype(np.float32), levels)
    lap_r = _build_laplacian_pyramid(img_r.astype(np.float32), levels)

    # Build Gaussian pyramid of the alpha mask (same depth)
    gauss_alpha = _build_gaussian_pyramid(alpha, levels)

    # Blend each level of the pyramid
    blended_pyr = []
    for i in range(levels):
        a = gauss_alpha[i]
        blended_level = lap_l[i] * a + lap_r[i] * (1.0 - a)
        blended_pyr.append(blended_level)

    # Reconstruct
    result = _reconstruct_from_laplacian(blended_pyr)
    return np.clip(result, 0, 255).astype(np.uint8)


def linear_blend(img_l: np.ndarray,
                  img_r: np.ndarray,
                  seam_x: int,
                  blend_width: int = 80) -> np.ndarray:
    """
    Simpler linear alpha blend. Faster but may show colour seams.
    Useful as a fast preview or fallback.

    Args:
        img_l:        Left warped frame (BGR).
        img_r:        Right warped frame (BGR).
        seam_x:       Seam X coordinate.
        blend_width:  Blend transition width.

    Returns:
        Blended frame (BGR uint8).
    """
    h, w = img_l.shape[:2]
    alpha = make_blend_mask(w, h, seam_x, blend_width)

    result = (img_l.astype(np.float32) * alpha +
              img_r.astype(np.float32) * (1.0 - alpha))
    return np.clip(result, 0, 255).astype(np.uint8)
