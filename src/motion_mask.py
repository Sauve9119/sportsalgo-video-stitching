"""
src/motion_mask.py
------------------
Background subtraction and motion detection for the overlap zone.

Players and the ball cross the seam region continuously during a match.
If we blend these frames naively, we get ghosting — a player appears
semi-transparent or doubled. This module detects moving regions and
feeds a binary mask to the blending step to suppress that ghosting.

Two modes:
  - BackgroundSubtractor (MOG2): accurate, stateful, needs warm-up frames
  - FrameDiff:                   fast, stateless, good for per-frame use

Usage:
    from src.motion_mask import MotionDetector

    detector = MotionDetector(mode="mog2")
    for frame in frames:
        mask = detector.get_mask(frame)   # (H, W, 1) float32
        # pass mask to multiband_blend(...)
"""

import cv2
import numpy as np


class MotionDetector:
    """
    Detects moving pixels and returns a static-region mask for blending.

    Args:
        mode:          "mog2" | "framediff"
        threshold:     Pixel difference threshold (framediff mode).
        dilate_px:     Dilation radius to expand detected motion areas.
        restrict_to_x: Optional (x_start, x_end) tuple — limit detection
                       to the overlap zone only, for efficiency.
    """

    def __init__(self,
                 mode: str = "mog2",
                 threshold: float = 30.0,
                 dilate_px: int = 25,
                 restrict_to_x: tuple = None):

        self.mode          = mode
        self.threshold     = threshold
        self.dilate_px     = dilate_px
        self.restrict_to_x = restrict_to_x

        self._prev_gray    = None
        self._kernel       = None   # built on first call once we know frame size

        if mode == "mog2":
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200,
                varThreshold=50,
                detectShadows=False,
            )
        elif mode == "framediff":
            self._subtractor = None
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'mog2' or 'framediff'.")

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute a motion mask for the given frame.

        Returns:
            (H, W, 1) float32 array: 0.0 = moving, 1.0 = static.
        """
        h, w = frame.shape[:2]

        # Build dilation kernel once
        if self._kernel is None:
            ksize = max(3, self.dilate_px | 1)   # ensure odd
            self._kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (ksize, ksize)
            )

        # Optionally restrict to overlap zone
        if self.restrict_to_x is not None:
            x0, x1 = self.restrict_to_x
            region = frame[:, x0:x1]
        else:
            x0, x1 = 0, w
            region = frame

        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        if self.mode == "mog2":
            fg_mask = self._subtractor.apply(gray_region)
            # MOG2 returns 255 for foreground, 0 for background
            motion_binary = (fg_mask > 127).astype(np.uint8)

        else:  # framediff
            if self._prev_gray is None:
                self._prev_gray = gray_region
                return np.ones((h, w, 1), dtype=np.float32)

            diff = cv2.absdiff(gray_region, self._prev_gray)
            motion_binary = (diff > self.threshold).astype(np.uint8)
            self._prev_gray = gray_region

        # Dilate to cover player silhouette edges
        motion_dilated = cv2.dilate(motion_binary, self._kernel)

        # Build full-frame mask (everything outside overlap zone = static = 1.0)
        full_mask = np.ones((h, w), dtype=np.float32)
        full_mask[:, x0:x1] = 1.0 - motion_dilated.astype(np.float32)

        return full_mask[:, :, np.newaxis]

    def reset(self):
        """Reset internal state (useful when switching between match clips)."""
        self._prev_gray = None
        if self.mode == "mog2":
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=50, detectShadows=False
            )


def is_static_frame(frame: np.ndarray,
                     prev_frame: np.ndarray,
                     motion_threshold: float = 2.0) -> bool:
    """
    Returns True if there is very little motion between frame and prev_frame.
    Used during calibration to select background-only frames for homography.

    Args:
        frame:             Current frame.
        prev_frame:        Previous frame.
        motion_threshold:  Mean pixel difference threshold (0–255 scale).
                           2.0 is conservative — catches even slight crowd movement.

    Returns:
        True if the frame is sufficiently static.
    """
    gray_curr = cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mean_diff = np.abs(gray_curr - gray_prev).mean()
    return mean_diff < motion_threshold
