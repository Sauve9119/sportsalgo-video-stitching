"""
utils/visualise.py
------------------
Debug and evaluation visualisation helpers.
Produces side-by-side previews, seam overlays, and alignment validation frames.
"""

import cv2
import numpy as np
from pathlib import Path


def side_by_side(frame_l: np.ndarray,
                 frame_r: np.ndarray,
                 stitched: np.ndarray,
                 target_height: int = 360) -> np.ndarray:
    """
    Produce a debug frame showing [Left | Right | Stitched] side by side.
    All panels are resized to target_height for display.

    Args:
        frame_l:       Left camera frame (BGR).
        frame_r:       Right camera frame (BGR).
        stitched:      Stitched panoramic frame (BGR).
        target_height: Height in pixels for all panels.

    Returns:
        Concatenated BGR image.
    """
    def resize_to_height(img, h):
        ratio = h / img.shape[0]
        new_w = int(img.shape[1] * ratio)
        return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

    l_small = resize_to_height(frame_l, target_height)
    r_small = resize_to_height(frame_r, target_height)
    s_small = resize_to_height(stitched, target_height)

    # Draw label banners
    for img, label in [(l_small, "LEFT"), (r_small, "RIGHT"), (s_small, "STITCHED")]:
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (20, 20, 20), -1)
        cv2.putText(img, label, (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    # Add dividers
    divider = np.zeros((target_height, 3, 3), dtype=np.uint8)
    divider[:] = (80, 80, 80)

    return np.concatenate([l_small, divider, r_small, divider, s_small], axis=1)


def draw_seam_overlay(stitched: np.ndarray,
                       seam_x: int,
                       colour: tuple = (0, 255, 100),
                       alpha: float = 0.5) -> np.ndarray:
    """
    Draw a vertical seam indicator on the stitched frame.

    Args:
        stitched: Stitched panoramic frame.
        seam_x:   X coordinate of the seam centre.
        colour:   BGR colour of the seam line.
        alpha:    Blend alpha for the overlay.

    Returns:
        Frame with seam overlay drawn.
    """
    overlay = stitched.copy()
    cv2.line(overlay, (seam_x, 0), (seam_x, stitched.shape[0]), colour, 2)
    return cv2.addWeighted(overlay, alpha, stitched, 1 - alpha, 0)


def draw_keypoints_match(frame_l: np.ndarray,
                          frame_r: np.ndarray,
                          kp_l, kp_r, matches,
                          max_matches: int = 60) -> np.ndarray:
    """
    Draw SIFT match lines between left and right frames.
    Used during calibration to visually validate feature matching quality.

    Args:
        frame_l:     Left undistorted frame.
        frame_r:     Right undistorted frame.
        kp_l:        Keypoints from left frame.
        kp_r:        Keypoints from right frame.
        matches:     DMatch list (after ratio test filtering).
        max_matches: Cap on number of matches drawn.

    Returns:
        Side-by-side image with match lines drawn.
    """
    # Sample subset to keep image readable
    drawn = matches[:max_matches]
    vis = cv2.drawMatches(
        frame_l, kp_l,
        frame_r, kp_r,
        drawn, None,
        matchColor=(50, 200, 100),
        singlePointColor=(150, 150, 150),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis


def save_debug_frame(frame: np.ndarray, out_dir: str, name: str):
    """Save a single debug frame as JPEG."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(out_dir) / f"{name}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  [debug] saved → {path}")


def annotate_frame(frame: np.ndarray,
                    text: str,
                    pos: tuple = (20, 40),
                    scale: float = 0.8,
                    colour: tuple = (255, 255, 255)) -> np.ndarray:
    """Overlay a text annotation onto a frame (in-place copy)."""
    out = frame.copy()
    cv2.putText(out, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)
    return out
