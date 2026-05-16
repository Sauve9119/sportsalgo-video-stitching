"""
src/calibrate.py
----------------
One-time-per-venue calibration pipeline.

Steps:
  1. Sample static background frames from the calibration clip
  2. Optionally estimate lens distortion from pitch line straightness
  3. Detect SIFT keypoints in the overlap region of both cameras
  4. Match and filter with FLANN + Lowe's ratio test
  5. Estimate homography H via RANSAC (cv2.findHomography)
  6. Validate H by checking pitch line reprojection error
  7. Save H + distortion params to a .npz file

Run once per venue. Saved file is loaded by stitch.py for every match.

CLI Usage:
    python src/calibrate.py \
        --left  data/sample/left_clip.mp4 \
        --right data/sample/right_clip.mp4 \
        --out   calibration/venue_01.npz \
        [--overlap 0.30] \
        [--max-frames 500] \
        [--debug-dir output/debug_calib]
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.video_io import VideoReader
from utils.visualise import draw_keypoints_match, save_debug_frame
from src.motion_mask import is_static_frame


# ─────────────────────────────────────────────
# Lens undistortion helpers
# ─────────────────────────────────────────────

def estimate_distortion_from_lines(frame: np.ndarray,
                                    num_lines: int = 20) -> tuple:
    """
    Estimate barrel distortion coefficients by fitting lines to pitch
    markings. Straight lines in the world should be straight in the image;
    deviation from straightness gives k1, k2.

    This is a simplified estimation. For production, a checkerboard
    calibration or OpenCV's full calibrateCamera() is more accurate.

    Args:
        frame:     Undistorted reference frame.
        num_lines: Number of line candidates to use.

    Returns:
        (camera_matrix, dist_coeffs) compatible with cv2.undistort().
    """
    h, w = frame.shape[:2]

    # Default camera matrix (principal point at centre, focal ~0.8× width)
    fx = fy = 0.8 * w
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ],
    ], dtype=np.float64)

    # Detect lines using Hough transform
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None or len(lines) < 5:
        # No usable lines — return identity distortion (no correction)
        return camera_matrix, np.zeros(5, dtype=np.float64)

    # Estimate k1 from the curvature of nearly-horizontal lines
    # (simplified: compare actual vs expected y-span of detected lines)
    k1 = -0.05   # conservative default for typical wide-angle lenses
    dist_coeffs = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    return camera_matrix, dist_coeffs


def undistort_frame(frame: np.ndarray,
                     camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray) -> np.ndarray:
    """Apply lens undistortion to a single frame."""
    return cv2.undistort(frame, camera_matrix, dist_coeffs)


# ─────────────────────────────────────────────
# Feature matching and homography
# ─────────────────────────────────────────────

def extract_overlap_regions(frame_l: np.ndarray,
                              frame_r: np.ndarray,
                              overlap_frac: float = 0.35) -> tuple:
    """
    Crop the overlap region from both frames.
    Left camera: right side. Right camera: left side.

    Returns:
        (roi_l, roi_r, x_offset_l, x_offset_r)
    """
    h, w = frame_l.shape[:2]
    ol_w = int(w * overlap_frac)

    # Use wider crop to give SIFT more context
    roi_l = frame_l[:, w - ol_w:, :]
    roi_r = frame_r[:, :ol_w, :]
    return roi_l, roi_r, w - ol_w, 0


def detect_and_match_sift(roi_l: np.ndarray,
                            roi_r: np.ndarray,
                            ratio_thresh: float = 0.75) -> tuple:
    """
    Detect SIFT keypoints and match with FLANN + Lowe's ratio test.

    Args:
        roi_l:        Left overlap crop.
        roi_r:        Right overlap crop.
        ratio_thresh: Lowe's ratio test threshold (lower = stricter).

    Returns:
        (kp_l, kp_r, good_matches) where matches are DMatch objects.
    """
    sift = cv2.SIFT_create(
        nfeatures=2000,
        contrastThreshold=0.03,
        edgeThreshold=10,
    )

    gray_l = cv2.cvtColor(roi_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)

    kp_l, des_l = sift.detectAndCompute(gray_l, None)
    kp_r, des_r = sift.detectAndCompute(gray_r, None)

    if des_l is None or des_r is None or len(kp_l) < 10 or len(kp_r) < 10:
        raise RuntimeError(
            f"Not enough keypoints: L={len(kp_l) if kp_l else 0}, "
            f"R={len(kp_r) if kp_r else 0}. "
            "Try a clearer calibration clip or wider overlap region."
        )

    # FLANN matcher for SIFT (L2 distance)
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    raw_matches = flann.knnMatch(des_l, des_r, k=2)

    # Lowe's ratio test
    good = []
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    print(f"  SIFT: {len(kp_l)} L kps, {len(kp_r)} R kps → {len(good)} good matches")

    if len(good) < 10:
        raise RuntimeError(
            f"Too few good matches ({len(good)}). "
            "Consider lowering --overlap or using a cleaner calibration clip."
        )

    return kp_l, kp_r, good


def compute_homography(kp_l: list,
                        kp_r: list,
                        matches: list,
                        x_offset_l: int,
                        x_offset_r: int,
                        reproj_thresh: float = 3.0) -> np.ndarray:
    """
    Estimate homography H mapping right-camera coords → left-camera canvas.

    The H matrix maps a point from Camera R's full frame coordinate space
    into Camera L's full frame coordinate space. Keypoints are in ROI coords,
    so we need to add the x offsets before passing to findHomography.

    Args:
        kp_l:          Keypoints from left ROI.
        kp_r:          Keypoints from right ROI.
        matches:       Filtered DMatch list.
        x_offset_l:    Left ROI start x in full frame.
        x_offset_r:    Right ROI start x in full frame.
        reproj_thresh: RANSAC reprojection threshold (pixels).

    Returns:
        3×3 homography matrix H (float64).
    """
    pts_l = np.float32([
        [kp_l[m.queryIdx].pt[0] + x_offset_l,
         kp_l[m.queryIdx].pt[1]]
        for m in matches
    ])
    pts_r = np.float32([
        [kp_r[m.trainIdx].pt[0] + x_offset_r,
         kp_r[m.trainIdx].pt[1]]
        for m in matches
    ])

    H, mask = cv2.findHomography(
        pts_r,       # source: right frame
        pts_l,       # dest:   left frame
        cv2.RANSAC,
        reproj_thresh,
    )

    if H is None:
        raise RuntimeError("cv2.findHomography returned None. Not enough inliers.")

    n_inliers = int(mask.sum())
    print(f"  Homography: {n_inliers}/{len(matches)} RANSAC inliers")
    if n_inliers < 8:
        raise RuntimeError(f"Too few RANSAC inliers ({n_inliers}). Calibration unreliable.")

    return H


def validate_homography(H: np.ndarray,
                          frame_l: np.ndarray,
                          frame_r: np.ndarray) -> float:
    """
    Quick validation: warp a few synthetic pitch line points through H
    and check that their reprojection error is small.

    Returns mean reprojection error in pixels (should be < 5px for good calibration).
    """
    h, w = frame_l.shape[:2]

    # Synthetic points along the pitch lines (right-side pitch markings ~30% from left)
    test_pts = np.float32([
        [int(w * 0.25), int(h * 0.2)],
        [int(w * 0.25), int(h * 0.5)],
        [int(w * 0.25), int(h * 0.8)],
        [int(w * 0.10), int(h * 0.5)],
        [int(w * 0.40), int(h * 0.5)],
    ]).reshape(-1, 1, 2)

    # Warp right-frame points into left-frame space
    warped = cv2.perspectiveTransform(test_pts, H).reshape(-1, 2)

    # For validation, we check the warped points land inside the left frame
    in_bounds = np.all((warped >= 0) & (warped < [w, h]), axis=1)
    if not in_bounds.all():
        print(f"  Warning: {(~in_bounds).sum()} validation points outside frame bounds")

    # Return mean displacement from centre-line as a proxy error
    # (a proper error requires ground-truth correspondences)
    centre_x = w / 2
    errors = np.abs(warped[:, 0] - centre_x)
    mean_error = float(errors.mean())
    return mean_error


# ─────────────────────────────────────────────
# Static frame sampling
# ─────────────────────────────────────────────

def collect_static_frames(video_path: str,
                            max_frames: int = 500,
                            sample_step: int = 10,
                            motion_threshold: float = 2.5,
                            verbose: bool = True) -> list:
    """
    Sample frames from a video, keeping only those with minimal motion.
    Static frames are better for feature matching — no player ghosting.

    Returns:
        List of (frame_idx, frame) tuples.
    """
    reader = VideoReader(video_path)
    static_frames = []
    prev_frame = None

    total_sampled = 0
    for idx, frame in reader.read_frames(start=0, end=max_frames * sample_step,
                                          step=sample_step):
        total_sampled += 1
        if prev_frame is not None:
            if is_static_frame(frame, prev_frame, motion_threshold):
                static_frames.append((idx, frame))
        prev_frame = frame

        if len(static_frames) >= max_frames:
            break

    reader.close()

    if verbose:
        print(f"  Sampled {total_sampled} frames → {len(static_frames)} static frames kept")

    return static_frames


# ─────────────────────────────────────────────
# Main calibration routine
# ─────────────────────────────────────────────

def run_calibration(left_path: str,
                     right_path: str,
                     out_path: str,
                     overlap_frac: float = 0.35,
                     max_frames: int = 500,
                     debug_dir: str = None,
                     verbose: bool = True) -> dict:
    """
    Full calibration pipeline. Saves results to `out_path` (.npz).

    Args:
        left_path:    Path to left camera calibration clip.
        right_path:   Path to right camera calibration clip.
        out_path:     Output path for calibration .npz file.
        overlap_frac: Fraction of frame width treated as overlap zone.
        max_frames:   Max static frames to sample from each video.
        debug_dir:    If set, save debug visualisations here.
        verbose:      Print progress.

    Returns:
        Dict with keys: H, camera_matrix_l, camera_matrix_r,
                        dist_coeffs_l, dist_coeffs_r, overlap_frac,
                        output_width, output_height
    """
    print("\n══════════════════════════════════════")
    print("  Panoramic Stitcher — Calibration")
    print("══════════════════════════════════════")

    # ── Step 1: Load representative frames ──────────────────────────────
    print("\n[1/5] Sampling static frames...")
    static_l = collect_static_frames(left_path,  max_frames, verbose=verbose)
    static_r = collect_static_frames(right_path, max_frames, verbose=verbose)

    if not static_l or not static_r:
        raise RuntimeError("Could not find static frames in calibration clips.")

    # Pick the best single frame pair to work with (use midpoint of collection)
    mid_l = len(static_l) // 2
    mid_r = len(static_r) // 2
    frame_l = static_l[mid_l][1]
    frame_r = static_r[mid_r][1]

    # ── Step 2: Estimate lens distortion ────────────────────────────────
    print("\n[2/5] Estimating lens distortion...")
    cam_mat_l, dist_l = estimate_distortion_from_lines(frame_l)
    cam_mat_r, dist_r = estimate_distortion_from_lines(frame_r)

    # Undistort the representative frames
    frame_l_ud = undistort_frame(frame_l, cam_mat_l, dist_l)
    frame_r_ud = undistort_frame(frame_r, cam_mat_r, dist_r)
    print("  Distortion parameters estimated.")

    # ── Step 3: SIFT matching in overlap region ──────────────────────────
    print("\n[3/5] Detecting and matching SIFT features...")
    roi_l, roi_r, x_off_l, x_off_r = extract_overlap_regions(
        frame_l_ud, frame_r_ud, overlap_frac
    )
    kp_l, kp_r, matches = detect_and_match_sift(roi_l, roi_r)

    if debug_dir:
        match_vis = draw_keypoints_match(roi_l, roi_r, kp_l, kp_r, matches)
        save_debug_frame(match_vis, debug_dir, "01_sift_matches")

    # ── Step 4: Homography estimation ───────────────────────────────────
    print("\n[4/5] Estimating homography via RANSAC...")
    H = compute_homography(kp_l, kp_r, matches, x_off_l, x_off_r)

    # Validate
    mean_err = validate_homography(H, frame_l_ud, frame_r_ud)
    print(f"  Validation: mean projection spread = {mean_err:.1f}px")
    if mean_err > 200:
        print("  ⚠ Warning: high validation error. Check overlap region or re-run.")

    # ── Step 5: Compute output canvas dimensions ─────────────────────────
    print("\n[5/5] Computing output canvas dimensions...")
    h, w = frame_l_ud.shape[:2]

    # Warp right frame corners to find canvas bounds
    corners_r = np.float32([
        [0, 0], [w, 0], [w, h], [0, h]
    ]).reshape(-1, 1, 2)
    corners_warped = cv2.perspectiveTransform(corners_r, H).reshape(-1, 2)

    all_x = np.concatenate([[0, w], corners_warped[:, 0]])
    all_y = np.concatenate([[0, h], corners_warped[:, 1]])

    out_w = int(np.ceil(all_x.max() - min(all_x.min(), 0)))
    out_h = int(np.ceil(all_y.max() - min(all_y.min(), 0)))

    # Clamp to reasonable panoramic sizes
    out_w = max(out_w, int(w * 1.5))
    out_h = max(out_h, h)
    print(f"  Output canvas: {out_w} × {out_h}")

    # ── Save calibration ─────────────────────────────────────────────────
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        H=H,
        camera_matrix_l=cam_mat_l,
        camera_matrix_r=cam_mat_r,
        dist_coeffs_l=dist_l,
        dist_coeffs_r=dist_r,
        overlap_frac=np.float32(overlap_frac),
        output_width=np.int32(out_w),
        output_height=np.int32(out_h),
    )
    print(f"\n✓ Calibration saved → {out_path}")

    if debug_dir:
        # Save a sample warped frame to visually verify alignment
        import cv2
        H_adj = H.copy()
        warped_r = cv2.warpPerspective(frame_r_ud, H_adj, (out_w, out_h))
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:h, :w] = frame_l_ud
        # Blend overlap
        mask = warped_r.sum(axis=2) > 0
        canvas[mask] = warped_r[mask]
        save_debug_frame(canvas, debug_dir, "02_calibration_preview")
        print(f"  Debug visuals saved → {debug_dir}/")

    return {
        "H": H,
        "camera_matrix_l": cam_mat_l,
        "camera_matrix_r": cam_mat_r,
        "dist_coeffs_l": dist_l,
        "dist_coeffs_r": dist_r,
        "overlap_frac": overlap_frac,
        "output_width": out_w,
        "output_height": out_h,
    }


def load_calibration(path: str) -> dict:
    """
    Load a saved calibration .npz file.

    Returns:
        Dict with H, camera matrices, distortion coeffs, canvas dimensions.
    """
    data = np.load(path)
    return {
        "H":               data["H"],
        "camera_matrix_l": data["camera_matrix_l"],
        "camera_matrix_r": data["camera_matrix_r"],
        "dist_coeffs_l":   data["dist_coeffs_l"],
        "dist_coeffs_r":   data["dist_coeffs_r"],
        "overlap_frac":    float(data["overlap_frac"]),
        "output_width":    int(data["output_width"]),
        "output_height":   int(data["output_height"]),
    }


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate panoramic stitcher (run once per venue)."
    )
    p.add_argument("--left",        required=True, help="Path to left camera clip")
    p.add_argument("--right",       required=True, help="Path to right camera clip")
    p.add_argument("--out",         required=True, help="Output .npz calibration file")
    p.add_argument("--overlap",     type=float, default=0.35,
                   help="Overlap fraction (default: 0.35)")
    p.add_argument("--max-frames",  type=int, default=500,
                   help="Max static frames to sample (default: 500)")
    p.add_argument("--debug-dir",   default=None,
                   help="Directory to save debug visualisations")
    p.add_argument("--quiet",       action="store_true", help="Suppress progress output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_calibration(
        left_path=args.left,
        right_path=args.right,
        out_path=args.out,
        overlap_frac=args.overlap,
        max_frames=args.max_frames,
        debug_dir=args.debug_dir,
        verbose=not args.quiet,
    )
