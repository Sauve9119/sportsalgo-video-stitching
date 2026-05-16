"""
src/stitch.py
-------------
Main panoramic video stitching pipeline.

Reads calibration from a saved .npz file (produced by calibrate.py),
then processes every frame of the match:
  1. Sync offset → read correct frame pair
  2. Undistort both frames
  3. Normalise exposure in the overlap zone
  4. Warp right frame → left canvas via homography H
  5. Detect motion in overlap zone (for ghosting suppression)
  6. Multi-band blend
  7. Write to output MP4

CLI Usage:
    python src/stitch.py \
        --left        data/left_match.mp4 \
        --right       data/right_match.mp4 \
        --calibration calibration/venue_01.npz \
        --out         output/stitched_match.mp4 \
        [--sync-method auto] \
        [--blend-width 120] \
        [--levels 5] \
        [--debug]
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibrate  import load_calibration, undistort_frame
from src.sync       import compute_sync_offset
from src.blend      import multiband_blend
from src.motion_mask import MotionDetector
from utils.video_io import VideoReader, VideoWriter
from utils.colour_match import normalise_exposure
from utils.visualise import side_by_side, draw_seam_overlay, annotate_frame, save_debug_frame


# ─────────────────────────────────────────────
# Canvas and warp helpers
# ─────────────────────────────────────────────

def compute_canvas_offset(H: np.ndarray,
                            frame_h: int,
                            frame_w: int,
                            out_w: int,
                            out_h: int) -> np.ndarray:
    """
    Compute a translation matrix T that shifts the left frame so both cameras
    fit on the output canvas without negative coordinates.

    Returns:
        H_adjusted — the homography to apply to the right frame on the full canvas.
        T          — the translation to apply to the left frame.
    """
    # Check if left frame starts at negative x after warping
    corners_r = np.float32([
        [0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]
    ]).reshape(-1, 1, 2)
    corners_w = cv2.perspectiveTransform(corners_r, H).reshape(-1, 2)

    min_x = min(0.0, corners_w[:, 0].min())
    min_y = min(0.0, corners_w[:, 1].min())

    # Translation matrix for left frame
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1     ],
    ], dtype=np.float64)

    # Adjust H to account for canvas offset
    H_adj = T @ H

    return H_adj, T


def warp_frame_onto_canvas(frame: np.ndarray,
                             H: np.ndarray,
                             out_w: int,
                             out_h: int) -> np.ndarray:
    """
    Warp a frame onto a canvas of size (out_h, out_w) using homography H.

    Args:
        frame: BGR frame to warp.
        H:     3×3 homography (adjusted for canvas offset).
        out_w: Canvas width.
        out_h: Canvas height.

    Returns:
        Warped frame on canvas (BGR, same canvas size).
    """
    return cv2.warpPerspective(
        frame, H, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def place_left_frame(frame: np.ndarray,
                      T: np.ndarray,
                      out_w: int,
                      out_h: int) -> np.ndarray:
    """
    Place the left frame on the canvas, translated by T.
    T is a 3×3 translation homography.
    """
    return cv2.warpPerspective(
        frame, T, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def find_seam_x(canvas_l: np.ndarray, canvas_r: np.ndarray) -> int:
    """
    Find the x coordinate where the left and right warped frames meet.
    Simple approach: find where left frame content ends (last nonzero col).

    Args:
        canvas_l: Left frame placed on canvas.
        canvas_r: Right frame warped onto canvas.

    Returns:
        Seam x coordinate.
    """
    gray_l = cv2.cvtColor(canvas_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(canvas_r, cv2.COLOR_BGR2GRAY)

    # Column mask: 1 where both images have content (overlap zone)
    has_l = (gray_l.max(axis=0) > 10).astype(np.uint8)
    has_r = (gray_r.max(axis=0) > 10).astype(np.uint8)
    overlap = has_l & has_r

    overlap_cols = np.where(overlap)[0]
    if len(overlap_cols) == 0:
        # No overlap found — use midpoint of L content end
        l_cols = np.where(has_l)[0]
        return int(l_cols[-1]) if len(l_cols) > 0 else canvas_l.shape[1] // 2

    # Use the centre of the overlap zone as the seam
    return int(overlap_cols[len(overlap_cols) // 2])


# ─────────────────────────────────────────────
# Main stitching loop
# ─────────────────────────────────────────────

def stitch_videos(left_path: str,
                   right_path: str,
                   calib_path: str,
                   out_path: str,
                   sync_method: str = "auto",
                   blend_width: int = 120,
                   pyramid_levels: int = 5,
                   motion_mode: str = "framediff",
                   debug: bool = False,
                   debug_dir: str = "output/debug_stitch",
                   verbose: bool = True) -> None:
    """
    Full stitching pipeline. Reads both videos, applies calibration,
    and writes panoramic output.

    Args:
        left_path:      Left camera MP4.
        right_path:     Right camera MP4.
        calib_path:     Calibration .npz file from calibrate.py.
        out_path:       Output MP4 path.
        sync_method:    "auto" | "audio" | "visual"
        blend_width:    Multi-band blend transition width (pixels).
        pyramid_levels: Laplacian pyramid depth.
        motion_mode:    "framediff" | "mog2"
        debug:          Save debug frames and side-by-side video.
        debug_dir:      Directory for debug output.
        verbose:        Print progress.
    """

    print("\n══════════════════════════════════════")
    print("  Panoramic Stitcher — Stitching")
    print("══════════════════════════════════════")

    # ── Load calibration ────────────────────────────────────────────────
    print(f"\n[1/5] Loading calibration from {calib_path}...")
    calib = load_calibration(calib_path)
    H_raw         = calib["H"]
    cam_mat_l     = calib["camera_matrix_l"]
    cam_mat_r     = calib["camera_matrix_r"]
    dist_l        = calib["dist_coeffs_l"]
    dist_r        = calib["dist_coeffs_r"]
    overlap_frac  = calib["overlap_frac"]

    # ── Open video readers ────────────────────────────────────────────────
    print("\n[2/5] Opening video files...")
    reader_l = VideoReader(left_path)
    reader_r = VideoReader(right_path)

    if verbose:
        print(f"  Left:  {reader_l}")
        print(f"  Right: {reader_r}")

    fps     = reader_l.fps
    frame_h = reader_l.height
    frame_w = reader_l.width

    # ── Compute canvas dimensions from H ─────────────────────────────────
    corners_r = np.float32([
        [0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]
    ]).reshape(-1, 1, 2)
    corners_w = cv2.perspectiveTransform(corners_r, H_raw).reshape(-1, 2)

    min_x = min(0.0, corners_w[:, 0].min())
    min_y = min(0.0, corners_w[:, 1].min())
    max_x = max(float(frame_w), corners_w[:, 0].max())
    max_y = max(float(frame_h), corners_w[:, 1].max())

    out_w = int(np.ceil(max_x - min_x))
    out_h = int(np.ceil(max_y - min_y))
    out_w = max(out_w, calib.get("output_width", out_w))
    out_h = max(out_h, calib.get("output_height", out_h))

    # Ensure minimum panoramic resolution
    out_w = max(out_w, 2560)
    out_h = max(out_h, frame_h)

    if verbose:
        print(f"  Output canvas: {out_w} × {out_h}")

    # Adjusted homographies for canvas placement
    H_adj, T = compute_canvas_offset(H_raw, frame_h, frame_w, out_w, out_h)

    # ── Frame sync ────────────────────────────────────────────────────────
    print("\n[3/5] Computing frame synchronisation offset...")
    frame_offset = compute_sync_offset(
        left_path, right_path, method=sync_method, verbose=verbose
    )
    if verbose:
        print(f"  Frame offset: {frame_offset} (right starts {'later' if frame_offset > 0 else 'earlier'})")

    # ── Setup writers and detectors ───────────────────────────────────────
    print("\n[4/5] Setting up output writers...")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = VideoWriter(out_path, fps, out_w, out_h)

    debug_writer = None
    if debug:
        debug_w = out_w // 2   # side-by-side is scaled to half height
        debug_h = out_h // 4 * 3  # approx
        debug_path = str(Path(out_path).with_stem(Path(out_path).stem + "_debug"))
        # We'll write debug frames at original resolution for clarity
        debug_writer = VideoWriter(debug_path, fps, out_w, out_h // 2)
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  Debug video → {debug_path}")

    motion_detector = MotionDetector(
        mode=motion_mode,
        dilate_px=25,
        restrict_to_x=(int(out_w * 0.3), int(out_w * 0.7)),   # overlap zone only
    )

    # ── Main frame loop ────────────────────────────────────────────────
    print("\n[5/5] Processing frames...")

    total_frames = reader_l.total_frames
    n_frames_to_process = total_frames - max(0, frame_offset)

    # Determine start indices
    start_l = max(0, -frame_offset)
    start_r = max(0,  frame_offset)

    reader_l.seek(start_l)
    reader_r.seek(start_r)

    seam_x = None    # Computed on first frame, reused after
    saved_debug_frames = 0
    t_start = time.time()

    for frame_idx in tqdm(range(n_frames_to_process), desc="Stitching",
                           unit="fr", disable=not verbose):

        ret_l, frame_l = reader_l.read()
        ret_r, frame_r = reader_r.read()

        if not ret_l or not ret_r:
            break

        # ── Undistort ──────────────────────────────────────────────────
        frame_l = undistort_frame(frame_l, cam_mat_l, dist_l)
        frame_r = undistort_frame(frame_r, cam_mat_r, dist_r)

        # ── Colour normalisation ───────────────────────────────────────
        frame_r = normalise_exposure(frame_l, frame_r, overlap_frac)

        # ── Warp onto canvas ───────────────────────────────────────────
        canvas_l = place_left_frame(frame_l, T, out_w, out_h)
        canvas_r = warp_frame_onto_canvas(frame_r, H_adj, out_w, out_h)

        # Compute seam once (it's stable for fixed cameras)
        if seam_x is None:
            seam_x = find_seam_x(canvas_l, canvas_r)
            if verbose:
                print(f"\n  Seam detected at x = {seam_x}")

        # ── Motion mask ────────────────────────────────────────────────
        motion_mask = motion_detector.get_mask(canvas_r)

        # ── Multi-band blend ───────────────────────────────────────────
        stitched = multiband_blend(
            canvas_l, canvas_r,
            seam_x=seam_x,
            blend_width=blend_width,
            levels=pyramid_levels,
            motion_mask=motion_mask,
        )

        # ── Crop/pad to target resolution ─────────────────────────────
        # Ensure output is exactly (out_h, out_w)
        if stitched.shape[0] != out_h or stitched.shape[1] != out_w:
            stitched = cv2.resize(stitched, (out_w, out_h),
                                   interpolation=cv2.INTER_LINEAR)

        writer.write(stitched)

        # ── Debug output ───────────────────────────────────────────────
        if debug:
            if frame_idx < 5 and saved_debug_frames < 5:
                seam_vis = draw_seam_overlay(stitched, seam_x)
                save_debug_frame(
                    seam_vis, debug_dir,
                    f"frame_{frame_idx:06d}_seam"
                )
                save_debug_frame(
                    stitched, debug_dir,
                    f"frame_{frame_idx:06d}_stitched"
                )
                saved_debug_frames += 1

            if debug_writer is not None:
                small = cv2.resize(stitched, (out_w, out_h // 2),
                                    interpolation=cv2.INTER_AREA)
                debug_writer.write(small)

    # ── Cleanup ─────────────────────────────────────────────────────────
    reader_l.close()
    reader_r.close()
    writer.close()
    if debug_writer:
        debug_writer.close()

    elapsed = time.time() - t_start
    fps_proc = frame_idx / elapsed if elapsed > 0 else 0

    print(f"\n✓ Stitching complete")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print(f"  Processing speed : {fps_proc:.1f} fps ({fps_proc / fps:.1f}× realtime)")
    print(f"  Output           : {out_path}")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Stitch two camera feeds into a panoramic video."
    )
    p.add_argument("--left",         required=True, help="Left camera MP4")
    p.add_argument("--right",        required=True, help="Right camera MP4")
    p.add_argument("--calibration",  required=True, help="Calibration .npz file")
    p.add_argument("--out",          required=True, help="Output MP4 path")
    p.add_argument("--sync-method",  default="auto",
                   choices=["auto", "audio", "visual"],
                   help="Synchronisation method (default: auto)")
    p.add_argument("--blend-width",  type=int, default=120,
                   help="Multi-band blend width in pixels (default: 120)")
    p.add_argument("--levels",       type=int, default=5,
                   help="Laplacian pyramid levels (default: 5)")
    p.add_argument("--motion-mode",  default="framediff",
                   choices=["framediff", "mog2"],
                   help="Motion detection mode (default: framediff)")
    p.add_argument("--debug",        action="store_true",
                   help="Save debug frames and side-by-side video")
    p.add_argument("--debug-dir",    default="output/debug_stitch",
                   help="Directory for debug output")
    p.add_argument("--quiet",        action="store_true",
                   help="Suppress progress output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stitch_videos(
        left_path=args.left,
        right_path=args.right,
        calib_path=args.calibration,
        out_path=args.out,
        sync_method=args.sync_method,
        blend_width=args.blend_width,
        pyramid_levels=args.levels,
        motion_mode=args.motion_mode,
        debug=args.debug,
        debug_dir=args.debug_dir,
        verbose=not args.quiet,
    )
