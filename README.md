# Panoramic Video Stitcher — SportsAlgo × ClayGrounds

An open, end-to-end pipeline that takes footage from **two fixed wide-angle cameras** mounted at a football venue and produces a single seamless **2560×720 panoramic video** — no proprietary hardware required.

Built for the SportsAlgo × ClayGrounds Hackathon.

---

## Pipeline Overview

```
Camera L (MP4)  +  Camera R (MP4)
         │                │
         └──── Sync ──────┘
                  │
         Lens Undistortion
                  │
       Homography Estimation
       (SIFT + RANSAC — once per venue)
                  │
         Perspective Warp
                  │
     Multi-band Blend + Motion Mask
                  │
        Panoramic Output (MP4)
```

---

## Project Structure

```
panoramic-stitcher/
├── src/
│   ├── calibrate.py          # Lens undistortion + homography (run once per venue)
│   ├── stitch.py             # Per-frame warp + blend (main pipeline)
│   ├── sync.py               # Frame synchronisation via audio/visual alignment
│   ├── blend.py              # Multi-band Laplacian pyramid blending
│   └── motion_mask.py        # Moving object detection for ghosting prevention
├── utils/
│   ├── video_io.py           # FFmpeg-backed video reader/writer
│   ├── colour_match.py       # Histogram-based colour/exposure normalisation
│   └── visualise.py          # Debug helpers — seam overlay, side-by-side preview
├── calibration/
│   └── (saved .npz files go here after running calibrate.py)
├── output/
│   └── (stitched videos written here)
├── data/
│   └── sample/               # Put your test clips here
├── requirements.txt
├── run_calibration.sh
├── run_stitch.sh
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+, OpenCV 4.8+, FFmpeg on PATH.

### 2. Run calibration (once per venue setup)

```bash
python src/calibrate.py \
  --left  data/sample/left_clip.mp4 \
  --right data/sample/right_clip.mp4 \
  --out   calibration/venue_01.npz
```

This saves the homography matrix `H` and lens distortion coefficients to `calibration/venue_01.npz`.

### 3. Stitch a full match

```bash
python src/stitch.py \
  --left        data/sample/left_match.mp4 \
  --right       data/sample/right_match.mp4 \
  --calibration calibration/venue_01.npz \
  --out         output/stitched_match.mp4
```

### 4. Preview with side-by-side debug view

```bash
python src/stitch.py \
  --left        data/sample/left_match.mp4 \
  --right       data/sample/right_match.mp4 \
  --calibration calibration/venue_01.npz \
  --out         output/stitched_match.mp4 \
  --debug
```

---

## Evaluation Criteria Coverage

| Criterion | Approach |
|---|---|
| **Alignment accuracy (30%)** | SIFT + RANSAC on pitch line intersections; validated against centre circle continuity |
| **Seam quality (25%)** | Multi-band Laplacian pyramid blend + histogram colour match |
| **Motion handling (25%)** | Per-frame motion mask excludes moving players from blend zone |
| **Reusability (10%)** | H matrix + distortion params saved to `.npz`, loaded at stitch time |
| **Performance (10%)** | Frame-parallel processing; targets 2–5× realtime offline |

---

## Technical Details

### Lens Undistortion
Wide-angle barrel distortion is corrected using pitch line straightness as a constraint — no checkerboard required. OpenCV's `cv2.undistort()` is applied per frame using saved `camera_matrix` and `dist_coeffs`.

### Homography Estimation
1. Extract static background frames (low inter-frame difference)
2. Detect SIFT keypoints in the overlap region only
3. Match with FLANN-based matcher + Lowe's ratio test
4. Estimate H via RANSAC (`cv2.findHomography`, `RANSAC`, reprojThresh=3.0)
5. Validate: reproject pitch line intersections; accept if mean error < 5px

### Multi-band Blending
Laplacian pyramid blend across 5 levels in the overlap zone. A smooth alpha ramp replaces hard alpha blending. Motion mask is ANDed with the blend mask to suppress ghosting from players crossing the seam.

### Synchronisation
Audio waveform cross-correlation (via `scipy.signal.correlate`) gives sub-frame accuracy. Fallback: visual event alignment on a high-motion frame (kick-off).

---

## Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
ffmpeg-python>=0.2.0
tqdm>=4.66.0
```

---

## Known Limitations

- Assumes fixed cameras (no pan/tilt/zoom during match)
- Homography valid for one venue setup; re-run `calibrate.py` if cameras are remounted
- Audio sync requires both cameras to have recorded audio tracks
- Extreme lens distortion (fisheye) may require a fuller calibration rig

---

## Contact

Built for SportsAlgo × ClayGrounds Hackathon  
Problem domain: Computer Vision / Image Processing  
