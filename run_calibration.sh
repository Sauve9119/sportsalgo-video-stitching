#!/bin/bash
# run_calibration.sh
# ------------------
# Run one-time venue calibration.
# Edit the paths below to match your file locations.

set -e

LEFT_CLIP="data/sample/left_clip.mp4"
RIGHT_CLIP="data/sample/right_clip.mp4"
OUTPUT_CALIB="calibration/venue_01.npz"
DEBUG_DIR="output/debug_calib"

echo "══════════════════════════════════════"
echo "  Running Calibration"
echo "══════════════════════════════════════"
echo "  Left  : $LEFT_CLIP"
echo "  Right : $RIGHT_CLIP"
echo "  Output: $OUTPUT_CALIB"
echo ""

python src/calibrate.py \
    --left      "$LEFT_CLIP" \
    --right     "$RIGHT_CLIP" \
    --out       "$OUTPUT_CALIB" \
    --overlap   0.35 \
    --max-frames 500 \
    --debug-dir "$DEBUG_DIR"

echo ""
echo "Done. Calibration file: $OUTPUT_CALIB"
echo "Check debug visuals at:  $DEBUG_DIR/"
