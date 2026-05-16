#!/bin/bash
# run_stitch.sh
# -------------
# Stitch a full match using saved calibration.
# Edit paths below to match your file locations.

set -e

LEFT_MATCH="data/sample/left_match.mp4"
RIGHT_MATCH="data/sample/right_match.mp4"
CALIBRATION="calibration/venue_01.npz"
OUTPUT="output/stitched_match.mp4"

# Set --debug flag to also save debug frames and a side-by-side preview
DEBUG_FLAG=""
# DEBUG_FLAG="--debug --debug-dir output/debug_stitch"

echo "══════════════════════════════════════"
echo "  Running Stitcher"
echo "══════════════════════════════════════"
echo "  Left        : $LEFT_MATCH"
echo "  Right       : $RIGHT_MATCH"
echo "  Calibration : $CALIBRATION"
echo "  Output      : $OUTPUT"
echo ""

python src/stitch.py \
    --left        "$LEFT_MATCH" \
    --right       "$RIGHT_MATCH" \
    --calibration "$CALIBRATION" \
    --out         "$OUTPUT" \
    --sync-method auto \
    --blend-width 120 \
    --levels      5 \
    --motion-mode framediff \
    $DEBUG_FLAG

echo ""
echo "Done. Output: $OUTPUT"
