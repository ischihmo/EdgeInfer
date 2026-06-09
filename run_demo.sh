#!/bin/bash
#
# run_demo.sh — copy runtime files to test/ and execute the demo
#
# Usage:
#   ./run_demo.sh                           # generate a test image and run
#   ./run_demo.sh /path/to/image.ppm        # use your own PPM image
#   ./run_demo.sh /path/to/image.jpg        # auto-convert JPEG → PPM via ffmpeg
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/install"
TEST_DIR="${SCRIPT_DIR}/test"

# ── Clean and recreate test directory ──────────────────────
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"/{bin,lib,model}

# ── Copy runtime files ─────────────────────────────────────
echo "=== Copying runtime files ==="

cp -v "$INSTALL_DIR/bin/demo"           "$TEST_DIR/bin/"
cp -v "$INSTALL_DIR/lib/"lib* "$TEST_DIR/lib/"
cp -v "$INSTALL_DIR/share/EdgeInfer/model/"yolov5n.* "$TEST_DIR/model/"
cp -v "$INSTALL_DIR/share/EdgeInfer/config.yaml" "$TEST_DIR/"

# ── Determine image ────────────────────────────────────────
if [ "${1:-}" != "" ]; then
    IMG_SRC="$1"
    IMG_NAME="input.ppm"

    if [[ "$IMG_SRC" == *.ppm ]] || [[ "$IMG_SRC" == *.PPM ]]; then
        echo "=== Using provided PPM image ==="
        cp -v "$IMG_SRC" "$TEST_DIR/$IMG_NAME"
    else
        echo "=== Converting image to PPM ==="
        ffmpeg -y -loglevel error -i "$IMG_SRC" -vframes 1 \
            -vf "scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2" \
            "$TEST_DIR/$IMG_NAME"
        echo "Converted: $TEST_DIR/$IMG_NAME"
    fi
else
    echo "=== Generating test PPM (640x640 gradient) ==="
    python3 -c "
import struct
w, h = 640, 640
with open('$TEST_DIR/input.ppm', 'wb') as f:
    f.write(f'P6\n{w} {h}\n255\n'.encode())
    for y in range(h):
        row = b''
        for x in range(w):
            r = (x * 255) // (w - 1)
            g = (y * 255) // (h - 1)
            b = 128
            row += struct.pack('BBB', r, g, b)
        f.write(row)
"
    echo "Generated: $TEST_DIR/input.ppm"
fi

# ── Run demo ───────────────────────────────────────────────
echo ""
echo "=== Running demo ==="
cd "$TEST_DIR"

export LD_LIBRARY_PATH="${TEST_DIR}/lib:${LD_LIBRARY_PATH:-}"

./bin/demo input.ppm config.yaml

echo ""
echo "=== Done ==="
echo "Test artifacts in: $TEST_DIR"
