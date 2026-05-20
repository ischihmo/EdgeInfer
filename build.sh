#!/bin/bash
#
# EdgeInfer build script
#
# Configurable via environment variables:
#   CC, CXX         - C/C++ compiler (default: gcc, g++)
#   BUILD_TYPE      - Debug | Release (default: Release)
#   INSTALL_PREFIX   - install directory (default: ./install)
#   JOBS            - parallel build jobs (default: nproc)
#
# Usage:
#   ./build.sh                                           # native gcc build
#   CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++   # cross-compile
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Toolchain ──────────────────────────────────────────────
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

echo "=== EdgeInfer Build ==="
echo "CC  = $CC ($(command -v "$CC"))"
echo "CXX = $CXX ($(command -v "$CXX"))"

# ── Build switches ─────────────────────────────────────────
BUILD_DEMO="${EDGEINFER_BUILD_DEMO:-ON}"
ENABLE_NCNN="${EDGEINFER_ENABLE_NCNN:-ON}"
ENABLE_MNN="${EDGEINFER_ENABLE_MNN:-OFF}"
ENABLE_OPENCV="${EDGEINFER_ENABLE_OPENCV:-ON}"

# ── Paths ──────────────────────────────────────────────────
BUILD_TYPE="${BUILD_TYPE:-Release}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${SCRIPT_DIR}/install}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "BUILD_DEMO     = ${BUILD_DEMO}"
echo "ENABLE_NCNN    = ${ENABLE_NCNN}"
echo "ENABLE_MNN     = ${ENABLE_MNN}"
echo "ENABLE_OPENCV  = ${ENABLE_OPENCV}"
echo "BUILD_TYPE     = ${BUILD_TYPE}"
echo "INSTALL_PREFIX = ${INSTALL_PREFIX}"
echo "JOBS           = ${JOBS}"
echo ""

# ── Configure ──────────────────────────────────────────────
CMAKE_ARGS=(
    -DCMAKE_C_COMPILER="$CC"
    -DCMAKE_CXX_COMPILER="$CXX"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    -DEDGEINFER_BUILD_DEMO="$BUILD_DEMO"
    -DEDGEINFER_WITH_NCNN="$ENABLE_NCNN"
    -DEDGEINFER_WITH_MNN="$ENABLE_MNN"
    -DEDGEINFER_WITH_OPENCV="$ENABLE_OPENCV"
)

mkdir -p "$BUILD_DIR"

cmake -B "$BUILD_DIR" -S . "${CMAKE_ARGS[@]}"

# ── Build ──────────────────────────────────────────────────
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j "$JOBS"

# ── Install ────────────────────────────────────────────────
rm -rf "$INSTALL_PREFIX"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_PREFIX"

# ── Copy model for demo ────────────────────────────────────
if [ -d "${SCRIPT_DIR}/demo/model" ]; then
    mkdir -p "${INSTALL_PREFIX}/share/EdgeInfer"
    cp -r "${SCRIPT_DIR}/demo/model" "${INSTALL_PREFIX}/share/EdgeInfer/"
fi

echo ""
echo "=== Build complete ==="
echo "Install tree:"
find "$INSTALL_PREFIX" -type f | sort | sed 's/^/  /'
echo ""
echo "Run demo:"
echo "  ./run_demo.sh /path/to/image.ppm"
echo "  ./run_demo.sh /path/to/image.jpg"
