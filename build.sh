#!/bin/bash
#
# EdgeInfer build script
#
# Configurable via environment variables:
#   TARGET_ARCH        - x86 | arm-ca53 (auto-detected from CC if unset)
#   TOOLCHAIN_DIR      - ARM cross-compiler bin dir (default: ~/cross-compilier/arm-ca53-linux-gnueabihf-8.4/bin)
#   CC, CXX            - C/C++ compiler (auto-set from TOOLCHAIN_DIR when TARGET_ARCH=arm-ca53)
#   BUILD_TYPE         - Debug | Release (default: Release)
#   INSTALL_PREFIX     - install directory (default: ./install)
#   JOBS               - parallel build jobs (default: nproc)
#
# Usage:
#   EDGEINFER_ENABLE_NCNN=ON EDGEINFER_ENABLE_OPENCV=ON ./build.sh    # x86 native
#   TARGET_ARCH=arm-ca53 EDGEINFER_ENABLE_NCNN=ON ./build.sh           # ARM ncnn
#   EDGEINFER_ENABLE_NTCNN=ON ./build.sh                              # ARM NTCNN (auto arm-ca53)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Architecture ────────────────────────────────────────────
TARGET_ARCH="${TARGET_ARCH:-}"                          # user override
NTCNN_ON="${EDGEINFER_ENABLE_NTCNN:-OFF}"

# ARM toolchain path
TOOLCHAIN_DIR="${TOOLCHAIN_DIR:-${HOME}/cross-compilier/arm-ca53-linux-gnueabihf-8.4/bin}"
ARM_CC="${TOOLCHAIN_DIR}/arm-ca53-linux-gnueabihf-gcc"
ARM_CXX="${TOOLCHAIN_DIR}/arm-ca53-linux-gnueabihf-g++"

# Resolve TARGET_ARCH: explicit > NTCNN auto > compiler detection
if [ -z "$TARGET_ARCH" ]; then
    if [ "$NTCNN_ON" = "ON" ]; then
        TARGET_ARCH="arm-ca53"
    elif [ -n "${CC:-}" ] && echo "$CC" | grep -q "arm-ca53"; then
        TARGET_ARCH="arm-ca53"
    else
        TARGET_ARCH="x86"
    fi
fi

# ── Compiler ────────────────────────────────────────────────
if [ "$TARGET_ARCH" = "arm-ca53" ]; then
    CC="${CC:-${ARM_CC}}"
    CXX="${CXX:-${ARM_CXX}}"
    export PATH="${TOOLCHAIN_DIR}:${PATH}"
fi
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

echo "=== EdgeInfer Build ==="
echo "CC  = $CC ($(command -v "$CC"))"
echo "CXX = $CXX ($(command -v "$CXX"))"
echo "ARCH = ${TARGET_ARCH}"

# ── Build switches ─────────────────────────────────────────
BUILD_DEMO="${EDGEINFER_BUILD_DEMO:-ON}"
ENABLE_NCNN="${EDGEINFER_ENABLE_NCNN:-OFF}"
ENABLE_MNN="${EDGEINFER_ENABLE_MNN:-OFF}"
ENABLE_OPENCV="${EDGEINFER_ENABLE_OPENCV:-OFF}"
ENABLE_NTCNN="${EDGEINFER_ENABLE_NTCNN:-OFF}"

# ── Paths ──────────────────────────────────────────────────
BUILD_TYPE="${BUILD_TYPE:-Release}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${SCRIPT_DIR}/install}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "BUILD_DEMO     = ${BUILD_DEMO}"
echo "ENABLE_NCNN    = ${ENABLE_NCNN}"
echo "ENABLE_MNN     = ${ENABLE_MNN}"
echo "ENABLE_OPENCV  = ${ENABLE_OPENCV}"
echo "ENABLE_NTCNN   = ${ENABLE_NTCNN}"
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
    -DTHIRD_PARTY_ARCH="$TARGET_ARCH"
    -DEDGEINFER_BUILD_DEMO="$BUILD_DEMO"
    -DEDGEINFER_WITH_NCNN="$ENABLE_NCNN"
    -DEDGEINFER_WITH_MNN="$ENABLE_MNN"
    -DEDGEINFER_WITH_OPENCV="$ENABLE_OPENCV"
    -DEDGEINFER_WITH_NTCNN="$ENABLE_NTCNN"
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
