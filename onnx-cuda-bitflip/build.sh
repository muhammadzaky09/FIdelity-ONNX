#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"

if [ -z "${ONNXRUNTIME_INCLUDE_DIR:-}" ]; then
  ORT_INCLUDE_CANDIDATES=("$REPO_ROOT"/onnxruntime-dev/onnxruntime-linux-*/include)
  if [ -d "${ORT_INCLUDE_CANDIDATES[0]}" ]; then
    ONNXRUNTIME_INCLUDE_DIR="${ORT_INCLUDE_CANDIDATES[0]}"
  fi
fi

if [ -z "${ONNXRUNTIME_LIBRARY:-}" ]; then
  ORT_LIBRARY_CANDIDATES=("$REPO_ROOT"/onnxruntime-dev/onnxruntime-linux-*/lib/libonnxruntime.so*)
  if [ -f "${ORT_LIBRARY_CANDIDATES[0]}" ]; then
    ONNXRUNTIME_LIBRARY="${ORT_LIBRARY_CANDIDATES[0]}"
  fi
fi

if [ -z "${ONNXRUNTIME_INCLUDE_DIR:-}" ] || [ -z "${ONNXRUNTIME_LIBRARY:-}" ]; then
  echo "ONNX Runtime headers/library not found."
  echo "Set ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY, or place ONNX Runtime under ../onnxruntime-dev/."
  exit 1
fi

mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUSE_CUDA=ON \
  -DONNXRUNTIME_INCLUDE_DIR="$ONNXRUNTIME_INCLUDE_DIR" \
  -DONNXRUNTIME_LIBRARY="$ONNXRUNTIME_LIBRARY"
cmake --build . -- -j$(nproc)
cd ..
mkdir -p python
cp build/onnx_bitflip.so python/ || echo "Warning: Could not copy library to python directory"
echo "Build complete! The library is in build/onnx_bitflip.so"
