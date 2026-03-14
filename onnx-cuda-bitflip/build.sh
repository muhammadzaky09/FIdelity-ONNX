#!/bin/bash
set -e
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=ON
cmake --build . -- -j$(nproc)
cd ..
mkdir -p python
cp build/onnx_bitflip.so python/ || echo "Warning: Could not copy library to python directory"
echo "Build complete! The library is in build/onnx_bitflip.so"
