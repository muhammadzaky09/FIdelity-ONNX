#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Find ONNXRuntime
if [ -z "$ONNXRUNTIME_INCLUDE_DIR" ]; then
    # Try to find in common locations
    if [ -d "/usr/local/include/onnxruntime" ]; then
        export ONNXRUNTIME_INCLUDE_DIR="/usr/local/include/onnxruntime"
        export ONNXRUNTIME_LIB_DIR="/usr/local/lib"
    elif [ -d "/usr/include/onnxruntime" ]; then
        export ONNXRUNTIME_INCLUDE_DIR="/usr/include/onnxruntime"
        export ONNXRUNTIME_LIB_DIR="/usr/lib"
    else
        echo "Could not find ONNXRuntime. Please set ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIB_DIR environment variables."
        exit 1
    fi
    echo "Using ONNX Runtime from: $ONNXRUNTIME_INCLUDE_DIR"
fi

# Configure
cmake .. \
    -DONNXRUNTIME_INCLUDE_DIR=$ONNXRUNTIME_INCLUDE_DIR \
    -DONNXRUNTIME_LIB_DIR=$ONNXRUNTIME_LIB_DIR

# Build
cmake --build . -- -j$(nproc)

cd ..