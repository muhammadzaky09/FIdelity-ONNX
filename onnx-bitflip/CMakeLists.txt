cmake_minimum_required(VERSION 3.13)
project(onnx_bitflip_operator LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70")

# Find ONNX Runtime
set(ONNXRUNTIME_INCLUDE_DIR "/usr/local/include/onnxruntime" CACHE PATH "Path to ONNX Runtime include directory")
set(ONNXRUNTIME_LIB_DIR "/usr/local/lib" CACHE PATH "Path to ONNX Runtime library directory")

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)
link_directories(${ONNXRUNTIME_LIB_DIR})

# Add source files
set(SOURCES
    src/bit_flip_op.cc
    src/bit_flip_cuda.cu
)

# Create shared library
add_library(onnx_bitflip SHARED ${SOURCES})

# Link against ONNX Runtime
target_link_libraries(onnx_bitflip onnxruntime ${CUDA_LIBRARIES})

# Set library properties
set_target_properties(onnx_bitflip PROPERTIES
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
)

# Install
install(TARGETS onnx_bitflip 
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/
    DESTINATION include/onnx_bitflip
    FILES_MATCHING PATTERN "*.h"
)