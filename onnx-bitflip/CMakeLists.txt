cmake_minimum_required(VERSION 3.13)
project(onnx_bitflip_operator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_INCLUDE_DIR "/home/muhammadzaky/Repository/onnx-transformer/onnxruntime-dev/onnxruntime-linux-x64-gpu-1.20.1/include" CACHE PATH "ONNX Runtime include directory")
set(ONNXRUNTIME_LIBRARY "/home/muhammadzaky/Repository/onnx-transformer/onnxruntime-dev/onnxruntime-linux-x64-gpu-1.20.1/lib/libonnxruntime.so.1" CACHE FILEPATH "ONNX Runtime library")

if(NOT ONNXRUNTIME_INCLUDE_DIR)
  message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIR must be set")
endif()
if(NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNXRUNTIME_LIBRARY must be set")
endif()

message(STATUS "Using ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Using ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(SOURCES src/bit_flip_op.cc src/register_custom_ops.cc)
if(USE_CUDA)
  list(APPEND SOURCES src/bit_flip_cuda.cu)
endif()

add_library(onnx_bitflip SHARED ${SOURCES})

if(USE_CUDA)
  target_compile_definitions(onnx_bitflip PRIVATE USE_CUDA)
endif()

target_compile_options(onnx_bitflip PRIVATE -Wall -Wextra -fpermissive -fPIC)

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  target_link_libraries(onnx_bitflip ${ONNXRUNTIME_LIBRARY} ${CUDA_LIBRARIES})
else()
  target_link_libraries(onnx_bitflip ${ONNXRUNTIME_LIBRARY})
endif()

set_target_properties(onnx_bitflip PROPERTIES PREFIX "" POSITION_INDEPENDENT_CODE ON)
