cmake_minimum_required(VERSION 3.13)
project(onnx_bitflip_operator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "ONNX Runtime include directory")
set(ONNXRUNTIME_LIBRARY "" CACHE FILEPATH "ONNX Runtime library")

if(NOT ONNXRUNTIME_INCLUDE_DIR AND DEFINED ENV{ONNXRUNTIME_INCLUDE_DIR})
  set(ONNXRUNTIME_INCLUDE_DIR "$ENV{ONNXRUNTIME_INCLUDE_DIR}" CACHE PATH "ONNX Runtime include directory" FORCE)
endif()
if(NOT ONNXRUNTIME_LIBRARY AND DEFINED ENV{ONNXRUNTIME_LIBRARY})
  set(ONNXRUNTIME_LIBRARY "$ENV{ONNXRUNTIME_LIBRARY}" CACHE FILEPATH "ONNX Runtime library" FORCE)
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  file(GLOB ORT_ROOT_CANDIDATES "${CMAKE_SOURCE_DIR}/../onnxruntime-dev/onnxruntime-linux-*")
  if(ORT_ROOT_CANDIDATES)
    list(GET ORT_ROOT_CANDIDATES 0 ORT_ROOT)
    if(NOT ONNXRUNTIME_INCLUDE_DIR)
      set(ONNXRUNTIME_INCLUDE_DIR "${ORT_ROOT}/include" CACHE PATH "ONNX Runtime include directory" FORCE)
    endif()
    if(NOT ONNXRUNTIME_LIBRARY)
      file(GLOB ORT_LIBRARY_CANDIDATES "${ORT_ROOT}/lib/libonnxruntime.so*")
      if(ORT_LIBRARY_CANDIDATES)
        list(GET ORT_LIBRARY_CANDIDATES 0 ORT_LIBRARY)
        set(ONNXRUNTIME_LIBRARY "${ORT_LIBRARY}" CACHE FILEPATH "ONNX Runtime library" FORCE)
      endif()
    endif()
  endif()
endif()

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
