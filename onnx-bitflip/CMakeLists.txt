cmake_minimum_required(VERSION 3.13)
project(onnx_bitflip_operator LANGUAGES CXX CUDA)

# Set C++ standard.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set ONNX Runtime paths based on your pip installation.
# You can verify the include directory by checking:
#   python -c "import onnxruntime; print(onnxruntime.__file__)"
# Then use the "capi" subfolder for the headers.
set(ONNXRUNTIME_INCLUDE_DIR "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi" CACHE PATH "ONNX Runtime include directory from pip")
# Set ONNXRUNTIME_LIBRARY to the main shared library used by onnxruntime (adjust the file name if necessary).
set(ONNXRUNTIME_LIBRARY "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-312-x86_64-linux-gnu.so" CACHE FILEPATH "ONNX Runtime shared library from pip")

if(NOT ONNXRUNTIME_INCLUDE_DIR)
  message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIR must be set")
endif()
if(NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNXRUNTIME_LIBRARY must be set")
endif()

message(STATUS "Using ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Using ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# Add include directories: add your project "include" folder and the ONNX Runtime includes.
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  # Set CUDA architectures (adjust according to your GPU; for example, 75 for Turing)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# List source files.
set(SOURCES src/bit_flip_op.cc src/register_custom_ops.cc)
if(USE_CUDA)
  list(APPEND SOURCES src/bit_flip_cuda.cu)
endif()

# Create shared library.
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

# Embed an rpath so that at runtime the library can locate its dependencies.
# $ORIGIN refers to the directory containing this shared library.
set_target_properties(onnx_bitflip PROPERTIES
  PREFIX ""
  POSITION_INDEPENDENT_CODE ON
  INSTALL_RPATH "$ORIGIN;/usr/local/lib/python3.12/dist-packages/onnxruntime/capi"
)
