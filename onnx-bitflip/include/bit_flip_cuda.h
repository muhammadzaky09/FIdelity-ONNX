#pragma once
#include <cuda_runtime.h>
namespace onnxruntime {
namespace contrib {
extern "C" {
  // Declare the CUDA kernel launcher for FP16.
  cudaError_t LaunchBitFlipKernelFP16(const void* input, void* output, int bit_position, size_t count, cudaStream_t stream);
}
} // namespace contrib
} // namespace onnxruntime
