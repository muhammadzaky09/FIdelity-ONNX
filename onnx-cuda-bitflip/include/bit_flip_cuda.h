#pragma once
#include <cuda_runtime.h>
namespace onnxruntime {
namespace contrib {
extern "C" {
  // Updated launcher: accepts an extra parameter for random_index.
  cudaError_t LaunchBitFlipKernelFP16(const void* input, void* output, int bit_position, int random_index, size_t count, cudaStream_t stream);
}
}  // namespace contrib
}  // namespace onnxruntime
