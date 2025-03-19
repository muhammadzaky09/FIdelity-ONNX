#pragma once

#include <cuda_runtime.h>

// Forward declarations
namespace onnxruntime {
namespace contrib {

// External C functions to be called from the BitFlipKernel class
extern "C" {

cudaError_t LaunchBitFlipKernelFP32(const float* input, float* output, int bit_position, size_t count, cudaStream_t stream);
cudaError_t LaunchBitFlipKernelFP16(const void* input, void* output, int bit_position, size_t count, cudaStream_t stream);

}  // extern "C"

}  // namespace contrib
}  // namespace onnxruntime