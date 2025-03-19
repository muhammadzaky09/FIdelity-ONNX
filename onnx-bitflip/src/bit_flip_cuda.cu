#include "bit_flip_cuda.h"
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {

// CUDA kernel for FP32
__global__ void BitFlipKernelFP32(const float* input, float* output, int bit_position, size_t count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    // Using the same bit manipulation as in CPU version
    union {
      float float_value;
      unsigned int int_bits;
    } data;
    
    data.float_value = input[idx];
    data.int_bits ^= (1u << bit_position);
    output[idx] = data.float_value;
  }
}

// CUDA kernel for FP16
__global__ void BitFlipKernelFP16(const half* input, half* output, int bit_position, size_t count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    // Using the same bit manipulation as in CPU version
    union {
      half float_value;
      unsigned short short_bits;
    } data;
    
    data.float_value = input[idx];
    data.short_bits ^= (1u << bit_position);
    output[idx] = data.float_value;
  }
}

// External C functions to call from the BitFlipKernel class
extern "C" {

cudaError_t LaunchBitFlipKernelFP32(const float* input, float* output, int bit_position, size_t count, cudaStream_t stream) {
  constexpr int threads_per_block = 256;
  const int blocks = (count + threads_per_block - 1) / threads_per_block;
  
  BitFlipKernelFP32<<<blocks, threads_per_block, 0, stream>>>(input, output, bit_position, count);
  return cudaGetLastError();
}

cudaError_t LaunchBitFlipKernelFP16(const void* input, void* output, int bit_position, size_t count, cudaStream_t stream) {
  constexpr int threads_per_block = 256;
  const int blocks = (count + threads_per_block - 1) / threads_per_block;
  
  BitFlipKernelFP16<<<blocks, threads_per_block, 0, stream>>>((const half*)input, (half*)output, bit_position, count);
  return cudaGetLastError();
}

}  // extern "C"

}  // namespace contrib
}  // namespace onnxruntime