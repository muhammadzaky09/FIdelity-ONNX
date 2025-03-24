#include "bit_flip_cuda.h"
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {

__global__ void BitFlipKernelFP16(const void* input, void* output, int bit_position, size_t count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    // Interpret input and output as pointers to 16-bit unsigned integers.
    const unsigned short* in = reinterpret_cast<const unsigned short*>(input);
    unsigned short* out = reinterpret_cast<unsigned short*>(output);
    unsigned short val = in[idx];
    // Flip the specified bit exactly as in the CPU branch.
    val ^= (1u << bit_position);
    out[idx] = val;
  }
}

cudaError_t LaunchBitFlipKernelFP16(const void* input, void* output, int bit_position, size_t count, cudaStream_t stream) {
  constexpr int threads_per_block = 256;
  int blocks = (count + threads_per_block - 1) / threads_per_block;
  BitFlipKernelFP16<<<blocks, threads_per_block, 0, stream>>>(input, output, bit_position, count);
  // Wait for the kernel to finish before returning.
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;
  return cudaGetLastError();
}

} // namespace contrib
} // namespace onnxruntime
