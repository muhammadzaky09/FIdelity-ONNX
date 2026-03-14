#include "bit_flip_cuda.h"
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {

__global__ void BitFlipKernelFP16(const void* input, void* output, int bit_position, int fault_index, size_t count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    const unsigned short* in  = reinterpret_cast<const unsigned short*>(input);
    unsigned short*       out = reinterpret_cast<unsigned short*>(output);
    // Copy every element; flip exactly one bit at fault_index (supplied by caller).
    out[idx] = (idx == fault_index) ? (in[idx] ^ (1u << bit_position)) : in[idx];
  }
}

cudaError_t LaunchBitFlipKernelFP16(const void* input, void* output, int bit_position, int random_index, size_t count, cudaStream_t stream) {
  constexpr int threads_per_block = 256;
  int blocks = (count + threads_per_block - 1) / threads_per_block;
  BitFlipKernelFP16<<<blocks, threads_per_block, 0, stream>>>(input, output, bit_position, random_index, count);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;
  return cudaGetLastError();
}

} // namespace contrib
} // namespace onnxruntime
