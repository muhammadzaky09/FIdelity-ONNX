#include "bit_flip_cuda.h"
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {

__global__ void BitFlipKernelFP16(const void* input, void* output, int bit_position, int random_index, size_t count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    const unsigned short* in = reinterpret_cast<const unsigned short*>(input);
    unsigned short* out = reinterpret_cast<unsigned short*>(output);
    if (idx == random_index) {
      unsigned short orig = in[idx];
      unsigned short flipped = orig ^ (1u << bit_position);
      // Convert to float using __half conversions.
      __half h_orig = *reinterpret_cast<const __half*>(&orig);
      __half h_flipped = *reinterpret_cast<const __half*>(&flipped);
      float f_orig = __half2float(h_orig);
      float f_flipped = __half2float(h_flipped);
      float delta = f_flipped - f_orig;
      __half h_delta = __float2half_rn(delta);
      out[idx] = *reinterpret_cast<unsigned short*>(&h_delta);
    } else {
      out[idx] = 0;
    }
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
