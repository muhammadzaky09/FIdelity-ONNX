// bitflip_custom_op.cpp
#include "onnxruntime_cxx_api.h"
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

//////////////////////////////////////////////////////////////////////////////
// CPU Implementation
//////////////////////////////////////////////////////////////////////////////

// CPU helper for bitflip for FP32 and FP16 (FP16 stored as uint16_t)
template <typename T>
void Bitflip_CPU(const T* input, T* output, size_t num_elements, uint32_t mask) {
  // For FP32: reinterpret as uint32_t; for FP16 (as uint16_t) this works similarly.
  for (size_t i = 0; i < num_elements; i++) {
    union {
      T value;
      typename std::conditional<std::is_same<T, float>::value, uint32_t, uint16_t>::type bits;
    } u;
    u.value = input[i];
    u.bits ^= mask;
    output[i] = u.value;
  }
}

//////////////////////////////////////////////////////////////////////////////
// GPU Implementation (CUDA)
//////////////////////////////////////////////////////////////////////////////
#ifdef USE_CUDA

// CUDA kernel for FP32
template <typename T>
__global__ void BitflipKernelGPU(const T* input, T* output, size_t num_elements, uint32_t mask);

template <>
__global__ void BitflipKernelGPU<float>(const float* input, float* output, size_t num_elements, uint32_t mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < num_elements; i += stride) {
    union {
      float f;
      uint32_t i;
    } u;
    u.f = input[i];
    u.i ^= mask;
    output[i] = u.f;
  }
}

// CUDA kernel for FP16 using __half
template <>
__global__ void BitflipKernelGPU<__half>(const __half* input, __half* output, size_t num_elements, uint32_t mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < num_elements; i += stride) {
    // reinterpret __half bits via unsigned short pointer
    unsigned short bits = *(reinterpret_cast<const unsigned short*>(&input[i]));
    bits ^= static_cast<unsigned short>(mask);
    __half h;
    *((unsigned short*)(&h)) = bits;
    output[i] = h;
  }
}

#endif  // USE_CUDA

//////////////////////////////////////////////////////////////////////////////
// Kernel classes for the custom op
//////////////////////////////////////////////////////////////////////////////

// CPU kernel class template.
template <typename T>
struct BitflipKernel_CPU {
  BitflipKernel_CPU(const OrtApi& api, const OrtKernelInfo* info) : api_(api), info_(info) {
    // Read the "bit_index" attribute (default = 0).
    bit_index_ = 0;
    try {
      bit_index_ = static_cast<uint32_t>(Ort::KernelInfo(info).GetAttribute<int64_t>("bit_index"));
    } catch (std::exception&) {
      // Attribute not provided; use default 0.
    }
    mask_ = 1u << bit_index_;
  }

  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    const OrtValue* input_tensor = ctx.GetInput(0);
    auto input_data = input_tensor.GetTensorData<T>();
    auto shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
    size_t num_elements = 1;
    for (auto d : shape)
      num_elements *= d;
    OrtValue* output_tensor = ctx.GetOutput(0, shape);
    T* output_data = output_tensor.GetTensorMutableData<T>();

    // Execute the CPU bitflip loop.
    Bitflip_CPU<T>(input_data, output_data, num_elements, mask_);
  }

 private:
  const OrtApi& api_;
  const OrtKernelInfo* info_;
  uint32_t bit_index_;
  uint32_t mask_;
};

#ifdef USE_CUDA
// GPU kernel class template.
template <typename T>
struct BitflipKernel_CUDA {
  BitflipKernel_CUDA(const OrtApi& api, const OrtKernelInfo* info) : api_(api), info_(info) {
    bit_index_ = 0;
    try {
      bit_index_ = static_cast<uint32_t>(Ort::KernelInfo(info).GetAttribute<int64_t>("bit_index"));
    } catch (std::exception&) {
      // default remains 0.
    }
    mask_ = 1u << bit_index_;
  }

  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    const OrtValue* input_tensor = ctx.GetInput(0);
    auto shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
    size_t num_elements = 1;
    for (auto d : shape)
      num_elements *= d;
    const T* input_data = input_tensor.GetTensorData<T>();
    OrtValue* output_tensor = ctx.GetOutput(0, shape);
    T* output_data = output_tensor.GetTensorMutableData<T>();

    // Choose block and grid sizes.
    int blockSize = 256;
    int gridSize = (num_elements + blockSize - 1) / blockSize;

    // Launch the CUDA kernel. (Error handling omitted for brevity.)
    if constexpr (std::is_same<T, float>::value) {
      BitflipKernelGPU<float><<<gridSize, blockSize>>>(input_data, output_data, num_elements, mask_);
    } else if constexpr (std::is_same<T, __half>::value) {
      BitflipKernelGPU<__half><<<gridSize, blockSize>>>(input_data, output_data, num_elements, mask_);
    }
    cudaDeviceSynchronize();
  }

 private:
  const OrtApi& api_;
  const OrtKernelInfo* info_;
  uint32_t bit_index_;
  uint32_t mask_;
};
#endif  // USE_CUDA

//////////////////////////////////////////////////////////////////////////////
// CustomOp classes (CPU and, conditionally, GPU)
//////////////////////////////////////////////////////////////////////////////

// CPU custom op. We use T = float for FP32 and T = uint16_t for FP16 on the CPU.
template <typename T>
struct CustomOpBitflip_CPU : Ort::CustomOpBase<CustomOpBitflip_CPU<T>, BitflipKernel_CPU<T>> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new BitflipKernel_CPU<T>(api, info);
  }
  const char* GetName() const { return "Bitflip"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    if (std::is_same<T, float>::value)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else if (std::is_same<T, uint16_t>::value)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    if (std::is_same<T, float>::value)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else if (std::is_same<T, uint16_t>::value)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
};

#ifdef USE_CUDA
// GPU custom op. For GPU we use T = float for FP32 and T = __half for FP16.
template <typename T>
struct CustomOpBitflip_CUDA : Ort::CustomOpBase<CustomOpBitflip_CUDA<T>, BitflipKernel_CUDA<T>> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new BitflipKernel_CUDA<T>(api, info);
  }
  const char* GetName() const { return "Bitflip"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    if (std::is_same<T, float>::value)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else if (std::is_same<T, __half>::value)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    if (std::is_same<T, float>::value)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else if (std::is_same<T, __half>::value)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
};
#endif  // USE_CUDA

//////////////////////////////////////////////////////////////////////////////
// Registration function (to be called by ONNX Runtime)
//////////////////////////////////////////////////////////////////////////////

// Use extern "C" and proper export decoration so that ONNX Runtime can load this library.
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {

EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);

  // Create a custom op domain (you can choose any name)
  Ort::CustomOpDomain custom_op_domain("custom_domain");

  // Register CPU kernels:
  static CustomOpBitflip_CPU<float> custom_op_bitflip_cpu_float;
  static CustomOpBitflip_CPU<uint16_t> custom_op_bitflip_cpu_fp16;
  custom_op_domain.Add(&custom_op_bitflip_cpu_float);
  custom_op_domain.Add(&custom_op_bitflip_cpu_fp16);

#ifdef USE_CUDA
  // Register GPU kernels:
  static CustomOpBitflip_CUDA<float> custom_op_bitflip_cuda_float;
  static CustomOpBitflip_CUDA<__half> custom_op_bitflip_cuda_fp16;
  custom_op_domain.Add(&custom_op_bitflip_cuda_float);
  custom_op_domain.Add(&custom_op_bitflip_cuda_fp16);
#endif

  // Add the custom op domain to the session options.
  Ort::ThrowOnError(api->AddCustomOpDomain(options, custom_op_domain));

  return nullptr;  // Return nullptr on success.
}

}  // extern "C"
