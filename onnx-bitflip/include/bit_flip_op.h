#pragma once

#include "onnxruntime_cxx_api.h"
#include <vector>
#include <cstdint>
#ifdef USE_CUDA
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>  // Provides __half.
  #include "bit_flip_cuda.h"  // Declaration is in namespace onnxruntime::contrib.
#endif

#ifndef OrtMemTypeGPUInput
#define OrtMemTypeGPUInput 2
#endif

namespace onnx_bitflip {

struct BitFlipKernel {
  BitFlipKernel(const OrtApi& api, const OrtKernelInfo* /*info*/)
      : api_(api) {}

  void Compute(OrtKernelContext* context) {
    // Get input tensor and bit position tensor.
    const OrtValue* input = nullptr;
    api_.KernelContext_GetInput(context, 0, &input);
    const OrtValue* bit_pos_tensor = nullptr;
    api_.KernelContext_GetInput(context, 1, &bit_pos_tensor);

    // Get tensor shape and type.
    OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    api_.GetTensorTypeAndShape(input, &tensor_info);
    size_t dim_count;
    api_.GetDimensionsCount(tensor_info, &dim_count);
    std::vector<int64_t> dims(dim_count);
    api_.GetDimensions(tensor_info, dims.data(), dim_count);
    size_t element_count;
    api_.GetTensorShapeElementCount(tensor_info, &element_count);
    ONNXTensorElementDataType element_type;
    api_.GetTensorElementType(tensor_info, &element_type);

    // Create output tensor.
    OrtValue* output = nullptr;
    api_.KernelContext_GetOutput(context, 0, dims.data(), dim_count, &output);

    // Retrieve bit position from host memory (scalar tensors are usually on host).
    void* bit_pos_data_ptr = nullptr;
    api_.GetTensorMutableData(const_cast<OrtValue*>(bit_pos_tensor), &bit_pos_data_ptr);
    int bit_position = *static_cast<int*>(bit_pos_data_ptr);

    // Ensure the tensor type is FP16.
    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      fprintf(stderr, "BitFlip op expects FP16 tensor\n");
      return;
    }

#ifdef USE_CUDA
    // Get device pointers.
    void* device_input = nullptr;
    api_.GetTensorMutableData(const_cast<OrtValue*>(input), &device_input);
    void* device_output = nullptr;
    api_.GetTensorMutableData(output, &device_output);

    // Query the CUDA compute stream.
    cudaStream_t stream = nullptr;
    OrtStatus* stream_status = api_.KernelContext_GetGPUComputeStream(context, reinterpret_cast<void**>(&stream));
    if (stream_status != nullptr || stream == nullptr) {
      fprintf(stderr, "Failed to obtain CUDA compute stream\n");
    }

    // Launch the CUDA kernel via the launcher in onnxruntime::contrib.
    cudaError_t err = onnxruntime::contrib::LaunchBitFlipKernelFP16(
        device_input, device_output, bit_position, element_count, stream);
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA kernel failed with error: %d\n", err);
    }
#else
    // CPU implementation.
    void* host_input = nullptr;
    api_.GetTensorMutableData(const_cast<OrtValue*>(input), &host_input);
    void* host_output = nullptr;
    api_.GetTensorMutableData(output, &host_output);
    const uint16_t* in_fp16 = static_cast<const uint16_t*>(host_input);
    uint16_t* out_fp16 = static_cast<uint16_t*>(host_output);
    for (size_t i = 0; i < element_count; i++) {
      uint16_t val = in_fp16[i];
      val ^= (1u << bit_position);
      out_fp16[i] = val;
    }
#endif

    api_.ReleaseTensorTypeAndShapeInfo(tensor_info);
  }

  // ComputeV2 is a thin wrapper.
  OrtStatus* ComputeV2(OrtKernelContext* context) {
    Compute(context);
    return nullptr;
  }

private:
  const OrtApi& api_;
};

struct BitFlipOp : Ort::CustomOpBase<BitFlipOp, BitFlipKernel> {
  #ifdef USE_CUDA
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return (index == 0) ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  }
  // IMPORTANT: Override GetInputMemoryType for both inputs.
  OrtMemType GetInputMemoryType(size_t index) const {
    // Force input 0 (the FP16 tensor) to be GPU memory,
    // and input 1 (the scalar) to be CPU memory.
    return (index == 0) ? OrtMemTypeGPUInput : OrtMemTypeCPUInput;
  }
#else
  // CPU branch remains unchanged.
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return (index == 0) ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  }
#endif


  // Provide a default parameter so that GetName() can be called without an argument.
  const char* GetName(const OrtKernelInfo* info = nullptr) const { return "BitFlip"; }

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new BitFlipKernel(api, info);
  }
  
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* op_kernel) const {
    *reinterpret_cast<void**>(op_kernel) = CreateKernel(api, info);
    return nullptr;
  }
};

} // namespace onnx_bitflip
