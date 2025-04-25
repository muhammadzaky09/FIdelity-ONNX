#pragma once

#include "onnxruntime_cxx_api.h"
#include <vector>
#include <cstdint>
#include <cstdlib>  // for rand()
#ifdef USE_CUDA
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>  // Provides __half.
  #include "bit_flip_cuda.h"  // Declaration is in namespace onnxruntime::contrib.
#endif

// Define OrtMemTypeGPUInput if not defined.
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


    void* bit_pos_data_ptr = nullptr;
    api_.GetTensorMutableData(const_cast<OrtValue*>(bit_pos_tensor), &bit_pos_data_ptr);
    int bit_position = *static_cast<int*>(bit_pos_data_ptr);

    // Ensure input is FP16.
    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      fprintf(stderr, "BitFlip op expects FP16 tensor\n");
      return;
    }

    // Select a random index in the flattened tensor.
    int random_index = static_cast<int>(rand() % element_count);

#ifdef USE_CUDA
    // GPU branch.
    void* device_input = nullptr;
    api_.GetTensorMutableData(const_cast<OrtValue*>(input), &device_input);
    void* device_output = nullptr;
    api_.GetTensorMutableData(output, &device_output);
    cudaStream_t stream = nullptr;
    OrtStatus* stream_status = api_.KernelContext_GetGPUComputeStream(context, reinterpret_cast<void**>(&stream));
    if (stream_status != nullptr || stream == nullptr) {
      fprintf(stderr, "Failed to obtain CUDA compute stream\n");
    }
    cudaError_t err = onnxruntime::contrib::LaunchBitFlipKernelFP16(
        device_input, device_output, bit_position, random_index, element_count, stream);
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA kernel failed with error: %d\n", err);
    }
#else
    // CPU branch.
    void* host_input = nullptr;
    api_.GetTensorMutableData(const_cast<OrtValue*>(input), &host_input);
    void* host_output = nullptr;
    api_.GetTensorMutableData(output, &host_output);
    const unsigned short* in = static_cast<const unsigned short*>(host_input);
    unsigned short* out = static_cast<unsigned short*>(host_output);
    for (size_t i = 0; i < element_count; i++) {
      out[i] = 0;
    }
    unsigned short orig_bits = in[random_index];
    unsigned short flipped_bits = orig_bits ^ (1u << bit_position);
    __half h_orig = *reinterpret_cast<const __half*>(&orig_bits);
    __half h_flipped = *reinterpret_cast<const __half*>(&flipped_bits);
    float f_orig = __half2float(h_orig);
    float f_flipped = __half2float(h_flipped);
    float delta = f_flipped - f_orig;
    __half h_delta = __float2half_rn(delta);
    out[random_index] = *reinterpret_cast<unsigned short*>(&h_delta);
#endif

    api_.ReleaseTensorTypeAndShapeInfo(tensor_info);
  }

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
  // Force input 0 on GPU and input 1 on CPU.
  OrtMemType GetInputMemoryType(size_t index) const {
    return (index == 0) ? OrtMemTypeGPUInput : OrtMemTypeCPUInput;
  }
#else
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

  // Provide a default parameter so GetName() can be called with zero arguments.
  const char* GetName(const OrtKernelInfo* info = nullptr) const { return "Perturb"; }

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new BitFlipKernel(api, info);
  }
  
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* op_kernel) const {
    *reinterpret_cast<void**>(op_kernel) = CreateKernel(api, info);
    return nullptr;
  }
};

} // namespace onnx_bitflip
