#pragma once

#include "onnxruntime_cxx_api.h"
#include <vector>
#include <cstdint>
#ifdef USE_CUDA
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  #include "bit_flip_cuda.h"
#endif

#ifndef OrtMemTypeGPUInput
#define OrtMemTypeGPUInput 2
#endif

namespace onnx_bitflip {

struct BitFlipKernel {
  BitFlipKernel(const OrtApi& api, const OrtKernelInfo* /*info*/)
      : api_(api) {}

  void Compute(OrtKernelContext* context) {
    // Inputs: 0=tensor(fp16), 1=bit_position(int32), 2=fault_index(int64)
    const OrtValue* input          = nullptr;
    const OrtValue* bit_pos_tensor = nullptr;
    const OrtValue* idx_tensor     = nullptr;
    api_.KernelContext_GetInput(context, 0, &input);
    api_.KernelContext_GetInput(context, 1, &bit_pos_tensor);
    api_.KernelContext_GetInput(context, 2, &idx_tensor);

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

    OrtValue* output = nullptr;
    api_.KernelContext_GetOutput(context, 0, dims.data(), dim_count, &output);

    void* bit_pos_data_ptr = nullptr;
    api_.GetTensorMutableData(const_cast<OrtValue*>(bit_pos_tensor), &bit_pos_data_ptr);
    int bit_position = *static_cast<int*>(bit_pos_data_ptr);

    void* idx_data_ptr = nullptr;
    api_.GetTensorMutableData(const_cast<OrtValue*>(idx_tensor), &idx_data_ptr);
    int fault_index = static_cast<int>(*static_cast<int64_t*>(idx_data_ptr) % static_cast<int64_t>(element_count));

    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      fprintf(stderr, "BitFlip op expects FP16 tensor\n");
      return;
    }

#ifdef USE_CUDA
    void* device_input  = nullptr;
    void* device_output = nullptr;
    api_.GetTensorMutableData(const_cast<OrtValue*>(input), &device_input);
    api_.GetTensorMutableData(output, &device_output);
    cudaStream_t stream = nullptr;
    OrtStatus* stream_status = api_.KernelContext_GetGPUComputeStream(context, reinterpret_cast<void**>(&stream));
    if (stream_status != nullptr || stream == nullptr) {
      fprintf(stderr, "Failed to obtain CUDA compute stream\n");
    }
    cudaError_t err = onnxruntime::contrib::LaunchBitFlipKernelFP16(
        device_input, device_output, bit_position, fault_index, element_count, stream);
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA kernel failed with error: %d\n", err);
    }
#else
    void* host_input  = nullptr;
    void* host_output = nullptr;
    api_.GetTensorMutableData(const_cast<OrtValue*>(input), &host_input);
    api_.GetTensorMutableData(output, &host_output);
    const unsigned short* in  = static_cast<const unsigned short*>(host_input);
    unsigned short*       out = static_cast<unsigned short*>(host_output);
    memcpy(host_output, host_input, element_count * sizeof(unsigned short));
    out[fault_index] = in[fault_index] ^ (1u << bit_position);
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
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if (index == 1) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;  // fault_index
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  }
  // tensor on GPU, bit_position and fault_index on CPU
  OrtMemType GetInputMemoryType(size_t index) const {
    return (index == 0) ? (OrtMemType)OrtMemTypeGPUInput : OrtMemTypeCPUInput;
  }
#else
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if (index == 1) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;  // fault_index
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  }
#endif

  const char* GetName(const OrtKernelInfo* /*info*/ = nullptr) const { return "BitFlip"; }

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new BitFlipKernel(api, info);
  }

  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* op_kernel) const {
    *reinterpret_cast<void**>(op_kernel) = CreateKernel(api, info);
    return nullptr;
  }
};

} // namespace onnx_bitflip
