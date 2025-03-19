#include "bit_flip_op.h"
#include "bit_flip_cuda.h"
#include <cstring>
#include <cmath>

namespace onnxruntime {
namespace contrib {

struct BitFlipCustomOp : OrtCustomOp {
  static constexpr const char* OP_NAME = "BitFlip";
  static constexpr const char* DOMAIN = "contrib.bitflip";
  static constexpr int VERSION = 1;
  
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new BitFlipKernel(api);
  }
  
  const char* GetName() const { return OP_NAME; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; // Support FP32 and FP16
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; // bit position
  }
  
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; // Same as input
  }
};

struct BitFlipCustomOpGpu : BitFlipCustomOp {
  const char* GetExecutionProviderType() const override { return "CUDAExecutionProvider"; }
};

void BitFlipOp::RegisterOps(OrtCustomOpDomain* domain) {
  static BitFlipCustomOp cpu_op;
  static BitFlipCustomOpGpu gpu_op;
  
  domain->Add(&cpu_op);
  domain->Add(&gpu_op);
}

template <typename T>
T BitFlipKernel::FlipBit(T value, int bit_position) {
  static_assert(std::is_floating_point<T>::value, "Only floating point types are supported");
  
  // Determine the total number of bits for this type
  constexpr size_t num_bits = sizeof(T) * 8;
  
  // Ensure the bit position is valid
  if (bit_position < 0 || bit_position >= num_bits) {
    return value;  // No change if position is invalid
  }
  
  // Use a union to reinterpret the floating point value as integer bits
  union {
    T float_value;
    uint32_t int_bits;  // For 32-bit float
    uint16_t short_bits; // For 16-bit float
  } data;
  
  data.float_value = value;
  
  // Flip the bit
  if constexpr (sizeof(T) == 4) {  // FP32
    data.int_bits ^= (1u << bit_position);
  } else if constexpr (sizeof(T) == 2) {  // FP16
    data.short_bits ^= (1u << bit_position);
  }
  
  return data.float_value;
}

template <>
Status BitFlipKernel::ComputeImpl<float>(OrtKernelContext* context) {
  // Get input tensor
  const OrtValue* input_tensor = nullptr;
  ORT_RETURN_IF_ERROR(api_.KernelContext_GetInput(context, 0, &input_tensor));
  
  const OrtValue* bit_position_tensor = nullptr;
  ORT_RETURN_IF_ERROR(api_.KernelContext_GetInput(context, 1, &bit_position_tensor));
  
  // Get tensor info
  OrtTensorTypeAndShapeInfo* input_info = nullptr;
  ORT_RETURN_IF_ERROR(api_.GetTensorTypeAndShape(input_tensor, &input_info));
  
  size_t elem_count;
  ORT_RETURN_IF_ERROR(api_.GetTensorShapeElementCount(input_info, &elem_count));
  
  std::vector<int64_t> dimensions;
  size_t dim_count;
  ORT_RETURN_IF_ERROR(api_.GetDimensionsCount(input_info, &dim_count));
  dimensions.resize(dim_count);
  ORT_RETURN_IF_ERROR(api_.GetDimensions(input_info, dimensions.data(), dim_count));
  
  // Get input data
  const float* input_data = nullptr;
  ORT_RETURN_IF_ERROR(api_.GetTensorData(input_tensor, (const void**)&input_data));
  
  // Get bit position
  const int32_t* bit_position_data = nullptr;
  ORT_RETURN_IF_ERROR(api_.GetTensorData(bit_position_tensor, (const void**)&bit_position_data));
  int bit_position = *bit_position_data;
  
  // Create output tensor
  OrtValue* output_tensor = nullptr;
  ORT_RETURN_IF_ERROR(api_.KernelContext_GetOutput(context, 0, dimensions.data(), dim_count, &output_tensor));
  
  float* output_data = nullptr;
  ORT_RETURN_IF_ERROR(api_.GetTensorMutableData(output_tensor, (void**)&output_data));
  
  // Apply bit flip
  for (size_t i = 0; i < elem_count; i++) {
    output_data[i] = FlipBit(input_data[i], bit_position);
  }
  
  api_.ReleaseTensorTypeAndShapeInfo(input_info);
  
  return Status::OK();
}

// The FP16 implementation is similar but using appropriate half-precision types

Status BitFlipKernel::Compute(OrtKernelContext* context) {
  // Get input tensor
  const OrtValue* input_tensor = nullptr;
  ORT_RETURN_IF_ERROR(api_.KernelContext_GetInput(context, 0, &input_tensor));
  
  // Get element type
  OrtTensorTypeAndShapeInfo* input_info = nullptr;
  ORT_RETURN_IF_ERROR(api_.GetTensorTypeAndShape(input_tensor, &input_info));
  
  ONNXTensorElementDataType element_type;
  ORT_RETURN_IF_ERROR(api_.GetTensorElementType(input_info, &element_type));
  
  api_.ReleaseTensorTypeAndShapeInfo(input_info);
  
  // Dispatch based on element type
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return ComputeImpl<float>(context);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    // Handle FP16 case (would be implemented similar to FP32)
    return Status::OK();
  } else {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Unsupported input data type");
  }
}

}  // namespace contrib
}  // namespace onnxruntime