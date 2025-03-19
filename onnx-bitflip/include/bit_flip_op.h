// bitflip_op.h
#pragma once

#include "onnxruntime_cxx_api.h"
#include <vector>

namespace onnxruntime {
namespace contrib {

class BitFlipOp {
 public:
  static void RegisterOps(OrtCustomOpDomain* domain);
};

class BitFlipKernel {
 public:
  BitFlipKernel(const OrtApi& api) : api_(api) {}
  
  Status Compute(OrtKernelContext* context);
  
 private:
  const OrtApi& api_;
  
  template <typename T>
  Status ComputeImpl(OrtKernelContext* context);
  
  template <typename T>
  T FlipBit(T value, int bit_position);
};

}  // namespace contrib
}  // namespace onnxruntime