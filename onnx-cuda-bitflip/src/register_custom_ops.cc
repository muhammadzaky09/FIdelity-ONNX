#include "../include/bit_flip_op.h"
#include <cstdio>

extern "C" {

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  fprintf(stderr, "RegisterCustomOps called\n");

  static Ort::CustomOpDomain custom_op_domain("custom.bitflip");
  static onnx_bitflip::BitFlipOp bitflip_op;
  custom_op_domain.Add(&bitflip_op);

  const OrtApi* ort_api = api_base->GetApi(ORT_API_VERSION);
  OrtStatus* status = ort_api->AddCustomOpDomain(options, custom_op_domain);
  if (status == nullptr) {
    fprintf(stderr, "Custom op domain registered successfully\n");
  }
  return status;
}

} // extern "C"
