"""
Structural test for create_random_bitflip_injection (fp16).

BitFlip (custom.bitflip) takes 3 inputs:
  (fp16_tensor, bit_position:int32, fault_index:int64)
It copies the full tensor and flips one bit at fault_index — bit-exact.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from onnx import TensorProto
from inject_ops import create_random_bitflip_injection

nodes = create_random_bitflip_injection("Y", fp16=True,
                                        rand_idx_name="rand_idx_inject",
                                        bit_pos_name="bit_pos_inject")

print(f"Node count : {len(nodes)}  (expected 1)")
for n in nodes:
    print(f"  {n.op_type:10s} domain={n.domain!r:20s} inputs={list(n.input)} -> {list(n.output)}")

assert len(nodes) == 1
assert nodes[0].op_type == "BitFlip"
assert nodes[0].domain  == "custom.bitflip"
assert nodes[0].input[0] == "Y"
assert nodes[0].input[1] == "bit_pos_inject"
assert nodes[0].input[2] == "rand_idx_inject"
assert nodes[0].output[0] == "Y_faulty"

print("\nPASS")
print("NOTE: runtime test requires CUDA + llama/onnx_bitflip.so (GPU server).")
