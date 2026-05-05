"""
Structural test for create_random_bitflip_injection (fp16).

BitFlip (custom.bitflip) takes 3 inputs:
  (fp16_tensor, bit_position:int32, fault_index:int64)
It copies the full tensor and flips one bit at fault_index — bit-exact.
"""
from inject_ops import create_random_bitflip_injection


def test_fp16_random_bitflip_node_uses_custom_bitflip_domain():
    nodes = create_random_bitflip_injection(
        "Y",
        fp16=True,
        rand_idx_name="rand_idx_inject",
        bit_pos_name="bit_pos_inject",
    )

    assert len(nodes) == 1
    assert nodes[0].op_type == "BitFlip"
    assert nodes[0].domain == "custom.bitflip"
    assert nodes[0].input[0] == "Y"
    assert nodes[0].input[1] == "bit_pos_inject"
    assert nodes[0].input[2] == "rand_idx_inject"
    assert nodes[0].output[0] == "Y_faulty"
