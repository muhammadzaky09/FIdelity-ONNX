import onnx
import json
import os
import glob
import argparse
from collections import deque


def trace_tensor_to_round(graph, tensor_name: str):
    """
    Walk backward from *tensor_name* through producer nodes until a Round node
    is found.  Returns the output tensor name of that Round node, or None if
    no Round node exists on any upstream path.
    """
    producer_map = {}
    for node in graph.node:
        for output in node.output:
            producer_map[output] = node

    visited = set()
    queue = deque([tensor_name])

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        if current in producer_map:
            producer = producer_map[current]
            if producer.op_type == "Round":
                return current
            for inp in producer.input:
                queue.append(inp)

    return None


def resolve_starting_point(graph, tensor_name: str) -> str:
    """
    Return the best injection starting point for *tensor_name*.

    Strategy (auto-detect, no precision flag needed):
      1. Try tracing backward to a Round node (legacy fake-quant models).
      2. If not found, use *tensor_name* directly.

    Standard Q/DQ CNN models intentionally fall through to step 2.  Their Conv
    inputs are DequantizeLinear outputs, and graph.py adapts those to the real
    integer bitflip source: DequantizeLinear.input[0].
    """
    round_output = trace_tensor_to_round(graph, tensor_name)
    return round_output if round_output is not None else tensor_name


def layer_type_for_op(op_type: str) -> str:
    if op_type == "Conv":
        return "Conv"
    if op_type in {"Gemm", "Linear", "FullyConnected"}:
        return "FC"
    return "MatMul"


def parse_target_nodes(model_path: str, ops: list[str]):
    """
    Scan *model_path* for nodes whose op_type is in *ops*.

    For each node:
    - target_layer  = node.name if set, else node.output[0] (output tensor name).
    - input_tensor  = resolved starting point for node.input[0].
    - weight_tensor = resolved starting point for node.input[1].
    - layer_type    = MatMul, Conv, or FC.

    Returns a list of config dicts ready to be passed to modify_onnx_graph.
    """
    model = onnx.load(model_path)
    graph = model.graph

    results = []

    for node in graph.node:
        if node.op_type not in ops:
            continue

        # target_layer: prefer node name; fall back to first output tensor name
        target_layer = node.name if node.name else (node.output[0] if node.output else None)
        if target_layer is None:
            continue

        input0 = node.input[0] if len(node.input) > 0 else None
        input1 = node.input[1] if len(node.input) > 1 else None

        if not input0 or not input1:
            continue

        input_tensor  = resolve_starting_point(graph, input0)
        # Q/DQ weights remain at DequantizeLinear output here; graph.py detects
        # that source node and injects at its integer input. Initializer weights
        # without a producer remain as-is.
        weight_tensor = resolve_starting_point(graph, input1)

        results.append({
            "model_name":     model_path,
            "target_layer":   target_layer,
            "input_tensor":   input_tensor,
            "weight_tensor":  weight_tensor,
            "layer_type":     layer_type_for_op(node.op_type),
        })

        print(f"  [{node.op_type} / {layer_type_for_op(node.op_type)}] {target_layer}")
        print(f"    input_tensor  : {input_tensor}")
        print(f"    weight_tensor : {weight_tensor}")

    return results


parse_matmul_nodes = parse_target_nodes


def save_configs(configs: list[dict], output_dir: str):
    """Write one JSON file per config entry."""
    os.makedirs(output_dir, exist_ok=True)

    for i, cfg in enumerate(configs):
        model_stem = os.path.basename(cfg["model_name"]).replace(".onnx", "")
        safe_layer = cfg["target_layer"].replace("/", "_").replace("\\", "_")
        filename   = f"{model_stem}_{safe_layer}.json"
        path       = os.path.join(output_dir, filename)

        with open(path, "w") as f:
            json.dump(cfg, f, indent=4)

        print(f"  Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Parse ONNX files and generate one JSON injection config per target "
            "MatMul, Conv, or FC-like node.  Automatically detects quantized vs float models "
            "by tracing legacy fake-quant inputs backward to Round nodes. Q/DQ models "
            "remain at DequantizeLinear outputs for graph.py to adapt."
        )
    )
    parser.add_argument(
        "onnx_dir",
        help="Directory containing .onnx files to parse.",
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="injection_configs",
        help="Directory where JSON configs are written (default: injection_configs).",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=["MatMul", "Conv", "Gemm", "Linear", "FullyConnected"],
        help="Op types to target (default: MatMul Conv Gemm Linear FullyConnected).",
    )
    args = parser.parse_args()

    onnx_files = sorted(glob.glob(os.path.join(args.onnx_dir, "*.onnx")))
    if not onnx_files:
        print(f"No .onnx files found in '{args.onnx_dir}'")
        raise SystemExit(1)

    print(f"Found {len(onnx_files)} ONNX file(s) in '{args.onnx_dir}'")
    total = 0
    for model_path in onnx_files:
        print(f"\nProcessing {model_path}")
        configs = parse_target_nodes(model_path, ops=args.ops)
        save_configs(configs, args.output_dir)
        total += len(configs)

    print(f"\nDone. Wrote {total} config file(s) to '{args.output_dir}'")
