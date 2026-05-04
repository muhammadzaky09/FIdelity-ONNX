#!/usr/bin/env python3
import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from loguru import logger
from torchvision import datasets, transforms

from graph import modify_onnx_graph


FAULT_MODELS = ["INPUT", "WEIGHT", "INPUT16", "WEIGHT16", "RANDOM", "RANDOM_BITFLIP"]


def load_image(dataset_name, sample_idx):
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "imagenet":
        return load_imagenet_image(sample_idx)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    image, label = dataset[sample_idx]
    return image.unsqueeze(0).numpy().astype(np.float32), int(label)


def load_imagenet_image(sample_idx):
    from datasets import Dataset

    if sample_idx < 0:
        raise ValueError(f"sample_idx must be non-negative, got {sample_idx}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    remaining = sample_idx
    shard_paths = sorted(Path("data").glob("**/imagenet-1k-validation-*.arrow"))
    if not shard_paths:
        raise FileNotFoundError("No local ImageNet validation Arrow shards found under data/")

    for shard_path in shard_paths:
        dataset = Dataset.from_file(str(shard_path))
        shard_rows = len(dataset)
        if remaining >= shard_rows:
            remaining -= shard_rows
            continue

        row = dataset[int(remaining)]
        image = transform(row["image"].convert("RGB"))
        return image.unsqueeze(0).numpy().astype(np.float32), int(row["label"])

    raise IndexError(f"ImageNet validation sample_idx {sample_idx} is out of range")


def make_session(model_path, provider, needs_bitflip=False, needs_extensions=False):
    opts = ort.SessionOptions()
    if needs_bitflip:
        opts.register_custom_ops_library(os.path.abspath("llama/onnx_bitflip.so"))
    if needs_extensions:
        from onnxruntime_extensions import get_library_path
        opts.register_custom_ops_library(get_library_path())
    return ort.InferenceSession(model_path, sess_options=opts, providers=[provider])


def primary_input_name(session):
    inputs = [
        inp for inp in session.get_inputs()
        if inp.name not in {"rand_idx_inject", "bit_pos_inject"}
    ]
    rank4_inputs = [inp for inp in inputs if len(inp.shape) == 4]
    return (rank4_inputs[0] if rank4_inputs else inputs[0]).name


def bit_range_for_precision(precision):
    if precision == "float16":
        return range(16)
    if precision == "int8":
        return range(8)
    if precision == "int4":
        return range(4)
    if precision == "float32":
        return range(32)
    raise ValueError(f"Unsupported precision: {precision}")


def tensor_shape_from_value_info(value_info):
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value <= 0:
            return None
        shape.append(dim.dim_value)
    return shape


def tensor_shape(model_path, tensor_name):
    model = onnx.shape_inference.infer_shapes(onnx.load(model_path))
    for initializer in model.graph.initializer:
        if initializer.name == tensor_name:
            return list(initializer.dims)
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if value_info.name == tensor_name:
            return tensor_shape_from_value_info(value_info)
    return None


def tensor_numel(model_path, tensor_name):
    shape = tensor_shape(model_path, tensor_name)
    if shape is None:
        raise ValueError(f"Could not infer shape for tensor '{tensor_name}' in {model_path}")
    return int(np.prod(shape))


def target_output_tensor(model_path, target_layer):
    model = onnx.load(model_path)
    for node in model.graph.node:
        if node.name == target_layer or target_layer in node.output:
            return node.output[0]
    return target_layer


def target_tensor_for_fault(config, fault_model):
    if fault_model in {"INPUT", "INPUT16"}:
        return config["input_tensor"]
    if fault_model in {"WEIGHT", "WEIGHT16"}:
        return config["weight_tensor"]
    return target_output_tensor(config["model_name"], config["target_layer"])


def prediction(output):
    return int(np.argmax(output.reshape(-1, output.shape[-1]), axis=1)[0])


def read_layer_configs(config_dir):
    paths = sorted(Path(config_dir).glob("*.json"))
    if not paths:
        raise ValueError(f"No *.json layer configs found in {config_dir}")
    configs = []
    for path in paths:
        with open(path) as f:
            configs.append((path.name, json.load(f)))
    return configs


def main():
    parser = argparse.ArgumentParser(description="CNN single-image fault-injection inference")
    parser.add_argument("--config_dir", required=True,
                        help="Directory containing layer JSON configs from parser.py")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "imagenet"], required=True)
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Test-set image index to run (default: 0)")
    parser.add_argument("--precision", default="int8",
                        choices=["int8", "int4", "float16", "float32"])
    parser.add_argument("--fault_models", nargs="+", default=["INPUT", "WEIGHT", "INPUT16", "WEIGHT16"],
                        choices=FAULT_MODELS)
    parser.add_argument("--bit_position", type=int, default=None,
                        help="If omitted, iterate over all bit positions for --precision")
    parser.add_argument("--provider", default="CPUExecutionProvider",
                        choices=["CPUExecutionProvider", "CUDAExecutionProvider"])
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed mixed into deterministic rand_idx selection")
    parser.add_argument("--output_csv", default=None,
                        help="CSV output path. Defaults to cnn_results_<dataset>_<precision>.csv")
    args = parser.parse_args()

    image, label = load_image(args.dataset, args.sample_idx)
    bit_positions = [args.bit_position] if args.bit_position is not None else list(bit_range_for_precision(args.precision))
    output_csv = args.output_csv or f"cnn_results_{args.dataset}_{args.precision}.csv"

    layer_configs = read_layer_configs(args.config_dir)
    first_model = layer_configs[0][1]["model_name"]
    golden_session = make_session(first_model, args.provider)
    input_name = primary_input_name(golden_session)

    fieldnames = [
        "Timestamp", "Layer_Config", "Layer_Type", "Fault_Model", "Bit_Position",
        "Sample_Idx", "Rand_Idx", "Golden_Pred", "Faulty_Pred", "Label",
        "Prediction_Changed", "Golden_Correct", "Faulty_Correct", "Linf_Diff",
    ]
    file_exists = os.path.exists(output_csv)

    with open(output_csv, "a" if file_exists else "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for layer_file, config in layer_configs:
            if config["model_name"] != first_model:
                golden_session = make_session(config["model_name"], args.provider)
                input_name = primary_input_name(golden_session)

            golden_out = golden_session.run(None, {input_name: image})[0]
            golden_pred = prediction(golden_out)

            logger.info(f"Layer config: {layer_file}")
            logger.info(f"Golden prediction: {golden_pred}, label: {label}")

            for fault_model in args.fault_models:
                faulty_path = modify_onnx_graph(config, {"precision": args.precision}, fault_model)
                needs_bitflip = (
                    ("BITFLIP" in fault_model and args.precision == "float16")
                    or (args.precision == "float16" and "RANDOM" not in fault_model)
                )
                needs_extensions = "BITFLIP" in fault_model and args.precision == "float32"
                faulty_session = make_session(
                    faulty_path,
                    args.provider,
                    needs_bitflip=needs_bitflip,
                    needs_extensions=needs_extensions,
                )

                target_tensor = target_tensor_for_fault(config, fault_model)
                rand_bound = tensor_numel(config["model_name"], target_tensor)

                for bit_position in bit_positions:
                    inject_seed = hash((args.seed, layer_file, fault_model, bit_position, args.sample_idx)) & 0xFFFFFFFF
                    rng = np.random.default_rng(inject_seed)
                    rand_idx = int(rng.integers(0, rand_bound))

                    feed = {
                        input_name: image,
                        "rand_idx_inject": np.array(rand_idx, dtype=np.int64),
                    }
                    if "bit_pos_inject" in {inp.name for inp in faulty_session.get_inputs()}:
                        feed["bit_pos_inject"] = np.array(bit_position, dtype=np.int32)

                    faulty_out = faulty_session.run(None, feed)[0]
                    faulty_pred = prediction(faulty_out)
                    linf = float(np.max(np.abs(golden_out.astype(np.float32) - faulty_out.astype(np.float32))))

                    writer.writerow({
                        "Timestamp": datetime.now().isoformat(),
                        "Layer_Config": layer_file,
                        "Layer_Type": config.get("layer_type", ""),
                        "Fault_Model": fault_model,
                        "Bit_Position": bit_position,
                        "Sample_Idx": args.sample_idx,
                        "Rand_Idx": rand_idx,
                        "Golden_Pred": golden_pred,
                        "Faulty_Pred": faulty_pred,
                        "Label": label,
                        "Prediction_Changed": golden_pred != faulty_pred,
                        "Golden_Correct": golden_pred == label,
                        "Faulty_Correct": faulty_pred == label,
                        "Linf_Diff": linf,
                    })
                    csvfile.flush()

                    logger.info(
                        f"{fault_model} bit={bit_position} rand_idx={rand_idx} "
                        f"pred {golden_pred}->{faulty_pred} linf={linf:.4e}"
                    )

                del faulty_session

    logger.info(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()
