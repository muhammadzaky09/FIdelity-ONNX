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


def inference(args):
    image, label = load_image(args.dataset, args.sample_idx)

    bit_ranges = {
        "float16": range(16),
        "int8": range(8),
        "int4": range(4),
        "float32": range(32),
    }
    bit_positions = [args.bit_position] if args.bit_position is not None else list(bit_ranges[args.precision])
    output_csv = args.output_csv or f"cnn_results_{args.dataset}_{args.precision}.csv"

    paths = sorted(Path(args.config_dir).glob("*.json"))
    if not paths:
        raise ValueError(f"No *.json layer configs found in {args.config_dir}")

    layer_configs = []
    for path in paths:
        with open(path) as f:
            layer_configs.append((path.name, json.load(f)))

    golden_session = None
    input_name = None
    current_model = None

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
            if config["model_name"] != current_model:
                current_model = config["model_name"]
                golden_session = ort.InferenceSession(current_model, providers=[args.provider])
                candidate_inputs = [
                    inp for inp in golden_session.get_inputs()
                    if inp.name not in {"rand_idx_inject", "bit_pos_inject"}
                ]
                rank4_inputs = [inp for inp in candidate_inputs if len(inp.shape) == 4]
                input_name = (rank4_inputs[0] if rank4_inputs else candidate_inputs[0]).name

            golden_out = golden_session.run(None, {input_name: image})[0]
            golden_pred = int(np.argmax(golden_out.reshape(-1, golden_out.shape[-1]), axis=1)[0])
            inferred_model = onnx.shape_inference.infer_shapes(onnx.load(config["model_name"]))

            logger.info(f"Layer config: {layer_file}")
            logger.info(f"Golden prediction: {golden_pred}, label: {label}")

            for fault_model in args.fault_models:
                faulty_path = modify_onnx_graph(config, {"precision": args.precision}, fault_model)

                session_options = ort.SessionOptions()
                needs_bitflip = (
                    ("BITFLIP" in fault_model and args.precision == "float16")
                    or (args.precision == "float16" and "RANDOM" not in fault_model)
                )
                if needs_bitflip:
                    session_options.register_custom_ops_library(os.path.abspath("llama/onnx_bitflip.so"))
                if "BITFLIP" in fault_model and args.precision == "float32":
                    from onnxruntime_extensions import get_library_path
                    session_options.register_custom_ops_library(get_library_path())
                faulty_session = ort.InferenceSession(
                    faulty_path,
                    sess_options=session_options,
                    providers=[args.provider],
                )

                if fault_model in {"INPUT", "INPUT16"}:
                    target_tensor = config["input_tensor"]
                elif fault_model in {"WEIGHT", "WEIGHT16"}:
                    target_tensor = config["weight_tensor"]
                else:
                    target_tensor = config["target_layer"]
                    for node in inferred_model.graph.node:
                        if node.name == config["target_layer"] or config["target_layer"] in node.output:
                            target_tensor = node.output[0]
                            break

                target_shape = None
                for initializer in inferred_model.graph.initializer:
                    if initializer.name == target_tensor:
                        target_shape = list(initializer.dims)
                        break
                if target_shape is None:
                    for value_info in list(inferred_model.graph.input) + list(inferred_model.graph.value_info) + list(inferred_model.graph.output):
                        if value_info.name != target_tensor:
                            continue
                        shape = []
                        for dim in value_info.type.tensor_type.shape.dim:
                            if dim.dim_value <= 0:
                                shape = None
                                break
                            shape.append(dim.dim_value)
                        target_shape = shape
                        break
                if target_shape is None:
                    raise ValueError(f"Could not infer shape for tensor '{target_tensor}' in {config['model_name']}")
                rand_bound = int(np.prod(target_shape))

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
                    faulty_pred = int(np.argmax(faulty_out.reshape(-1, faulty_out.shape[-1]), axis=1)[0])
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
    inference(parser.parse_args())


if __name__ == "__main__":
    main()
