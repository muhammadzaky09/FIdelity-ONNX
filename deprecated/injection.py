#!/usr/bin/env python3

import argparse
import json
import os
import re  
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from loguru import logger
import glob, os
from collections import defaultdict
from torchvision import datasets, transforms
from graph import modify_onnx_graph

def load_cifar10(num_samples: int = 32):
    tfm = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = datasets.CIFAR10(root="/tmp", train=False, download=True, transform=tfm)
    images, labels = [], []
    for i in range(num_samples):
        img, lab = testset[i]
        images.append(img.numpy())
        labels.append(lab)
    x = np.stack(images, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return x, y


DATASET_LOADERS = {
    "cifar10": load_cifar10,
}


def main():
    ap = argparse.ArgumentParser(description="Fault-injection evaluation driver")
    ap.add_argument("--config_dir", required=True, help="Directory containing layer JSON config files")
    ap.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), required=True)
    ap.add_argument("--fault_model", required=True, choices=["INPUT", "INPUT16", "WEIGHT", "WEIGHT16", "RANDOM", "RANDOM_BITFLIP"])
    ap.add_argument("--bit_position", type=int, default=None, help="If omitted, iterate over all bit positions allowed by precision")
    ap.add_argument("--precision", default="int8", choices=["int8", "int4", "float16"])
    ap.add_argument("--provider", default="CPUExecutionProvider", choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
                    help="ONNX Runtime execution provider to use for **all** sessions")
    args = ap.parse_args()
    cfg_paths = glob.glob(os.path.join(args.config_dir, '*.json'))
    if not cfg_paths:
        raise RuntimeError('No *.json layer configs found in directory')

    layers_by_model = defaultdict(list)
    layer_cfgs = []
    for p in cfg_paths:
        with open(p,'r') as fh:
            cfg = json.load(fh)
        layers_by_model[cfg['model_name']].append(cfg)
        layer_cfgs.append(cfg)
    IDX_RE = re.compile(r"(\d+)(?=\.onnx$)")

    def _extract_idx(path: str):
        basename = os.path.basename(path)
        matches = IDX_RE.findall(basename)
        return int(matches[-1]) if matches else None

    def _sort_paths(paths: List[str]) -> List[str]:
        # If *all* paths carry a numeric suffix, use it; otherwise keep
        # the original discovery order to avoid mis-ordering encoder/decoder.
        idxs = [_extract_idx(p) for p in paths]
        if all(i is not None for i in idxs):
            return [p for p, _ in sorted(zip(paths, idxs), key=lambda t: t[1])]
        return paths  # preserve given order

    def build_session_chain(paths: List[str]) -> List[ort.InferenceSession]:
        sorted_paths = _sort_paths(paths)
        return [ort.InferenceSession(p, providers=[args.provider]) for p in sorted_paths]

    def forward_chain(sess_chain: List[ort.InferenceSession], first_feed: Dict[str, np.ndarray]):
        feed = dict(first_feed) 
        for sess in sess_chain:
            outputs = sess.run(None, feed)
            feed.update({out.name: tensor for out, tensor in zip(sess.get_outputs(), outputs)})
        return outputs[0]

    seen = set()
    ordered_unique = []
    for cfg in layer_cfgs:
        p = cfg['model_name']
        if p not in seen:
            seen.add(p)
            ordered_unique.append(p)

    unique_model_paths = _sort_paths(ordered_unique)

    tmp_sess = ort.InferenceSession(unique_model_paths[0], providers=[args.provider])
    input_name = tmp_sess.get_inputs()[0].name
    del tmp_sess

    x, y = DATASET_LOADERS[args.dataset]()
    inputs_dict = {input_name: x}

    golden_chain = build_session_chain(unique_model_paths)
    path_to_position = {p: i for i, p in enumerate(unique_model_paths)}

    llama_cfg = { 'precision': args.precision }

    if args.bit_position is None:
        max_bit = { 'int8':7, 'int4':3, 'float16':15}[args.precision]
        bit_positions = range(max_bit+1)
    else:
        bit_positions = [args.bit_position]

    # iterate experiments
    for layer_cfg in layer_cfgs:
        print(f"\nLayer {layer_cfg['target_layer']} in {layer_cfg['model_name']}")
        for bit in bit_positions:
            gold_out = forward_chain(golden_chain, inputs_dict)
            is_classification = (
                gold_out.ndim == 2 and gold_out.shape[1] > 1 and y is not None
            )

            gold_pred = np.argmax(gold_out, axis=1) if is_classification else None
            faulty_path = modify_onnx_graph(layer_cfg, llama_cfg, args.fault_model, bit)
            faulty_chain = list(golden_chain)
            stage_idx = path_to_position[layer_cfg['model_name']]
            faulty_chain[stage_idx] = ort.InferenceSession(faulty_path, providers=[args.provider])

            faulty_out = forward_chain(faulty_chain, inputs_dict)

            if is_classification:
                faulty_pred = np.argmax(faulty_out, axis=1)
                acc_fault = (faulty_pred == y).mean()
                changed = (faulty_pred != gold_pred).sum()
                print(f"bit {bit:2d}: acc {acc_fault:.3f}  Δpred {changed}")
            else:
                l_inf = float(np.max(np.abs(gold_out - faulty_out)))
                print(f"bit {bit:2d}: L∞ diff {l_inf:.4e}")

            del faulty_chain[stage_idx]
           

            if os.path.exists(faulty_path):
                try:
                    os.remove(faulty_path)
                except OSError as e:
                    logger.warning(f"Could not delete temp model {faulty_path}: {e}")


if __name__ == '__main__':
    main() 