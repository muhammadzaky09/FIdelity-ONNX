import json
from argparse import Namespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import cnn_inference


class FakeInput:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeSessionOptions:
    def __init__(self, registrations):
        self.registrations = registrations
        self.libs = []

    def register_custom_ops_library(self, path):
        self.libs.append(path)
        self.registrations.append(path)


class FakeSession:
    def __init__(self, model_path, sess_options=None, providers=None):
        self.model_path = model_path
        self.sess_options = sess_options
        self.providers = providers

    def get_inputs(self):
        return [
            FakeInput("X", [1, 2]),
            FakeInput("rand_idx_inject", []),
            FakeInput("bit_pos_inject", []),
        ]

    def run(self, output_names, feeds):
        return [np.array([[0.1, 0.9]], dtype=np.float32)]


def make_model(path):
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    w = helper.make_tensor("W", TensorProto.FLOAT, [2, 2], np.eye(2, dtype=np.float32).flatten())
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"], name="target")],
        "runtime_requirement_model",
        [x],
        [y],
        [w],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)


def run_inference_with_fake_runtime(tmp_path, monkeypatch, fault_model, precision):
    model_path = tmp_path / "model.onnx"
    config_dir = tmp_path / "configs"
    output_csv = tmp_path / "results.csv"
    config_dir.mkdir()
    make_model(model_path)

    config = {
        "model_name": str(model_path),
        "target_layer": "target",
        "input_tensor": "X",
        "weight_tensor": "W",
        "layer_type": "MatMul",
    }
    (config_dir / "layer.json").write_text(json.dumps(config))

    registrations = []
    monkeypatch.setattr(cnn_inference, "load_image", lambda dataset, sample_idx: (
        np.array([[1.0, 2.0]], dtype=np.float32),
        1,
    ))
    monkeypatch.setattr(cnn_inference, "modify_onnx_graph", lambda config, model_config, fault_model: str(model_path))
    monkeypatch.setattr(cnn_inference.ort, "SessionOptions", lambda: FakeSessionOptions(registrations))
    monkeypatch.setattr(cnn_inference.ort, "InferenceSession", FakeSession)

    args = Namespace(
        config_dir=str(config_dir),
        dataset="cifar10",
        sample_idx=0,
        precision=precision,
        fault_models=[fault_model],
        bit_position=0,
        provider="CPUExecutionProvider",
        seed=0,
        output_csv=str(output_csv),
    )
    cnn_inference.inference(args)
    return registrations


@pytest.mark.parametrize("fault_model", ["INPUT", "WEIGHT", "INPUT16", "WEIGHT16", "RANDOM_BITFLIP"])
def test_fp16_fault_models_that_flip_bits_register_onnx_bitflip(tmp_path, monkeypatch, fault_model):
    registrations = run_inference_with_fake_runtime(tmp_path, monkeypatch, fault_model, "float16")
    assert any(path.endswith("llama/onnx_bitflip.so") for path in registrations)


def test_fp16_random_fault_does_not_need_onnx_bitflip(tmp_path, monkeypatch):
    registrations = run_inference_with_fake_runtime(tmp_path, monkeypatch, "RANDOM", "float16")
    assert not any(path.endswith("llama/onnx_bitflip.so") for path in registrations)


def test_float32_random_bitflip_registers_onnxruntime_extensions(tmp_path, monkeypatch):
    registrations = run_inference_with_fake_runtime(tmp_path, monkeypatch, "RANDOM_BITFLIP", "float32")
    assert registrations
    assert not any(path.endswith("llama/onnx_bitflip.so") for path in registrations)
