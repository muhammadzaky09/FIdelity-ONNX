import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--run-cuda",
        action="store_true",
        default=False,
        help="Run tests marked cuda. These require CUDAExecutionProvider and compatible custom op libraries.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires CUDAExecutionProvider and CUDA-compatible custom ops")
    config.addinivalue_line("markers", "custom_ops: validates ONNX Runtime custom op libraries")
    config.addinivalue_line("markers", "slow: slower validation that may need model assets")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-cuda"):
        return
    skip_cuda = pytest.mark.skip(reason="CUDA validation is opt-in; run with --run-cuda")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
