FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS bitflip-builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    && apt-get clean

COPY onnxruntime-dev ./onnxruntime-dev
COPY onnx-cuda-bitflip ./onnx-cuda-bitflip

RUN ./onnx-cuda-bitflip/build.sh \
    && mkdir -p /artifacts \
    && cp onnx-cuda-bitflip/python/onnx_bitflip.so /artifacts/onnx_bitflip.so

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && apt-get clean

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt 

COPY . .

COPY --from=bitflip-builder /artifacts/onnx_bitflip.so /app/llama/onnx_bitflip.so

ENV LD_LIBRARY_PATH="/app/llama:/app/onnxruntime-dev/onnxruntime-linux-x64-gpu-1.20.1/lib:${LD_LIBRARY_PATH}"

CMD ["bash"]
