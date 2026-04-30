mkdir -p onnxruntime-dev
cd onnxruntime-dev
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.20.1.tgz
docker build -f Dockerfile -t onnx-transformer:local .