pip install -r requirements.txt
mkdir decoders/7B16
cp tokenizer.model decoders/7B16/
python parser.py decoders/7B16
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/onnx-transformer/llama
