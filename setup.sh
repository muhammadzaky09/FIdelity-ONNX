pip install -r requirements.txt
python parser.py decoders/7B16
mkdir decoders/7B16
cp tokenizer.model decoders/7B16/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/onnx-transformer/llama
