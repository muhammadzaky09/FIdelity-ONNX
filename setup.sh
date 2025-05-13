pip install -r requirements.txt
python parser.py
mkdir decoders/7B16
cp tokenizer.model decoders/7B16/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/onnx-transformer/llama
