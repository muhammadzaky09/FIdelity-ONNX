pip install -r requirements.txt
mkdir injection_llm 
mkdir alpaca
cp tokenizer.model alpaca/
python parser.py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/onnx-transformer/llama
