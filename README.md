# ONNX-transformer

## Setup Instructions for Fault Injection Code

To set up the fault injection code for the ONNX-transformer project, follow these steps:

1. **Install Required Packages:**
   Ensure you have Python and pip installed on your system. Then, install the necessary Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Decoder Directory:**
   Create the required directory structure for the decoders:
   ```bash
   mkdir -p decoders/7B16
   ```

3. **Copy Tokenizer Model:**
   Copy the tokenizer model to the decoder directory (Llama-2-7B Tokenizer):
   ```bash
   cp tokenizer.model decoders/7B16/
   ```

4. **Run the Parser Script:**
   Execute the parser script to process the ONNX models and identify injection targets
   ```bash
   python parser.py decoders/7B16
   ```

5. **Set Environment Variables:**
   Update the `LD_LIBRARY_PATH` environment variable to include the necessary library paths (in this case custom ONNX operator):
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/onnx-transformer/llama
   ```

By following these steps, you will set up the environment for running the fault injection code on ONNX models. Ensure that all paths and filenames are correct and adjust them as necessary for your specific setup.


