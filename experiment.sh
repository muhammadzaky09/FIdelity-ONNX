#python test_onnx_optimized_custom_inference.py --directory_name input --module Encoder 

#python parallelized_inject_onnx_transformer.py --directory_name input/decoder --module Decoder --experiment_output_name results_fault_injection/results.csv 
python parallelized_inject_onnx_transformer.py --directory_name input/encoder --module Encoder --experiment_output_name results_fault_injection/results.csv 
