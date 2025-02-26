import os
def extract_decoder_idx(path):
    filename = os.path.basename(path)
    if 'decoder-merge-' in filename:
        decoder_idx_str = filename.split('decoder-merge-')[1].split('-')[0]
        
        return decoder_idx_str
print(extract_decoder_idx('decoder-merge-20.onnx')) 