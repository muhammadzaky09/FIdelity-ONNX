pip install -r requirements.txt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$SCRIPT_DIR/llama"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124