module purge
module load baskerville
module load Python CUDA

export PIP_CACHE_DIR=$(PWD -P)/.cache/pip
export HF_HOME=$(PWD -P)/.cache/huggingface

python -m venv venv_a100
source venv_a100/bin/activate
echo $(which python)

python -m pip install -r requirements.txt

# run qwen_chat.py
python ../qwen_chat.py