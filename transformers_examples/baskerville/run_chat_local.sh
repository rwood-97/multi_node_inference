module purge
module load baskerville
module load Python CUDA

# for hpc training, set cache dirs as part of project folder
export PIP_CACHE_DIR=/bask/projects/v/vjgo8416-hpc2511/.cache/pip
export HF_HOME=/bask/projects/v/vjgo8416-hpc2511/.cache/huggingface

python -m venv venv_a100
source venv_a100/bin/activate
echo $(which python)

python -m pip install -r requirements.txt

# run qwen_chat.py
python ../qwen_chat.py