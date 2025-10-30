module purge
module load baskerville
module load Python

source venv_a100/bin/activate
echo $(which python)

# test nccl works
python ../qwen_chat.py