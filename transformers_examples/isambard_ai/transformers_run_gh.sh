
source /py3.10-vllm/bin/activate
echo $(which python)

export HF_HOME=/hf_home/

python /transformers_examples/qwen.py
