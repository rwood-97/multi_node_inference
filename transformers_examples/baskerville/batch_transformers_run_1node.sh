#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

#SBATCH --qos turing
#SBATCH --account usjs9456-ati-test
#SBATCH --time 0:30:0
#SBATCH --nodes 1 
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-gpu 36
#SBATCH --mem 0
#SBATCH --ntasks-per-node 1
#SBATCH --job-name one_node
#SBATCH --output one_node.log

echo "--------------------------------------"
echo 
echo 
echo "New job: ${SLURM_JOB_ID}"
echo "--------------------------------------"

module purge
module load baskerville
module load Python CUDA

export PIP_CACHE_DIR=$(PWD -P)/.cache/pip
export HF_HOME=$(PWD -P)/.cache/huggingface

python -m venv venv_a100
source venv_a100/bin/activate
echo $(which python)

python -m pip install -r requirements.txt

# run qwen.py
python ../qwen.py
