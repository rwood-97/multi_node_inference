#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

#SBATCH --qos turing
#SBATCH --account vjgo8416-hpc2511
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

# for hpc training, set cache dirs as part of project folder
export PIP_CACHE_DIR=/bask/projects/v/vjgo8416-hpc2511/.cache/pip
export HF_HOME=/bask/projects/v/vjgo8416-hpc2511/.cache/huggingface

python -m venv venv_a100
source venv_a100/bin/activate
echo $(which python)

python -m pip install -r requirements.txt

# run qwen.py
python ../qwen.py
