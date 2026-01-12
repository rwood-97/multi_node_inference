#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

#SBATCH --time 0:30:0
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-gpu 72
#SBATCH --mem 0
#SBATCH --job-name one_node
#SBATCH --output one_node.log

echo "--------------------------------------"
echo 
echo 
echo "New job: ${SLURM_JOB_ID}"
echo "--------------------------------------"

module purge
module load brics/default

# for vllm run
export PRIMARY_PORT=$((30000 + $SLURM_JOB_ID % 16384))
export PRIMARY_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export PRIMARY_IP=$(srun --nodes=1 --ntasks=1 -w $PRIMARY_HOST hostname -i | tr -d ' ')
echo "Primary IP: $PRIMARY_IP"

# create venv
uv venv --allow-existing --seed --python=3.12
source .venv/bin/activate
echo $(which python)

# install vllm 0.13.0 and ray
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/0.13.0/vllm
uv pip install ray[client]

srun -N${SLURM_NNODES} -n${SLURM_NNODES} -l vllm_run.sh
wait

