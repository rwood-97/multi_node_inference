#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

#SBATCH --qos turing
#SBATCH --account usjs9456-ati-test
#SBATCH --time 0:30:0
#SBATCH --nodes 1 
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-gpu 36
#SBATCh --mem 0
#SBATCH --ntasks-per-node 4
#SBATCH --job-name one_node
#SBATCH --output one_node.log
#SBATCH --constraint=a100_80

echo "--------------------------------------"
echo 
echo 
echo "New job: ${SLURM_JOB_ID}"
echo "--------------------------------------"

module purge
module load baskerville
module load Python NCCL

export NCCL_SOCKET_IFNAME=ib0        # or the IB interface on Baskerville
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=INFO               # optional, for debugging

# for the python script
export MASTER_PORT=$((30000 + $SLURM_JOB_ID % 16384))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# for vllm run
export PRIMARY_PORT=$MASTER_PORT
export PRIMARY_HOST=$MASTER_ADDR
#export PRIMARY_IP=$(getent hosts $PRIMARY_HOST | awk '{print $1}')
export PRIMARY_IP=$(srun --nodes=1 --ntasks=1 -w $PRIMARY_HOST hostname -i | tr -d ' ')
echo "Primary IP: $PRIMARY_IP"

source ../venv_a100/bin/activate
echo $(which python)

# run vllm
srun -N${SLURM_NNODES}  --ntasks-per-node=1 -l ./vllm_vlm.sh
wait

