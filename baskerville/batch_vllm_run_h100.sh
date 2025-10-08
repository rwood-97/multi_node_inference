#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

#SBATCH --qos turing
#SBATCH --account usjs9456-ati-test
#SBATCH --time 0:30:0
#SBATCH --nodes 2
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-gpu 36
#SBATCH --mem 0
#SBATCH --job-name test_multi_node_h100
#SBATCH --output test_multi_node_h100.log
#SBATCH --constraint=h100_80

module purge
module load baskerville

echo "--------------------------------------"
echo 
echo 
echo "New job: ${SLURM_JOB_ID}"
echo "--------------------------------------"

export APPTAINERENV_SLURM_NNODES=$SLURM_NNODES

# for vllm run
export PRIMARY_PORT=$((30000 + $SLURM_JOB_ID % 16384))
export PRIMARY_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export PRIMARY_IP=$(srun --nodes=1 --ntasks=1 -w $PRIMARY_HOST hostname -i | tr -d ' ')
echo "Primary IP: $PRIMARY_IP"

export APPTAINERENV_PRIMARY_PORT=$PRIMARY_PORT
export APPTAINERENV_PRIMARY_IP=$PRIMARY_IP

srun -n2 apptainer exec --nv --bind ${PWD}:/baskerville,/scratch-global/slurm-jobs/rwood/:/scratch,${HF_HOME}:/hf_home container/container_vllm.sif /baskerville/vllm_run_h100.sh
wait

