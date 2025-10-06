#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

#SBATCH --time 0:30:0
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4
#SBATCH --job-name one_node_gh
#SBATCH --output one_node_gh.log

echo "--------------------------------------"
echo 
echo 
echo "New job: ${SLURM_JOB_ID}"
echo "--------------------------------------"

module purge
module load brics/default
module load brics/apptainer-multi-node

export APPTAINERENV_SLURM_NNODES=$SLURM_NNODES

# for vllm run
export PRIMARY_PORT=$((30000 + $SLURM_JOB_ID % 16384))
export PRIMARY_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export PRIMARY_IP=$(srun --nodes=1 --ntasks=1 -w $PRIMARY_HOST hostname -i | tr -d ' ')
echo "Primary IP: $PRIMARY_IP"

export APPTAINERENV_PRIMARY_PORT=$PRIMARY_PORT
export APPTAINERENV_PRIMARY_IP=$PRIMARY_IP

srun -N${SLURM_NNODES} -n${SLURM_NNODES} -l apptainer exec --nv --bind ${PWD}:/isambard_ai,${SCRATCH}:/scratch,${HF_HOME}:/hf_home container/e4s-cuda90-aarch64-25.06.4.sif /isambard_ai/vllm_run_gh.sh
wait

