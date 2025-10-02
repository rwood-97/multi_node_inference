#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

#SBATCH --time 0:30:0
#SBATCH --nodes 2
#SBATCH --gpus-per-node 4
#SBATCH --job-name test_multi_node_gh
#SBATCH --output test_multi_node_gh.log

module purge
module load brics/apptainer-multi-node

echo "--------------------------------------"
echo 
echo 
echo "New job: ${SLURM_JOB_ID}"
echo "--------------------------------------"

export APPTAINERENV_SLURM_NNODES=$SLURM_NNODES
# export APPTAINERENV_SLURM_PROCID=$SLURM_PROCID
# export APPTAINERENV_SLURM_LOCALID=$SLURM_LOCALID
# export APPTAINERENV_SLURM_NODEID=$SLURM_NODEID

# for vllm run
export PRIMARY_PORT=$((16384 + $SLURM_JOB_ID % 16384))
export PRIMARY_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export PRIMARY_IP=$(srun --nodes=1 --ntasks=1 -w $PRIMARY_HOST hostname -i | tr -d ' ')
echo "Primary IP: $PRIMARY_IP"

export APPTAINERENV_PRIMARY_PORT=$PRIMARY_PORT
export APPTAINERENV_PRIMARY_IP=$PRIMARY_IP

srun -n2 -l apptainer exec --nv --bind ${PWD}:/isambard_ai,${SCRATCH}:/scratch,${HF_HOME}:/hf_home container/e4s-cuda90-aarch64-25.06.4.sif /isambard_ai/vllm_run_gh.sh
wait

