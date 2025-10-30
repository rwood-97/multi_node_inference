#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

#SBATCH --time 0:30:0
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-gpu 72
#SBATCH --mem 0
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

apptainer exec --nv --bind ${PWD}/../:/transformers_examples,$HF_HOME:/hf_home container/e4s-cuda90-aarch64-25.06.4.sif /transformers_examples/isambard_ai/transformers_run_gh.sh
wait