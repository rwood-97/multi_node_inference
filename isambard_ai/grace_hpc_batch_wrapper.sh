#!/bin/bash

# get start date
start=$(date +%Y-%m-%d)
sleep 5

# run the job
job_id=$(sbatch batch_vllm_run_gh.sh | awk '{print $NF}')

# wait till the job is finished
until sacct -j "$job_id" --format=JobID,State --noheader | grep -qE "^${job_id}\s+(COMPLETED|FAILED|CANCELLED|TIMEOUT)"; do
    echo "Waiting for job (${job_id}) to finish"
    sleep 100
done

echo "Job finished, running GRACE-HPC"
# get end date
end=$(date -d "+1 day" +%Y-%m-%d)

source ../../venv_gracehpc/bin/activate
gracehpc run --StartDate $start --EndDate $end --JobID $job_id --Region 'South West England' --Scope3 'IsambardAI' > carbon_usage_${job_id}.txt
