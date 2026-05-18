#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=vllm_bench
#SBATCH --output=vllm_bench.log

#--- Configuration ---
# Essential environment variables and paths are centralized here for clarity.

# Set the home directory for Hugging Face assets.
export HF_HOME="<<replace_me>>"
# Specify a scratch directory for temporary files.
export SCRATCH_DIR="<<replace_me>>"
# Define the cache directory for Hugging Face models.
export HF_CACHE_DIR="${HF_HOME}/huggingface"
# Define the primary network interface for communication.
export NETWORK_INTERFACE="hsn0"

# Path to the Apptainer container image.
CONTAINER_IMAGE_PATH=$(realpath "${SLURM_SUBMIT_DIR}/e4s-cuda90-aarch64-25.06.4.sif")

#--- End of Configuration ---

set -e # Exit immediately if a command exits with a non-zero status.

# Load necessary modules for the environment.
module load brics/default
module load brics/apptainer-multi-node

# Ensure the script is run from the submission directory.
cd "$SLURM_SUBMIT_DIR" || exit

# Create necessary directories if they don't exist.
mkdir -p "$HF_HOME" "$HF_CACHE_DIR"

#--- Network Discovery ---
# Dynamically discover the IP addresses of the allocated nodes.

mapfile -t ALL_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

NODE_IP_LIST=""
for host in "${ALL_HOSTS[@]}"; do
    ip=$(srun --nodes=1 --ntasks=1 -w "$host" ip -4 addr show "$NETWORK_INTERFACE" < /dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
    if [[ -z "$ip" ]]; then
        echo "FATAL: Could not discover IP for host ${host} on interface ${NETWORK_INTERFACE}"
        exit 1
    fi
    NODE_IP_LIST+="${ip} "
done

export PRIMARY_IP=$(echo "$NODE_IP_LIST" | awk '{print $1}')
export GCS_PORT=$((20000 + ($SLURM_JOB_ID % 16384)))
export PRIMARY_HOST=${ALL_HOSTS[0]}

#--- Job Information ---
echo "--------------------------------------"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Allocated Nodes: ${SLURM_JOB_NODELIST}"
echo "Primary Host: ${PRIMARY_HOST}"
echo "Primary IP (${NETWORK_INTERFACE}): ${PRIMARY_IP}"
echo "Ray GCS Port: ${GCS_PORT}"
echo "Discovered Node IPs: ${NODE_IP_LIST}"
echo "--------------------------------------"

#--- Cleanup Function ---
# Ensures a clean exit by stopping the Ray cluster.

# A file to signal the completion of the main task.
DONE_FILE="${SCRATCH_DIR}/ray_job_done_${SLURM_JOB_ID}"
rm -f "$DONE_FILE"

cleanup() {
    echo "--- Performing cleanup ---"
    # Gracefully stop the Ray cluster on the primary node.
    srun --nodes=1 --ntasks=1 -w "$PRIMARY_HOST" \
        apptainer exec --nv \
        --bind "${SLURM_SUBMIT_DIR}:/workspace" \
        "${CONTAINER_IMAGE_PATH}" \
        bash -c "source /py3.10-vllm/bin/activate && ray stop -f" &> /dev/null || true
    rm -f "$DONE_FILE"
    echo "Cleanup complete."
}
trap cleanup EXIT

#--- Ray Cluster and vLLM Execution ---
# This section launches the Ray cluster and executes the vLLM benchmark.

srun --label --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    apptainer exec --nv \
    --bind "${SLURM_SUBMIT_DIR}:/workspace,${SCRATCH_DIR}:/scratch,${HF_HOME}:/hf_home,${HF_CACHE_DIR}:/hf_cache_dir" \
    --env "PRIMARY_IP=${PRIMARY_IP},GCS_PORT=${GCS_PORT},NODE_IP_LIST=${NODE_IP_LIST},NETWORK_INTERFACE=${NETWORK_INTERFACE},DONE_FILE=${DONE_FILE},HF_HOME=/hf_home,HF_CACHE_DIR=/hf_cache_dir" \
    "${CONTAINER_IMAGE_PATH}" \
    bash -c '
        set -e
        source /py3.10-vllm/bin/activate

        NODE_ID=$((SLURM_NODEID + 1))
        NODE_IP=$(echo "$NODE_IP_LIST" | cut -d" " -f$NODE_ID)
        NODE_TEMP_DIR="/tmp/ray_temp_${SLURM_JOB_ID}"

        # Set the host IP for vLLM to ensure correct communication.
        export VLLM_HOST_IP="${NODE_IP}"

        echo "Host: $(hostname -s), Assigned IP: ${NODE_IP}"

        if [[ "$SLURM_NODEID" -eq 0 ]]; then
            echo "Starting Ray head node..."
            ray start --head --node-ip-address "$PRIMARY_IP" --port "$GCS_PORT" --temp-dir "$NODE_TEMP_DIR" --disable-usage-stats
        else
            sleep 10
            echo "Starting Ray worker node, connecting to ${PRIMARY_IP}:${GCS_PORT}..."
            ray start --address "${PRIMARY_IP}:${GCS_PORT}" --node-ip-address "$NODE_IP" --disable-usage-stats
        fi

        # The primary node orchestrates the benchmark.
        if [[ "$SLURM_NODEID" -eq 0 ]]; then
            echo "Waiting for worker nodes to register..."
            sleep 25
            ray status

            echo "Running vLLM throughput benchmark..."
            export RAY_ADDRESS="${PRIMARY_IP}:${GCS_PORT}"
            export NCCL_SOCKET_IFNAME="${NETWORK_INTERFACE}"

            vllm bench throughput \
                --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
                --download-dir /hf_cache_dir \
                --input-len 512 \
                --output-len 1024 \
                --tensor-parallel-size 4 \
                --pipeline-parallel-size ${SLURM_NNODES} \
                --distributed-executor-backend ray

            echo "Benchmark finished. Signaling worker nodes to exit."
            touch "$DONE_FILE"
        else
            echo "Worker node is ready and awaiting job completion signal..."
            while [ ! -f "$DONE_FILE" ]; do
                sleep 5
            done
            echo "Completion signal received. Worker node exiting."
        fi

        echo "Node task finished."
    '

echo "Job completed successfully."