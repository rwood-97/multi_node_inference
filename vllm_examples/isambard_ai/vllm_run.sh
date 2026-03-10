#!/bin/bash
set -e

source .venv/bin/activate
echo $(which python)

echo "Primary IP: $PRIMARY_IP"

# get this node’s ip
NODE_IP=$(hostname -i)
export VLLM_HOST_IP=$NODE_IP
echo "vLLM host IP: $VLLM_HOST_IP"

# get Slurm task ID
PROC_ID=$SLURM_PROCID
echo "Process ID: $PROC_ID"

echo "Host IP: $VLLM_HOST_IP, Primary IP: $PRIMARY_IP"

# if this is the 0th process
if [[ "$SLURM_NODEID" -eq 0 && "$SLURM_LOCALID" -eq 0 ]]; then
    echo "Starting Ray head on $PRIMARY_IP:$PRIMARY_PORT"
    ray start --head --node-ip-address $PRIMARY_IP --port $PRIMARY_PORT
elif [[ "$SLURM_LOCALID" -eq 0 ]]; then
    echo "Starting Ray worker (proc $PROC_ID) connecting to $PRIMARY_IP:$PRIMARY_PORT"
    ray start --block --address $PRIMARY_IP:$PRIMARY_PORT --node-ip-address $VLLM_HOST_IP
    echo "Worker (proc $PROC_ID) stopped"
fi

# sleep to ensure ray is set up
sleep 20

echo
echo

# only proc 0 runs vLLM benchmark
if [[ "$SLURM_PROCID" -eq 0 ]]; then
    ray status 
    
    # Track GPU metrics
    nvidia-smi dmon -o TD -s puct -d 1 > log-train-gpu.txt &
    NVIDIA_SMI_PID=$!
    trap 'kill $NVIDIA_SMI_PID 2>/dev/null' EXIT

    if [[ "$SLURM_NNODES" -eq 1 ]]; then
        MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
    else
        MODEL="Qwen/Qwen3-235B-A22B-Thinking-2507"
    fi
    
    echo "Running ${MODEL} with vLLM..."
    vllm serve $MODEL \
	--tokenizer-mode auto \
    	--tensor-parallel-size 4 \
	--pipeline-parallel-size ${SLURM_NNODES} \
	--enable-auto-tool-choice \
	--tool-call-parser hermes &
    VLLM_PID=$!
    trap 'kill $VLLM_PID 2>/dev/null' EXIT

    # Wait for the REST API to be available
    until curl -s http://localhost:8000/v1/models >/dev/null 2>&1; do
        sleep 20
        echo "Waiting for vLLM to start..."
    done

    sleep 1800
    echo "Done!"
fi


