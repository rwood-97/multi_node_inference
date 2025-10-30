#!/bin/bash

source venv_a100/bin/activate
echo $(which python)

echo "Primary IP: $PRIMARY_IP"

# get this node’s hostname
NODE_HOST=$(hostname -s)
export VLLM_HOST_IP=$(hostname -i)

echo "Node ID: $NODE_HOST"
echo "Node IP: $VLLM_HOST_IP"

# get Slurm task ID
PROC_ID=$SLURM_PROCID
echo "Process ID: $PROC_ID"

echo "Host IP: $VLLM_HOST_IP, Primary IP: $PRIMARY_IP"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# if this is the 0th process
if [[ "$SLURM_NODEID" -eq 0 && "$SLURM_LOCALID" -eq 0 ]]; then
    echo "Starting Ray head on $PRIMARY_IP:$PRIMARY_PORT"
    ray start --head --node-ip-address $PRIMARY_IP --port $PRIMARY_PORT
elif [[ "$SLURM_LOCALID" -eq 0 ]]; then
    echo "Starting Ray worker (proc $PROC_ID) connecting to $PRIMARY_IP:$PRIMARY_PORT"
    ray start --block --address $PRIMARY_IP:$PRIMARY_PORT --node-ip-address $VLLM_HOST_IP
fi

# sleep to ensure ray is set up
sleep 20

# only proc 0 runs vLLM benchmark
if [[ "$SLURM_PROCID" -eq 0 ]]; then
    ray status
    ray list nodes
	
    if [[ "$SLURM_NNODES" -eq 1 ]]; then
        MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
    else
        MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
    fi

    echo "Running ${MODEL} with vLLM..."
    vllm serve $MODEL \
	--tokenizer-mode auto \
    -tp 4 -pp ${SLURM_NNODES} \
	--distributed-executor-backend ray &

    # Wait for the REST API to be available
    until curl -s http://localhost:8000/v1/models >/dev/null 2>&1; do
        sleep 20
        echo "Waiting for vLLM to start..."
    done

    sleep 3000
    echo "Done!"
fi


