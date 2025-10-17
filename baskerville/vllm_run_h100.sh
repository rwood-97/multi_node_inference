#!/bin/bash

export HF_HOME=/hf_home/

echo $(which python)

export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_HCA=mlx5

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
    ray start --address $PRIMARY_IP:$PRIMARY_PORT --node-ip-address $VLLM_HOST_IP
fi

# sleep to ensure ray is set up
sleep 20

# only proc 0 runs vLLM benchmark
if [[ "$SLURM_PROCID" -eq 0 ]]; then
    ray status 

    if [[ "$SLURM_NNODES" -eq 1 ]]; then
        MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    else
        MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    fi

    echo "Running ${MODEL} with vLLM..."
    vllm bench throughput \
        --model $MODEL \
        --input-len 512 \
        --output-len 1024 \
        -tp 4 -pp ${SLURM_NNODES} \
        --distributed-executor-backend ray
fi
