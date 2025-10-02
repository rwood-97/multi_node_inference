#!/bin/bash

/.singularity.d/runscript

# adapt container for multi-node
source /host/adapt.sh

source /py3.10-vllm/bin/activate
echo $(which python)

echo "Primary IP: $PRIMARY_IP"
echo $SLURM_NNODES
echo $SLURM_PROCID
echo $SLURM_LOCALID
echo $SLURM_NODEID

export HF_HOME=/hf_home/

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

# only proc 0 runs ray status/list nodes
if [[ "$SLURM_NODEID" -eq 0 && "$SLURM_PROCID" -eq 0 ]]; then
    ray status 
    ray list nodes
fi

python -c "import torch; torch.cuda.is_available()"

echo
echo

# only proc 0 runs vLLM benchmark
if [[ "$SLURM_PROCID" -eq 0 ]]; then
    echo "Running vLLM benchmark..."
    vllm bench throughput \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --input-len 512 \
        --output-len 1024 \
        -tp 4 -pp ${SLURM_NNODES} \
        --distributed-executor-backend ray
fi


