#!/bin/bash

#/.singularity.d/runscript

# adapt container for multi-node
if [[ "$SLRUM_NNODES" -gt 1 ]]; then
    source /host/adapt.sh
fi

source /py3.10-vllm/bin/activate
echo $(which python)

echo "Primary IP: $PRIMARY_IP"

export HF_HOME=/hf_home/
export NCCL_SOCKET_IFNAME="${NETWORK_INTERFACE}"

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
    
    echo "Running vLLM benchmark..."
    vllm bench throughput \
        --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
        --input-len 512 \
        --output-len 1024 \
        -tp 4 -pp ${SLURM_NNODES} \
        --distributed-executor-backend ray
    ray stop
    echo "Done!"
fi


