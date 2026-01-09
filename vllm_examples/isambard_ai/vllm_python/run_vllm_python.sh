#!/bin/bash

source .venv/bin/activate
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

# if this is the 0th process
if [[ "$SLURM_NODEID" -eq 0 && "$SLURM_LOCALID" -eq 0 ]]; then
    echo "Starting Ray head on $PRIMARY_IP:$PRIMARY_PORT"
    ray start --head --node-ip-address $PRIMARY_IP --port $PRIMARY_PORT
elif [[ "$SLURM_LOCALID" -eq 0 ]]; then
    sleep 10
    echo "Starting Ray worker (proc $PROC_ID) connecting to $PRIMARY_IP:$PRIMARY_PORT"
    ray start --address $PRIMARY_IP:$PRIMARY_PORT --node-ip-address $VLLM_HOST_IP
fi

# sleep to ensure ray is set up
sleep 20

echo "Running vLLM benchmark..."
python run_vllm.py


