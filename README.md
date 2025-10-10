# Example scripts for running multi-node inference with vLLM on HPCs (Baskerville A100 GPUs and H100 GPUs and Isambard-AI GH200 GPUs)

This repository contains example scripts for running multi-node inference with vLLM on HPCs (Baskerville A100 GPUs and H100 GPUs and Isambard-AI GH200 chips with H100 GPUs).

The models used are:
- [Qwen/Qwen3-235B-A22B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8)
- [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)

## Baskerville

The Baskerville system is made up of 48 nodes with A100 GPUs, 37 of which have 40GB vRAM and 11 of which have 80GB vRAM. 
There are also 2 nodes with H100 GPUs, each with 80GB vRAM. 
For both A100 and H100 nodes, there are 4 GPUs per node and you can request up to 36 cores per GPU.
More information can be found [here](https://docs.baskerville.ac.uk/system/).

Baskerville's A100 nodes primarily use modules to manage software environments, loaded via `module load <module_name>`. This includes thing like Python, CUDA, NCCL, etc.
In contrast, the H100 nodes require users to install their own software and so are more suited to containerised workflows.

Baskerville scripts are in the `baskerville` directory.

### A100 nodes

The scripts for the A100 nodes are:
- `batch_vllm_run.sh`
- `batch_vllm_run_1node.sh`
- `vllm_run.sh`

These scripts use the Python (3.11.3) and NCCL (2.20.5 + CUDA 12.3) modules.
They also use a venv located at `../venv_a100/`.

For the venv, you will need to:
- Use `srun` to get onto an A100 node
- Load the `Python` and `NCCL` modules 
- Create the venv (`python -m venv venv_a100`) 
- Install `vllm==0.8.5`

### H100 nodes

The scripts for the H100 nodes are:
- `batch_vllm_run_h100.sh`
- `vllm_run_h100.sh`

These scripts use a container defined by the `container/container_vllm.def` file.
You will need to build the container yourself, to do this:
- Use `srun` to get onto an H100 node
- Build the container with `apptainer build container_vllm.sif container/container_vllm.def` (note: It might take a while to build!)

## Isambard-AI

The Isambard-AI (Phase 2) system is made up of 1320 nodes with GH200 chips. Each GH200 chip has 1 Grace CPU and 1 H100 GPU. There are 4 GH200 chips per node and 72 cores per GPU. 
Each H100 GPU has 80GB vRAM. 
More information can be found [here](https://docs.isambard.ac.uk/system/).

The Isambard-AI scripts are in the `isambard-ai` directory.

The scripts for Isambard-AI are:
- `batch_vllm_run_gh.sh`
- `batch_vllm_run_gh_1node.sh`
- `vllm_run_gh.sh`

Additionally, there are some scripts for running [Llama-3.3-70B-Instruct-FP8](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8) and for running vLLM using python (rather than command line):
- `batch_vllm_llama.sh`
- `vllm_llama.sh`
- `batch_vllm_python.sh`
- `vllm_python.sh`
- `run_vllm.py`

All the above scripts use the `e4s-cuda90-aarch64-25.06.4.sif` container.
This is a pre-built container image which is accessible on the Isambard-AI filesystem at `/projects/public/brics/containers/e4s/cuda90-aarch64-25.06.4.sif`.
You will need to copy or symlink this image into to the `container` directory before running the scripts.

## Refs

- https://docs.vllm.ai/en/v0.9.2/serving/distributed_serving.html
- https://github.com/vllm-project/vllm/blob/main/examples/online_serving/run_cluster.sh
- [vllm #26028](https://github.com/vllm-project/vllm/issues/26028)
- https://swaglu.com/llama-405b-vllm-slurm-multinode/#step-1-multi-node-slurm-configuration
- https://docs.isambard.ac.uk/user-documentation/guides/containers/apptainer-multi-node/
