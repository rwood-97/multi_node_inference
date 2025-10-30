# Multi-node inference on HPC

This repository contains example scripts for running multi-node inference with [vLLM](https://github.com/vllm-project/vllm) on Baskerville and Isambard-AI HPCs.
It also contains some simple example scripts for running single-node inference on these HPCs with the [transformers](https://github.com/huggingface/transformers) library.

The models used are:
- [Qwen/Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) 
- [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- [nvidia/Llama-3.3-70B-Instruct-FP8](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8)

## Baskerville

The Baskerville system is made up of 48 nodes with A100 GPUs, 37 of which have 40GB vRAM and 11 of which have 80GB vRAM. 
There are also 2 nodes with H100 GPUs, each with 80GB vRAM. 
For both A100 and H100 nodes, there are 4 GPUs per node and you can request up to 36 cores per GPU.
More information can be found [here](https://docs.baskerville.ac.uk/system/).

Baskerville's A100 nodes primarily use modules to manage software environments, loaded via `module load <module_name>`. This includes thing like Python, CUDA, NCCL, etc.
In contrast, the H100 nodes require users to install their own software and so are more suited to containerised workflows.

## Isambard-AI

The Isambard-AI (Phase 2) system is made up of 1320 nodes with GH200 chips. Each GH200 chip has 1 Grace CPU and 1 H100 GPU. There are 4 GH200 chips per node and 72 cores per GPU. 
Each H100 GPU has 80GB vRAM. 
More information can be found [here](https://docs.isambard.ac.uk/system/).

## vLLM scripts

Scripts for running multi-node inference with vLLM are in the `vllm_examples` directory.

### Baskerville

#### A100 nodes

The scripts for the A100 nodes are:
- `batch_vllm_run.sh`
- `batch_vllm_run_1node.sh`
- `vllm_run.sh`

These scripts use the Python (3.11.3) and NCCL (2.20.5 + CUDA 12.3) modules.
They also use a venv located at `venv_a100`.

To run the scripts, use:

``` bash
sbatch batch_vllm_run.sh
```

or, for a single node:

``` bash
sbatch batch_vllm_run_1node.sh
```

#### H100 nodes

The scripts for the H100 nodes are:
- `batch_vllm_run_h100.sh`
- `vllm_run_h100.sh`

These scripts use a container defined by the `container/container_vllm.def` file.
You will need to build the container yourself, to do this:
- Use the following command to launch an interactive job on an H100 node (replace `xxxx` with your project code): 
```bash 
srun --constraint h100_80 --qos turing --account xxxx --time 2:00:0 --nodes 1 --gpus-per-node 1 --cpus-per-gpu 36 --mem 0 --pty /bin/bash
```
- Build the container with the following command:
``` bash
apptainer build container_vllm.sif container/container_vllm.def
``` 

Then, to run the script, use:

``` bash
sbatch batch_vllm_run_h100.sh
```

### Isambard-AI

The scripts for Isambard-AI are:
- `batch_vllm_run_gh.sh`
- `batch_vllm_run_gh_1node.sh`
- `vllm_run_gh.sh`

There are als some scripts for running [Llama-3.3-70B-Instruct-FP8](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8) in `llama3.3` folder: 
- `batch_vllm_llama.sh`
- `vllm_llama.sh`

And, for running vLLM using python (rather than command line) in the `vllm_python` folder:
- `batch_vllm_python.sh`
- `vllm_python.sh`
- `run_vllm.py`

All the above scripts use the `e4s-cuda90-aarch64-25.06.4.sif` container.
This is a pre-built container image which is accessible on the Isambard-AI filesystem at `/projects/public/brics/containers/e4s/cuda90-aarch64-25.06.4.sif`.

You will need to copy or symlink this image into to a `container` directory before running the scripts. 

To do this, create the `container` directory if it does not already exist using `mkdir container`, then run the following command:

``` bash
ln -s /projects/public/brics/containers/e4s/cuda90-aarch64-25.06.4.sif container/e4s-cuda90-aarch64-25.06.4.sif
```

Then, to run the scripts, use:

``` bash
sbatch batch_vllm_run_gh.sh
```
or, for a single node:

``` bash
sbatch batch_vllm_run_gh_1node.sh
```

Likewise, for the `llama3.3` and `vllm_python` examples.

### Hosting a frontend

The repository also contains instructions and a small script for running a frontend using Open WebUI to interact with your model after serving with vLLM.
The are in the `frontend` directory.

Go to the `vllm_examples/frontend/README.md` for more information.

## transformers scripts

Scripts for running single-node inference with the transformers library are in the `transformers_examples` directory.

The python scripts are:
1. `qwen.py` - This script defines a prompt, asks the model to generate a response and then prints the output.
2. `qwen_chat.py` - This script defines a simple chat interface where the user can input prompts and receive responses from the model in a loop.

The two options for running these scripts are via an interactive job (both for `qwen_chat.py` and `qwen.py`) or a batch job (only for `qwen.py`).

### Baskerville

To launch an interactive job, run the following command (replace `xxxx` with your project code):

```bash
srun --qos turing --account xxxx --time 0:30:0 --nodes 1 --gpus-per-node 4 --cpus-per-gpu 36 --mem 0 --pty /bin/bash
```

Once your job is allocated resources, you can run the `run_local.sh` script to run `qwen.py` or the `run_chat_local.py` script to run `qwen_chat.py` and chat with the model.

Alternatively, to run the `qwen.py` script as a batch job, you can use the `batch_run_local.sh` script:

```bash
sbatch batch_run_local.sh
```

Once the job is running, you can see the outputs in the `one_node.log` file.

### Isambard-AI

On Isambard-AI, you will need to use the container defined by the `container/container_vllm.def` file to run the `qwen.py` and `qwen_chat.py` scripts.
You will need to build the container yourself, to do this:
- Use the following command to launch an interactive job:
```bash 
srun --time=0:30:0 --nodes=1 --gpus-per-node 1 --cpus-per-node 72 --mem=0 --pty /bin/bash
```
- Build the container with the following command:
``` bash
apptainer build container.sif container/container.def
``` 

Once the container is built, you can run the `run_apptainer.sh` script to start the it.
This will drop you into a shell inside the container from which you can run the `qwen.py` or `qwen_chat.py` scripts.

Before running either script, you will need to set the `HF_HOME` environment variable to `/hf_home` and `cd` to the `transformers_examples` directory:
```bash
export $HF_HOME=/hf_home
cd /transformers_examples
```

From there, you can run the chat script with:
```bash
python qwen_chat.py
```

(or `python qwen.py` to run the other script).

Alternatively, to use a batch job you can run the `batch_run_apptainer.sh` script:

```bash
sbatch batch_run_apptainer.sh
```

Once the job is running, you can see the outputs in the `one_node_gh.log` file.

## Refs

- https://docs.vllm.ai/en/v0.9.2/serving/distributed_serving.html
- https://github.com/vllm-project/vllm/blob/main/examples/online_serving/run_cluster.sh
- [vllm #26028](https://github.com/vllm-project/vllm/issues/26028)
- https://swaglu.com/llama-405b-vllm-slurm-multinode/#step-1-multi-node-slurm-configuration
- https://docs.isambard.ac.uk/user-documentation/guides/containers/apptainer-multi-node/
