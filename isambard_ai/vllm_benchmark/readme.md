# vLLM Benchmarking

The following scripts and configurations are set up to benchmark the vLLM model using Ray on a multi-node cluster on Isambard AI.

## Prerequisites

- Modify `vllm_bench.sh` file to set the following environment variables according to your setup. I have tested with the following

    ```bash
    export HF_HOME="/lus/lfs1aip2/home/u5bi/tomas.u5bi/.cache"

    export SCRATCH_DIR="/scratch/u5bi/tomas.u5bi"
    ```

- Download the Singularty container using the `download_container.sh` script or copy it / link it to one that already exists on your system.

## Running the Benchmark

```bash
sbatch vllm_bench.sh
```

## Output

The benchmark results will be logged in a file named `vllm_bench.log` in the submission directory and should look something like this:

<details>
<summary>Show content</summary>

```text
--------------------------------------
Job ID: 1201661
Allocated Nodes: nid[010010-010011]
Primary Host: nid010010
Primary IP (hsn0): 10.242.22.60
Ray GCS Port: 25629
Discovered Node IPs: 10.242.22.60 10.242.21.180 
--------------------------------------
0: Host: nid010010, Assigned IP: 10.242.22.60
0: Starting Ray head node...
1: Host: nid010011, Assigned IP: 10.242.21.180
0: 2025-10-08 13:10:47,748	INFO usage_lib.py:447 -- Usage stats collection is disabled.
0: 2025-10-08 13:10:47,748	INFO scripts.py:913 -- Local node IP: 10.242.22.60
0: 2025-10-08 13:10:49,666	SUCC scripts.py:949 -- --------------------
0: 2025-10-08 13:10:49,667	SUCC scripts.py:950 -- Ray runtime started.
0: 2025-10-08 13:10:49,667	SUCC scripts.py:951 -- --------------------
0: 2025-10-08 13:10:49,667	INFO scripts.py:953 -- Next steps
0: 2025-10-08 13:10:49,667	INFO scripts.py:956 -- To add another node to this Ray cluster, run
0: 2025-10-08 13:10:49,667	INFO scripts.py:959 --   ray start --address='10.242.22.60:25629'
0: 2025-10-08 13:10:49,667	INFO scripts.py:968 -- To connect to this Ray cluster:
0: 2025-10-08 13:10:49,667	INFO scripts.py:970 -- import ray
0: 2025-10-08 13:10:49,667	INFO scripts.py:971 -- ray.init(_node_ip_address='10.242.22.60')
0: 2025-10-08 13:10:49,667	INFO scripts.py:1002 -- To terminate the Ray runtime, run
0: 2025-10-08 13:10:49,667	INFO scripts.py:1003 --   ray stop
0: 2025-10-08 13:10:49,667	INFO scripts.py:1006 -- To view the status of the cluster, use
0: 2025-10-08 13:10:49,667	INFO scripts.py:1007 --   ray status
0: Waiting for worker nodes to register...
1: Starting Ray worker node, connecting to 10.242.22.60:25629...
1: 2025-10-08 13:10:57,952	INFO scripts.py:1094 -- Local node IP: 10.242.21.180
1: 2025-10-08 13:10:58,699	SUCC scripts.py:1110 -- --------------------
1: 2025-10-08 13:10:58,699	SUCC scripts.py:1111 -- Ray runtime started.
1: 2025-10-08 13:10:58,699	SUCC scripts.py:1112 -- --------------------
1: 2025-10-08 13:10:58,699	INFO scripts.py:1114 -- To terminate the Ray runtime, run
1: 2025-10-08 13:10:58,699	INFO scripts.py:1115 --   ray stop
1: Worker node is ready and awaiting job completion signal...
0: ======== Autoscaler status: 2025-10-08 13:11:13.718539 ========
0: Node status
0: ---------------------------------------------------------------
0: Active:
0:  1 node_22a5c9e00dd6fb206f7b4deec7127287d655104f4b91918f3a3bdca9
0:  1 node_58828d4fb0f1b09feb79abdf2c769c0f96c9da64219f6c465f3a4d5a
0: Pending:
0:  (no pending nodes)
0: Recent failures:
0:  (no failures)
0: 
0: Resources
0: ---------------------------------------------------------------
0: Total Usage:
0:  0.0/576.0 CPU
0:  0.0/8.0 GPU
0:  0B/1.19TiB memory
0:  0B/372.53GiB object_store_memory
0: 
0: Total Constraints:
0:  (no request_resources() constraints)
0: Total Demands:
0:  (no resource demands)
0: Running vLLM throughput benchmark...
0: INFO 10-08 13:11:22 [__init__.py:241] Automatically detected platform cuda.
0: When dataset path is not set, it will default to random dataset
0: INFO 10-08 13:11:26 [datasets.py:358] Sampling input_len from [512, 512] and output_len from [1024, 1024]
0: INFO 10-08 13:11:28 [utils.py:326] non-default args: {'model': 'Qwen/Qwen3-30B-A3B-Instruct-2507', 'tokenizer': 'Qwen/Qwen3-30B-A3B-Instruct-2507', 'download_dir': '/hf_cache_dir', 'seed': 0, 'distributed_executor_backend': 'ray', 'pipeline_parallel_size': 2, 'tensor_parallel_size': 4, 'enable_lora': None}
0: INFO 10-08 13:11:37 [__init__.py:711] Resolved architecture: Qwen3MoeForCausalLM
0: `torch_dtype` is deprecated! Use `dtype` instead!
0: INFO 10-08 13:11:37 [__init__.py:1750] Using max model len 262144
0: INFO 10-08 13:11:37 [scheduler.py:222] Chunked prefill is enabled with max_num_batched_tokens=16384.
0: INFO 10-08 13:11:43 [__init__.py:241] Automatically detected platform cuda.
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:11:46 [core.py:636] Waiting for init message from front-end.
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:11:46 [core.py:74] Initializing a V1 LLM engine (v0.10.1.2.dev0+g1da94e673.d20250903) with config: model='Qwen/Qwen3-30B-A3B-Instruct-2507', speculative_config=None, tokenizer='Qwen/Qwen3-30B-A3B-Instruct-2507', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=262144, download_dir='/hf_cache_dir', load_format=auto, tensor_parallel_size=4, pipeline_parallel_size=2, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507, enable_
0: prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=False, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":1,"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"pass_config":{},"max_capture_size":512,"local_cache_dir":null}
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:11:46,906	INFO worker.py:1630 -- Using address 10.242.22.60:25629 set in the environment variable RAY_ADDRESS
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:11:46,918	INFO worker.py:1771 -- Connecting to existing Ray cluster at address: 10.242.22.60:25629...
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:11:46,924	INFO worker.py:1951 -- Connected to Ray cluster.
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:11:50 [ray_utils.py:339] No current placement group found. Creating a new placement group.
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:11:50 [ray_distributed_executor.py:169] use_ray_spmd_worker: True
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(pid=132250)[0m INFO 10-08 13:11:56 [__init__.py:241] Automatically detected platform cuda.
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:01 [ray_env.py:63] RAY_NON_CARRY_OVER_ENV_VARS from config: set()
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:01 [ray_env.py:65] Copying the following environment variables to workers: ['VLLM_USE_V1', 'VLLM_USE_RAY_SPMD_WORKER', 'LD_LIBRARY_PATH', 'VLLM_USE_RAY_COMPILED_DAG', 'VLLM_WORKER_MULTIPROC_METHOD']
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:01 [ray_env.py:68] If certain env vars should NOT be copied, add them to /home/u5bi/tomas.u5bi/.config/vllm/ray_non_carry_over_env_vars.json file
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Loading safetensors checkpoint shards:   0% Completed | 0/16 [00:00<?, ?it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Loading safetensors checkpoint shards:   6% Completed | 1/16 [00:00<00:06,  2.26it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Loading safetensors checkpoint shards:  12% Completed | 2/16 [00:00<00:06,  2.16it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Loading safetensors checkpoint shards:  31% Completed | 5/16 [00:01<00:01,  5.99it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Loading safetensors checkpoint shards:  44% Completed | 7/16 [00:01<00:02,  4.30it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Loading safetensors checkpoint shards:  50% Completed | 8/16 [00:02<00:02,  3.16it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Loading safetensors checkpoint shards:  56% Completed | 9/16 [00:02<00:02,  2.71it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Loading safetensors checkpoint shards:  69% Completed | 11/16 [00:03<00:01,  4.03it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Loading safetensors checkpoint shards:  81% Completed | 13/16 [00:03<00:00,  3.75it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Loading safetensors checkpoint shards:  88% Completed | 14/16 [00:04<00:00,  3.11it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Loading safetensors checkpoint shards:  94% Completed | 15/16 [00:04<00:00,  2.62it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Loading safetensors checkpoint shards: 100% Completed | 16/16 [00:04<00:00,  3.33it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206151, ip=10.242.21.180)[0m INFO 10-08 13:12:11 [__init__.py:1418] Found nccl from library libnccl.so.2
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206151, ip=10.242.21.180)[0m INFO 10-08 13:12:11 [pynccl.py:70] vLLM is using nccl==2.26.2
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(pid=206149, ip=10.242.21.180)[0m INFO 10-08 13:11:59 [__init__.py:241] Automatically detected platform cuda.[32m [repeated 7x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132259)[0m INFO 10-08 13:12:14 [custom_all_reduce.py:35] Skipping P2P check and trusting the driver's P2P report.
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m INFO 10-08 13:12:14 [shm_broadcast.py:289] vLLM message queue communication handle: Handle(local_reader_ranks=[1, 2, 3], buffer_handle=(3, 4194304, 6, 'psm_c50fe601'), local_subscribe_addr='ipc:///local/user/1483801525/53d88caf-4db9-43c4-832a-f876359ed2e8', remote_subscribe_addr=None, remote_addr_ipv6=False)
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132266)[0m INFO 10-08 13:12:14 [parallel_state.py:1134] rank 1 in world size 8 is assigned as DP rank 0, PP rank 0, TP rank 1, EP rank 1
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132266)[0m WARNING 10-08 13:12:14 [topk_topp_sampler.py:61] FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132266)[0m INFO 10-08 13:12:14 [gpu_model_runner.py:1953] Starting to load model Qwen/Qwen3-30B-A3B-Instruct-2507...
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206149, ip=10.242.21.180)[0m INFO 10-08 13:12:15 [gpu_model_runner.py:1985] Loading model from scratch...
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206149, ip=10.242.21.180)[0m INFO 10-08 13:12:15 [cuda.py:328] Using Flash Attention backend on V1 engine.
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m INFO 10-08 13:12:15 [weight_utils.py:296] Using model weights format ['*.safetensors']
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m INFO 10-08 13:12:20 [default_loader.py:262] Loading weights took 4.81 seconds
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206149, ip=10.242.21.180)[0m INFO 10-08 13:12:14 [__init__.py:1418] Found nccl from library libnccl.so.2[32m [repeated 15x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206149, ip=10.242.21.180)[0m INFO 10-08 13:12:14 [pynccl.py:70] vLLM is using nccl==2.26.2[32m [repeated 15x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206148, ip=10.242.21.180)[0m INFO 10-08 13:12:14 [custom_all_reduce.py:35] Skipping P2P check and trusting the driver's P2P report.[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206148, ip=10.242.21.180)[0m INFO 10-08 13:12:14 [shm_broadcast.py:289] vLLM message queue communication handle: Handle(local_reader_ranks=[1, 2, 3], buffer_handle=(3, 4194304, 6, 'psm_9c3a0862'), local_subscribe_addr='ipc:///local/user/1483801525/54784ac8-8bf8-4647-b194-d417ae3ec35c', remote_subscribe_addr=None, remote_addr_ipv6=False)
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206149, ip=10.242.21.180)[0m INFO 10-08 13:12:14 [parallel_state.py:1134] rank 7 in world size 8 is assigned as DP rank 0, PP rank 1, TP rank 3, EP rank 3[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206149, ip=10.242.21.180)[0m WARNING 10-08 13:12:14 [topk_topp_sampler.py:61] FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206149, ip=10.242.21.180)[0m INFO 10-08 13:12:14 [gpu_model_runner.py:1953] Starting to load model Qwen/Qwen3-30B-A3B-Instruct-2507...[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132259)[0m INFO 10-08 13:12:15 [gpu_model_runner.py:1985] Loading model from scratch...[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132265)[0m INFO 10-08 13:12:15 [cuda.py:328] Using Flash Attention backend on V1 engine.[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132265)[0m INFO 10-08 13:12:15 [weight_utils.py:296] Using model weights format ['*.safetensors'][32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m INFO 10-08 13:12:20 [gpu_model_runner.py:2007] Model loading took 7.3272 GiB and 5.488410 seconds
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206151, ip=10.242.21.180)[0m INFO 10-08 13:12:25 [default_loader.py:262] Loading weights took 8.49 seconds[32m [repeated 4x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206148, ip=10.242.21.180)[0m INFO 10-08 13:12:26 [gpu_model_runner.py:2007] Model loading took 7.3272 GiB and 10.485893 seconds[32m [repeated 4x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132259)[0m INFO 10-08 13:12:31 [backends.py:548] Using cache directory: /home/u5bi/tomas.u5bi/.cache/vllm/torch_compile_cache/7c5b91ce58/rank_3_0/backbone for vLLM's torch.compile
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132259)[0m INFO 10-08 13:12:31 [backends.py:559] Dynamo bytecode transform time: 4.86 s
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206150, ip=10.242.21.180)[0m INFO 10-08 13:12:25 [default_loader.py:262] Loading weights took 9.08 seconds[32m [repeated 3x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206150, ip=10.242.21.180)[0m INFO 10-08 13:12:26 [gpu_model_runner.py:2007] Model loading took 7.3272 GiB and 10.515026 seconds[32m [repeated 3x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132259)[0m INFO 10-08 13:12:38 [backends.py:161] Directly load the compiled graph(s) for dynamic shape from the cache, took 6.391 s
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206149, ip=10.242.21.180)[0m INFO 10-08 13:12:31 [backends.py:548] Using cache directory: /home/u5bi/tomas.u5bi/.cache/vllm/torch_compile_cache/325b8303af/rank_7_0/backbone for vLLM's torch.compile[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206149, ip=10.242.21.180)[0m INFO 10-08 13:12:31 [backends.py:559] Dynamo bytecode transform time: 5.10 s[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206151, ip=10.242.21.180)[0m WARNING 10-08 13:12:39 [fused_moe.py:727] Using default MoE config. Performance might be sub-optimal! Config file not found at ['/py3.10-vllm/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs/E=128,N=192,device_name=NVIDIA_GH200_120GB.json']
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206151, ip=10.242.21.180)[0m INFO 10-08 13:12:40 [monitor.py:34] torch.compile takes 5.07 s in total
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m INFO 10-08 13:12:40 [gpu_worker.py:276] Available KV cache memory: 76.15 GiB
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132266)[0m INFO 10-08 13:12:39 [backends.py:161] Directly load the compiled graph(s) for dynamic shape from the cache, took 7.154 s[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:849] GPU KV cache size: 6,654,432 tokens
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:853] Maximum concurrency for 262,144 tokens per request: 25.38x
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:849] GPU KV cache size: 6,654,432 tokens
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:853] Maximum concurrency for 262,144 tokens per request: 25.38x
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:849] GPU KV cache size: 6,654,432 tokens
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:853] Maximum concurrency for 262,144 tokens per request: 25.38x
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:849] GPU KV cache size: 6,654,432 tokens
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:853] Maximum concurrency for 262,144 tokens per request: 25.38x
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:849] GPU KV cache size: 6,167,568 tokens
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:853] Maximum concurrency for 262,144 tokens per request: 23.53x
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:849] GPU KV cache size: 6,167,536 tokens
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:853] Maximum concurrency for 262,144 tokens per request: 23.53x
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:849] GPU KV cache size: 6,167,536 tokens
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:853] Maximum concurrency for 262,144 tokens per request: 23.53x
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:849] GPU KV cache size: 6,167,536 tokens
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:44 [kv_cache_utils.py:853] Maximum concurrency for 262,144 tokens per request: 23.53x
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   0%|          | 0/67 [00:00<?, ?it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   3%|▎         | 2/67 [00:00<00:04, 15.71it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   7%|▋         | 5/67 [00:00<00:03, 19.10it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  12%|█▏        | 8/67 [00:00<00:02, 20.27it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  16%|█▋        | 11/67 [00:00<00:02, 20.61it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  21%|██        | 14/67 [00:00<00:02, 20.69it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  25%|██▌       | 17/67 [00:00<00:02, 20.67it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  30%|██▉       | 20/67 [00:00<00:02, 20.72it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  34%|███▍      | 23/67 [00:01<00:02, 20.58it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  39%|███▉      | 26/67 [00:01<00:01, 20.54it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  43%|████▎     | 29/67 [00:01<00:01, 20.45it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  48%|████▊     | 32/67 [00:01<00:01, 20.48it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  52%|█████▏    | 35/67 [00:01<00:01, 20.12it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  57%|█████▋    | 38/67 [00:01<00:01, 20.01it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  61%|██████    | 41/67 [00:02<00:01, 19.89it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  66%|██████▌   | 44/67 [00:02<00:01, 19.95it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  70%|███████   | 47/67 [00:02<00:01, 19.95it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  73%|███████▎  | 49/67 [00:02<00:00, 18.20it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  76%|███████▌  | 51/67 [00:02<00:00, 18.43it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  79%|███████▉  | 53/67 [00:02<00:00, 18.40it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  84%|████████▎ | 56/67 [00:02<00:00, 18.94it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  88%|████████▊ | 59/67 [00:02<00:00, 19.40it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  91%|█████████ | 61/67 [00:03<00:00, 19.53it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  94%|█████████▍| 63/67 [00:03<00:00, 19.52it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  99%|█████████▊| 66/67 [00:03<00:00, 19.83it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132250)[0m 
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████| 67/67 [00:03<00:00, 19.58it/s]
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206151, ip=10.242.21.180)[0m INFO 10-08 13:12:48 [custom_all_reduce.py:196] Registering 3216 cuda graph addresses
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132265)[0m WARNING 10-08 13:12:40 [fused_moe.py:727] Using default MoE config. Performance might be sub-optimal! Config file not found at ['/py3.10-vllm/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs/E=128,N=192,device_name=NVIDIA_GH200_120GB.json'][32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132265)[0m INFO 10-08 13:12:40 [monitor.py:34] torch.compile takes 5.12 s in total[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206149, ip=10.242.21.180)[0m INFO 10-08 13:12:43 [gpu_worker.py:276] Available KV cache memory: 70.58 GiB[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206151, ip=10.242.21.180)[0m INFO 10-08 13:12:48 [gpu_model_runner.py:2708] Graph capturing finished in 4 secs, took 0.63 GiB
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:48 [core.py:214] init engine (profile, create kv cache, warmup model) took 22.41 seconds
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:50 [core.py:141] Batch queue is enabled with size 2
0: INFO 10-08 13:12:58 [llm.py:298] Supported_tasks: ['generate']
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:58 [ray_distributed_executor.py:550] RAY_CGRAPH_get_timeout is set to 300
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:58 [ray_distributed_executor.py:552] VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE = auto
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:58 [ray_distributed_executor.py:554] VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM = False
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:12:58 [ray_distributed_executor.py:619] Using RayPPCommunicator (which wraps vLLM _PP GroupCoordinator) for Ray Compiled Graph communication.
0: 
Adding requests:   0%|          | 0/1000 [00:00<?, ?it/s]
Adding requests:   6%|▌         | 61/1000 [00:00<00:01, 606.00it/s]
Adding requests:  13%|█▎        | 127/1000 [00:00<00:01, 632.17it/s][1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:12:58,810	INFO torch_tensor_accelerator_channel.py:807 -- Creating communicator group 74a6e4e3-425c-4676-a1d2-01650c8a3373 on actors: [Actor(RayWorkerWrapper, 288448082d15843e01ac1f0301000000), Actor(RayWorkerWrapper, cb860d9c4690ae87519dc02b01000000), Actor(RayWorkerWrapper, 380dfb36f122a9d64c1ef27a01000000), Actor(RayWorkerWrapper, f744a3406dbe047e5258fd2b01000000), Actor(RayWorkerWrapper, 7467d3518cb625b967e38ca401000000), Actor(RayWorkerWrapper, 92caaf72ff470c9fe78aa15401000000), Actor(RayWorkerWrapper, d9f9ff71d406ff3bb721afdb01000000), Actor(RayWorkerWrapper, 79a75155b04f13434b4db8c801000000)]
0: 
Adding requests:  20%|█▉        | 197/1000 [00:00<00:01, 660.19it/s][1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206151, ip=10.242.21.180)[0m /py3.10-vllm/lib/python3.10/site-packages/vllm/distributed/device_communicators/ray_communicator.py:107: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1577.)
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=206151, ip=10.242.21.180)[0m   actor_id_tensor = torch.frombuffer(
0: 
Adding requests:  26%|██▋       | 264/1000 [00:00<00:01, 657.74it/s]
Adding requests:  33%|███▎      | 330/1000 [00:00<00:01, 644.53it/s]
Adding requests:  40%|███▉      | 396/1000 [00:00<00:00, 649.54it/s]
Adding requests:  47%|████▋     | 468/1000 [00:00<00:00, 670.27it/s]
Adding requests:  54%|█████▍    | 538/1000 [00:00<00:00, 678.76it/s]
Adding requests:  61%|██████    | 610/1000 [00:00<00:00, 690.72it/s]
Adding requests:  68%|██████▊   | 685/1000 [00:01<00:00, 706.32it/s]
Adding requests:  76%|███████▌  | 760/1000 [00:01<00:00, 717.33it/s]
Adding requests:  84%|████████▎ | 835/1000 [00:01<00:00, 725.16it/s]
Adding requests:  91%|█████████ | 908/1000 [00:01<00:00, 721.60it/s]
Adding requests:  98%|█████████▊| 983/1000 [00:01<00:00, 726.91it/s]
Adding requests: 100%|██████████| 1000/1000 [00:01<00:00, 692.94it/s]
0: 
Processed prompts:   0%|          | 0/1000 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s][1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:13:00,903	INFO torch_tensor_accelerator_channel.py:833 -- Communicator group initialized.
0: 
Processed prompts:   0%|          | 1/1000 [01:27<24:20:24, 87.71s/it, est. speed input: 5.84 toks/s, output: 11.67 toks/s]
Processed prompts:   3%|▎         | 34/1000 [01:27<29:23,  1.83s/it, est. speed input: 198.17 toks/s, output: 396.38 toks/s]
Processed prompts:  10%|▉         | 97/1000 [01:27<07:30,  2.01it/s, est. speed input: 564.56 toks/s, output: 1129.40 toks/s]
Processed prompts:  16%|█▌        | 161/1000 [01:28<03:24,  4.11it/s, est. speed input: 935.59 toks/s, output: 1872.27 toks/s]
Processed prompts:  22%|██▎       | 225/1000 [01:28<01:49,  7.09it/s, est. speed input: 1303.70 toks/s, output: 2613.52 toks/s]
Processed prompts:  32%|███▏      | 321/1000 [01:28<00:50, 13.40it/s, est. speed input: 1857.53 toks/s, output: 3723.00 toks/s]
Processed prompts:  42%|████▏     | 417/1000 [01:28<00:26, 22.23it/s, est. speed input: 2408.57 toks/s, output: 4829.54 toks/s]
Processed prompts:  51%|█████▏    | 513/1000 [01:28<00:14, 34.61it/s, est. speed input: 2957.
0: 86 toks/s, output: 5934.50 toks/s]
Processed prompts:  64%|██████▍   | 640/1000 [01:28<00:06, 57.31it/s, est. speed input: 3684.62 toks/s, output: 7393.97 toks/s]
Processed prompts:  80%|███████▉  | 798/1000 [01:28<00:02, 96.29it/s, est. speed input: 4589.88 toks/s, output: 9208.36 toks/s]
Processed prompts: 100%|██████████| 1000/1000 [01:28<00:00, 96.29it/s, est. speed input: 5749.96 toks/s, output: 11529.46 toks/s]
Processed prompts: 100%|██████████| 1000/1000 [01:28<00:00, 11.26it/s, est. speed input: 5749.96 toks/s, output: 11529.46 toks/s]
0: *** SIGTERM received at time=1759954468 on cpu 5 ***
0: PC: @     0x4002f3c31ea0  (unknown)  (unknown)
0:     @     0x4004ebf56848        464  absl::lts_20230802::AbslFailureSignalHandler()
0:     @     0x4002f3a507c0  172579056  (unknown)
0:     @     0x4002f3c3e2f0        112  (unknown)
0:     @           0x481f7c         48  PyThread_acquire_lock_timed
0:     @   0x13000000584dbc         80  (unknown)
0:     @   0x54000000584b74         64  (unknown)
0:     @   0x390000004b5000         64  (unknown)
0:     @   0x7400000049e374        256  (unknown)
0:     @   0x310000004b3884        384  (unknown)
0:     @   0x1400000049e374         48  (unknown)
0:     @   0x1e0000004b3884        384  (unknown)
0:     @   0x4100000049e374         48  (unknown)
0:     @   0x290000004b3884        384  (unknown)
0:     @   0x5800000049e374         48  (unknown)
0:     @   0x270000004b3884        384  (unknown)
0:     @   0x7800000049e374         48  (unknown)
0:     @   0x160000004b3884        384  (unknown)
0:     @   0x1c0000004c2a18         48  (unknown)
0:     @   0x4d0000004a044c        112  (unknown)
0:     @   0x5b0000004b3884        384  (unknown)
0:     @   0x7500000049e374         48  (unknown)
0:     @   0x3c0000004b3884        384  (unknown)
0:     @   0x7e00000049e374         48  (unknown)
0:     @   0x1a0000004b3884        384  (unknown)
0:     @   0x6c00000049e000         48  (unknown)
0:     @   0x1b0000004b3884        384  (unknown)
0:     @   0x1f00000049f130         48  (unknown)
0:     @   0x3300000057b274        384  (unknown)
0:     @   0x7700000057b174         48  (unknown)
0:     @   0x790000005c478c        112  (unknown)
0:     @   0x490000005bddb4         64  (unknown)
0:     @   0x690000005b0c0c         48  (unknown)
0:     @ ... and at least 5 more frames
0: [2025-10-08 13:14:28,827 E 132079 132079] logging.cc:474: *** SIGTERM received at time=1759954468 on cpu 5 ***
0: [2025-10-08 13:14:28,828 E 132079 132079] logging.cc:474: PC: @     0x4002f3c31ea0  (unknown)  (unknown)
0: [2025-10-08 13:14:28,831 E 132079 132079] logging.cc:474:     @     0x4004ebf56870        464  absl::lts_20230802::AbslFailureSignalHandler()
0: [2025-10-08 13:14:28,831 E 132079 132079] logging.cc:474:     @     0x4002f3a507c0  172579056  (unknown)
0: [2025-10-08 13:14:28,832 E 132079 132079] logging.cc:474:     @     0x4002f3c3e2f0        112  (unknown)
0: [2025-10-08 13:14:28,832 E 132079 132079] logging.cc:474:     @           0x481f7c         48  PyThread_acquire_lock_timed
0: [2025-10-08 13:14:28,833 E 132079 132079] logging.cc:474:     @   0x13000000584dbc         80  (unknown)
0: [2025-10-08 13:14:28,835 E 132079 132079] logging.cc:474:     @   0x54000000584b74         64  (unknown)
0: [2025-10-08 13:14:28,836 E 132079 132079] logging.cc:474:     @   0x390000004b5000         64  (unknown)
0: [2025-10-08 13:14:28,838 E 132079 132079] logging.cc:474:     @   0x7400000049e374        256  (unknown)
0: [2025-10-08 13:14:28,839 E 132079 132079] logging.cc:474:     @   0x310000004b3884        384  (unknown)
0: [2025-10-08 13:14:28,841 E 132079 132079] logging.cc:474:     @   0x1400000049e374         48  (unknown)
0: [2025-10-08 13:14:28,842 E 132079 132079] logging.cc:474:     @   0x1e0000004b3884        384  (unknown)
0: [2025-10-08 13:14:28,844 E 132079 132079] logging.cc:474:     @   0x4100000049e374         48  (unknown)
0: [2025-10-08 13:14:28,845 E 132079 132079] logging.cc:474:     @   0x290000004b3884        384  (unknown)
0: [2025-10-08 13:14:28,847 E 132079 132079] logging.cc:474:     @   0x5800000049e374         48  (unknown)
0: [2025-10-08 13:14:28,849 E 132079 132079] logging.cc:474:     @   0x270000004b3884        384  (unknown)
0: [2025-10-08 13:14:28,850 E 132079 132079] logging.cc:474:     @   0x7800000049e374         48  (unknown)
0: [2025-10-08 13:14:28,852 E 132079 132079] logging.cc:474:     @   0x160000004b3884        384  (unknown)
0: [2025-10-08 13:14:28,853 E 132079 132079] logging.cc:474:     @   0x1c0000004c2a18         48  (unknown)
0: [2025-10-08 13:14:28,855 E 132079 132079] logging.cc:474:     @   0x4d0000004a044c        112  (unknown)
0: [2025-10-08 13:14:28,856 E 132079 132079] logging.cc:474:     @   0x5b0000004b3884        384  (unknown)
0: [2025-10-08 13:14:28,858 E 132079 132079] logging.cc:474:     @   0x7500000049e374         48  (unknown)
0: [2025-10-08 13:14:28,859 E 132079 132079] logging.cc:474:     @   0x3c0000004b3884        384  (unknown)
0: [2025-10-08 13:14:28,861 E 132079 132079] logging.cc:474:     @   0x7e00000049e374         48  (unknown)
0: [2025-10-08 13:14:28,862 E 132079 132079] logging.cc:474:     @   0x1a0000004b3884        384  (unknown)
0: [2025-10-08 13:14:28,864 E 132079 132079] logging.cc:474:     @   0x6c00000049e000         48  (unknown)
0: [2025-10-08 13:14:28,865 E 132079 132079] logging.cc:474:     @   0x1b0000004b3884        384  (unknown)
0: [2025-10-08 13:14:28,867 E 132079 132079] logging.cc:474:     @   0x1f00000049f130         48  (unknown)
0: [2025-10-08 13:14:28,869 E 132079 132079] logging.cc:474:     @   0x3300000057b274        384  (unknown)
0: [2025-10-08 13:14:28,870 E 132079 132079] logging.cc:474:     @   0x7700000057b174         48  (unknown)
0: [2025-10-08 13:14:28,872 E 132079 132079] logging.cc:474:     @   0x790000005c478c        112  (unknown)
0: [2025-10-08 13:14:28,873 E 132079 132079] logging.cc:474:     @   0x490000005bddb4         64  (unknown)
0: [2025-10-08 13:14:28,875 E 132079 132079] logging.cc:474:     @   0x690000005b0c0c         48  (unknown)
0: [2025-10-08 13:14:28,875 E 132079 132079] logging.cc:474:     @ ... and at least 5 more frames
0: [1;36m(EngineCore_0 pid=132079)[0;0m INFO 10-08 13:14:28 [ray_distributed_executor.py:120] Shutting down Ray distributed executor. If you see error log from logging.cc regarding SIGTERM received, please ignore because this is the expected termination process in Ray.
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:14:28,875	INFO compiled_dag_node.py:2171 -- Tearing down compiled DAG
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:14:28,875	INFO compiled_dag_node.py:2176 -- Cancelling compiled worker on actor: Actor(RayWorkerWrapper, 288448082d15843e01ac1f0301000000)
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:14:28,875	INFO compiled_dag_node.py:2176 -- Cancelling compiled worker on actor: Actor(RayWorkerWrapper, 380dfb36f122a9d64c1ef27a01000000)
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:14:28,875	INFO compiled_dag_node.py:2176 -- Cancelling compiled worker on actor: Actor(RayWorkerWrapper, f744a3406dbe047e5258fd2b01000000)
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:14:28,875	INFO compiled_dag_node.py:2176 -- Cancelling compiled worker on actor: Actor(RayWorkerWrapper, 79a75155b04f13434b4db8c801000000)
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:14:28,875	INFO compiled_dag_node.py:2176 -- Cancelling compiled worker on actor: Actor(RayWorkerWrapper, 92caaf72ff470c9fe78aa15401000000)
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:14:28,875	INFO compiled_dag_node.py:2176 -- Cancelling compiled worker on actor: Actor(RayWorkerWrapper, cb860d9c4690ae87519dc02b01000000)
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:14:28,875	INFO compiled_dag_node.py:2176 -- Cancelling compiled worker on actor: Actor(RayWorkerWrapper, 7467d3518cb625b967e38ca401000000)
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:14:28,875	INFO compiled_dag_node.py:2176 -- Cancelling compiled worker on actor: Actor(RayWorkerWrapper, d9f9ff71d406ff3bb721afdb01000000)
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:14:28,882	INFO compiled_dag_node.py:2198 -- Waiting for worker tasks to exit
0: [1;36m(EngineCore_0 pid=132079)[0;0m 2025-10-08 13:14:28,883	INFO compiled_dag_node.py:2201 -- Teardown complete
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132265)[0m /py3.10-vllm/lib/python3.10/site-packages/vllm/distributed/device_communicators/ray_communicator.py:107: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1577.)[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132265)[0m   actor_id_tensor = torch.frombuffer([32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132266)[0m INFO 10-08 13:12:48 [custom_all_reduce.py:196] Registering 3283 cuda graph addresses[32m [repeated 7x across cluster][0m
0: [1;36m(EngineCore_0 pid=132079)[0;0m [36m(RayWorkerWrapper pid=132265)[0m INFO 10-08 13:12:48 [gpu_model_runner.py:2708] Graph capturing finished in 4 secs, took 0.70 GiB[32m [repeated 7x across cluster][0m
0: Throughput: 11.08 requests/s, 17002.75 total tokens/s, 11344.86 output tokens/s
0: Total num prompt tokens:  510688
0: Total num output tokens:  1024000
0: Benchmark finished. Signaling worker nodes to exit.
0: Node task finished.
0: INFO:    Terminating squashfuse_ll after timeout
0: INFO:    Timeouts can be caused by a running background process
1: Completion signal received. Worker node exiting.
1: Node task finished.
1: INFO:    Terminating squashfuse_ll after timeout
1: INFO:    Timeouts can be caused by a running background process
Job completed successfully.
--- Performing cleanup ---
Cleanup complete.
```

</details>


The SIGTERM received in the logs is expected as part of Ray's termination process and can be safely ignored.