import os
import json
import subprocess
import time
import ray

from vllm import LLM
from zeus.monitor import ZeusMonitor

# Get Slurm job information from environment variables
RANK = int(os.environ["SLURM_PROCID"])
NNODES = int(os.environ["SLURM_NNODES"])    

def main():
    """
    Initializes a Ray cluster and runs the vLLM engine on the head node.
    """
    ray.init(address="auto")

    # --- Synchronization Point ---
    # Wait until all expected nodes have joined the Ray cluster.
    print(f"Rank {RANK}: Waiting for all {NNODES} nodes to join the cluster...")
    while len(ray.nodes()) < NNODES:
        print(f"Current cluster size: {len(ray.nodes())}/{NNODES}. Waiting...")
        time.sleep(5)

    # Only the head node (rank 0) will initialize and run the vLLM engine.
    if RANK == 0:
        print("\n" + "="*40)
        print("Head node is initializing the vLLM Engine.")
        print("="*40 + "\n")

        # tensor and pipeline parallelism
        tp=4
        pp=NNODES

        # Initialize the LLM engine
        llm = LLM(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            distributed_executor_backend="ray",
        )

        prompts = [
            "Give me a short introduction to garden flowers.",
            "The best flower for bees is",
            "Gardening jobs for October include",
        ]
        
        print("\nStarting generation...")
        outputs = llm.generate(prompts)
        print("Generation complete.\n")

        # Print the outputs
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated: {generated_text!r}")
            
        print("\n\nSuccessfully ran distributed vLLM inference.")
        print("Shutting down in 30 seconds.")
        time.sleep(30)

        # create empty file as shutdown signal
        open(".shutdown_signal", "w").close()
        
    else:
        # loop until shutdown file is created
        while not os.path.exists(".shutdown_signal"):
            time.sleep(5)

        os.remove(".shutdown_signal")
        

if __name__ == "__main__":
    monitor =  ZeusMonitor()
    monitor.begin_window("main")
    main()
    energy_usage = monitor.end_window("main")
    
    with open(f"log-energy-usage-{RANK}.json", "w") as f:
        json.dump(energy_usage.__dict__, f)

    print(f"Energy used: {energy_usage.total_energy} in {energy_usage.time}s")
    print("Done!")
