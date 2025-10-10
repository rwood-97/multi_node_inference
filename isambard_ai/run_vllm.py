import os
import time
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, EngineArgs

# For tensor and pipeline parrallelism we need to create placement groups for vLLM.
# Every actor has to have its own placement group.
def scheduling_strategy_fn(tp=int, pp=int):
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 1
        }] * tp*pp,
        strategy="STRICT_PACK",
    )
    return dict(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, 
            placement_group_capture_child_tasks=True
        )
    )

def main():
    """
    Initializes a Ray cluster and runs the vLLM engine on the head node.
    """
    # Get Slurm job information from environment variables
    rank = int(os.environ["SLURM_PROCID"])
    nnodes = int(os.environ["SLURM_NNODES"])
    
    ray.init(address="auto")

    # --- Synchronization Point ---
    # Wait until all expected nodes have joined the Ray cluster.
    print(f"Rank {rank}: Waiting for all {nnodes} nodes to join the cluster...")
    while len(ray.nodes()) < nnodes:
        print(f"Current cluster size: {len(ray.nodes())}/{nnodes}. Waiting...")
        time.sleep(5)

# Only the head node (rank 0) will initialize and run the vLLM engine.
    if rank == 0:
        print("\n" + "="*40)
        print("Head node is initializing the vLLM Engine.")
        print("="*40 + "\n")

        # tensor and pipeline parallelism
        tp=4
        pp=nnodes
        
        # Define scheduling strategy fn
        ss = scheduling_strategy_fn(tp=tp, pp=pp)

        # Initialize the LLM engine
        # This will now correctly create the placement group across the full Ray cluster
        llm = LLM(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            # Your distributed configuration
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
        )

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
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

    else:
        # Worker nodes just keep their Ray process alive until the job ends.
        print("Worker node is idle, waiting for work from the head node.")
        # We need to keep the script running for the Ray worker to stay alive
        while True:
            time.sleep(60)

if __name__ == "__main__":
    main()
