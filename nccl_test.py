import os
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ["SLURM_PROCID"])        # global rank
    world_size = int(os.environ["SLURM_NTASKS"])  # total tasks
    local_rank = int(os.environ["SLURM_LOCALID"]) # local rank on this node
    
    # Slurm provides these
    #rank = int(os.environ["RANK"])
    #world_size = int(os.environ["WORLD_SIZE"])
    #local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    # Init NCCL backend
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    # Each rank starts with its own value
    tensor = torch.ones(1, device="cuda") * (rank + 1)

    # All-reduce (sum across all ranks)
    dist.all_reduce(tensor)

    print(f"Rank {rank}: tensor={tensor.item()} (world_size={world_size})")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

