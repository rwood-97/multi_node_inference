import json
import os

NNODES=int(os.environ["SLURM_NNODES"])

total_energy_usage = 0
for i in range(NNODES):
    with open(f"log-energy-usage-{i}.json") as f:
        e = json.load(f)

    total_energy_usage+=sum(e["gpu_energy"].values())

print(f"Total energy usage: {total_energy_usage}")
