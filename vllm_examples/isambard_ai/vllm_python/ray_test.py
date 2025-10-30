import time
import ray
import socket
from collections import Counter
from ray.util import placement_group

# Connect to the running cluster
ray.init(address="auto")  # auto connects to the Ray cluster from CLI

# test using 1 GPU per task
@ray.remote(num_gpus=1)
def get_node_ip():
    return socket.gethostbyname(socket.gethostname())

# Launch tasks to test scheduling
num_tasks = 8
futures = [get_node_ip.remote() for _ in range(num_tasks)]
results = ray.get(futures)

time.sleep(10)

print("Nodes:", ray.nodes())
print("Cluster resources:", ray.cluster_resources())
print("Available resources:", ray.available_resources())

while int(ray.available_resources()["GPU"]) < int(ray.cluster_resources()["GPU"]):
    print("Waiting for all resources to become available")
    time.sleep(5)

# test using 4 GPUs per task
@ray.remote(num_gpus=4)
def ping(node_name):
    import socket
    return f"Pong from {node_name} ({socket.gethostname()})"

futures = [ping.remote(f"Task {i}") for i in range(4)]
print(ray.get(futures))

print("Nodes:", ray.nodes())
print("Cluster resources:", ray.cluster_resources())
print("Available resources:", ray.available_resources())

while int(ray.available_resources()["GPU"]) < int(ray.cluster_resources()["GPU"]):
    print("Waiting for all resources to become available")
    time.sleep(5)

# Count tasks per node
node_counts = Counter(results)
print("Tasks ran on the following nodes:")
for node_ip, count in node_counts.items():
    print(f"  Node {node_ip}: {count} task(s)")

# Show Ray nodes info
print("\nRay cluster nodes:")
for node in ray.nodes():
    print(node)

ray.shutdown()

