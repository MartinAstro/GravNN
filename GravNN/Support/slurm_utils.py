import multiprocessing as mp
import os


def get_available_cores():
    try:
        int(os.environ["SLURM_JOB_NUM_NODES"])
        cores_per_nodes = int(os.environ["SLURM_JOB_CPUS_PER_NODE"].split("(")[0])
        processes = cores_per_nodes
    except Exception:
        processes = mp.cpu_count()

    print(f"Process Count: {processes}")
    return processes


def print_slurm_info():
    try:
        num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        cores_per_nodes = int(os.environ["SLURM_JOB_CPUS_PER_NODE"].split("(")[0])
        num_threads = os.cpu_count()

        print(f"Cores Per Node:{os.environ['SLURM_JOB_CPUS_PER_NODE']}")
        print(f"Num Nodes:{os.environ['SLURM_JOB_NUM_NODES']}")
        print(f"Available Cores:{num_nodes * cores_per_nodes}")
        print(f"Threads per core:{num_threads / (cores_per_nodes * num_nodes)}")
    except Exception:
        pass
