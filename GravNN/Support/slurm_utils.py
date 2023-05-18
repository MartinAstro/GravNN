import os
import multiprocessing as mp
def get_available_cores():
    try:
        int(os.environ["SLURM_JOB_NUM_NODES"])
        cores_per_nodes = int(os.environ["SLURM_JOB_CPUS_PER_NODE"].split("(")[0])
        processes = cores_per_nodes
    except Exception:
        processes = mp.cpu_count()

    print(f"Process Count: {processes}")
    return processes
