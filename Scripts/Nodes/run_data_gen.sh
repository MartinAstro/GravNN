#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --partition=amilan
#SBATCH --output=SlurmFiles/DataGen/run-%j.out

module purge

echo "== Load Anaconda =="

module load anaconda

echo "== Activate Env =="

conda activate research

# Installed package in the compile node already
# echo "== Install =="
# cd /projects/joma5012/GravNN/
# conda develop .

echo "== Run data generation =="

# Note the --exclusive flag ensures that only the one script can be run on the node at a time. 
# echo "Running random data"
# srun python /projects/joma5012/GravNN/Scripts/Data/Eros/generate_random_data.py

# echo "Running surface data on node 2..."
# srun python /projects/joma5012/GravNN/Scripts/Data/Eros/generate_surface_data.py

echo "Running surface data on node 2..."
# srun python /projects/joma5012/GravNN/Scripts/Data/Eros/generate_planes_data.py

srun python /projects/joma5012/GravNN/Scripts/Data/generate_eros_data.py

wait # Necessary to wait for all processes to finish
echo "== End of Job =="
exit 0