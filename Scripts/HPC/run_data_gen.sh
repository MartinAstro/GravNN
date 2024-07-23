#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=SlurmFiles/Data/generate-%j.out
#SBATCH --account=ucb387_asc1
#SBATCH --time=03:00:00
module purge

echo "== Load Anaconda =="

module load slurm/alpine
module load anaconda

echo "== Activate Env =="

conda activate research

echo "== Run data generation =="

echo "Running orbit search"

srun python /projects/joma5012/GravNN/Scripts/Data/generate_data.py
wait
echo "== End of Job =="
exit 0
