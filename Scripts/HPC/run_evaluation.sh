#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=atesting
#SBATCH --output=SlurmFiles/Regression/evaluate-%j.out

module purge

echo "== Load Anaconda =="

module load anaconda

echo "== Activate Env =="

conda activate research

echo "== Run data generation =="

echo "Running orbit search"

srun python /projects/joma5012/GravNN/Scripts/Regression/Eros/evaluate_nn.py
srun python /projects/joma5012/GravNN/Scripts/Regression/Eros/estimate_near_sh_BLLS.py

wait
echo "== End of Job =="
exit 0
