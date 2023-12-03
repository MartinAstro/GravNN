#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=atesting
#SBATCH --time=00:15:00
#SBATCH --partition=amilan

module purge

echo "== Load Anaconda =="

module load slurm/alpine
module load anaconda

echo "== Activate Env =="

conda activate research

echo "== Run data generation =="

echo "Running orbit search"

srun python /projects/joma5012/GravNN/Scripts/Train/train_ablation_study.py

wait # Necessary to wait for all processes to finish
echo "== End of Job =="
exit 0
