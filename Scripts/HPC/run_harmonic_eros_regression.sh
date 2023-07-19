#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=atesting
#SBATCH --output=SlurmFiles/Regression/regress-sh-%j.out

module purge

echo "== Load Anaconda =="

module load anaconda

echo "== Activate Env =="

conda activate research

echo "== Run data generation =="

echo "Running orbit search"

srun python /projects/joma5012/GravNN/Scripts/Regression/Eros/estimate_near_sh_BLLS.py --deg $1 --hoppers $2 --acc_noise $3 --pos_noise $4

wait
echo "== End of Job =="
exit 0
