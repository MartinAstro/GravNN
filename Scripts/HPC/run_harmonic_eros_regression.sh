#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-8
#SBATCH --account=ucb387_asc1
#SBATCH --time=01:30:00
#SBATCH --partition=amilan
#SBATCH --output=SlurmFiles/Regression/regress-sh-%j.out

module purge

echo "== Load Anaconda =="

module load anaconda

echo "== Activate Env =="

conda activate research

echo "== Run data generation =="

echo "Running orbit search"

srun python /projects/joma5012/GravNN/Scripts/Regression/Eros/estimate_near_sh_BLLS.py $SLURM_ARRAY_TASK_ID

wait
echo "== End of Job =="
exit 0
