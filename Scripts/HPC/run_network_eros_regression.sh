#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-192
#SBATCH --account=ucb387_asc1
#SBATCH --time=00:15:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --output=SlurmFiles/Regression/regress-nn-%j.out
module purge

echo "== Load Anaconda =="

module load slurm/alpine
module load anaconda

echo "== Activate Env =="

conda activate research

echo "== Run data generation =="

echo "Running orbit search"

srun python /projects/joma5012/GravNN/Scripts/Regression/Eros/estimate_nn.py $SLURM_ARRAY_TASK_ID

wait # Necessary to wait for all processes to finish
echo "== End of Job =="
exit 0
