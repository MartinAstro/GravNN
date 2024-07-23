#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0
#SBATCH --account=ucb387_asc1
#SBATCH --time=03:30:00
#SBATCH --output=SlurmFiles/Comparison/comparison-init-%A-%a.out

# wait until the data is generated

module purge

echo "== Load Anaconda =="

module load slurm/alpine
module load anaconda

echo "== Activate Env =="

conda activate research

echo "== Run data generation =="

srun python /projects/joma5012/GravNN/Scripts/Gen-III/Comparison/train_HPC.py $SLURM_ARRAY_TASK_ID

wait # Necessary to wait for all processes to finish
echo "== End of Job =="
exit 0

# This run script is meant to initialize all of the experiments (particularly the trajectory experiment)
