#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=ucb387_asc1
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=aa100
#SBATCH --output=SlurmFiles/Regression/train-nn-%j.out
#SBATCH --dependency=afterok:2601722
module purge

echo "== Load Anaconda =="

module load slurm/alpine
module load anaconda

module load gcc
module load cudnn/8.6
export PATH=$PATH:/curc/sw/cuda/11.8/bin:/curc/sw/cuda/11.8/lib64
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/curc/sw/cuda/11.8/

echo "== Activate Env =="

conda activate research

echo "== Run data generation =="

echo "Running orbit search"

srun python /projects/joma5012/GravNN/Scripts/Experiments/heterogeneous_eros_experiment.py

wait # Necessary to wait for all processes to finish
echo "== End of Job =="
exit 0
