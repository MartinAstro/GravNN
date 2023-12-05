#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=32-63
#SBATCH --account=ucb387_asc1
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=aa100
#SBATCH --output=SlurmFiles/Comparison/comparison-%A-%a.out
# --dependency=afterok:3840518_0

module purge

echo "== Load Anaconda =="

module load slurm/alpine
module load anaconda

echo "== Activate Env =="

conda activate research
module load gcc
module load cudnn/8.6
export PATH=$PATH:/curc/sw/cuda/11.8/bin:/curc/sw/cuda/11.8/lib64
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/curc/sw/cuda/11.8/
echo "== Run data generation =="

srun python /projects/joma5012/GravNN/Scripts/Comparison/train_HPC.py $SLURM_ARRAY_TASK_ID

wait # Necessary to wait for all processes to finish
echo "== End of Job =="
exit 0
