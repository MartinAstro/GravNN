#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --array=0-18
#SBATCH --account=ucb387_asc1
#SBATCH --gres=gpu:1
#SBATCH --partition=aa100
#SBATCH --time=02:30:00
#SBATCH --output=SlurmFiles/Ablation/ablation-%j-%a.out

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

# srun python /projects/joma5012/GravNN/Scripts/Figures/Gen-III/Ablation/hparam_ablation.py $SLURM_ARRAY_TASK_ID

srun python /projects/joma5012/GravNN/Scripts/Figures/Gen-III/Ablation/noise_ablation.py $SLURM_ARRAY_TASK_ID 0 &
srun python /projects/joma5012/GravNN/Scripts/Figures/Gen-III/Ablation/noise_ablation.py $SLURM_ARRAY_TASK_ID 1 &


wait # Necessary to wait for all processes to finish
echo "== End of Job =="
exit 0

# NOTES

# When running hparam ablation, IDX 0-3 take up a large memory footprint if running the two main() together.
# Break them into the first main() and the second main() to avoid this.
# Moreover, the 0-3 are actually faster on the CPU so do that. (3.5s per 10 vs 5.8s per 10).
