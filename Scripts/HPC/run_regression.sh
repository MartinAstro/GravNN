# args = (SH degree, hoppers, acceleration noise, position noise)
sbatch /projects/joma5012/GravNN/Scripts/HPC/run_harmonic_eros_regression.sh 4 False 0.0 0
sbatch /projects/joma5012/GravNN/Scripts/HPC/run_harmonic_eros_regression.sh 4 False 0.1 1
sbatch /projects/joma5012/GravNN/Scripts/HPC/run_harmonic_eros_regression.sh 16 False 0.0 0
sbatch /projects/joma5012/GravNN/Scripts/HPC/run_harmonic_eros_regression.sh 16 False 0.1 1

sbatch /projects/joma5012/GravNN/Scripts/HPC/run_harmonic_eros_regression.sh 4 True 0.0 0
sbatch /projects/joma5012/GravNN/Scripts/HPC/run_harmonic_eros_regression.sh 4 True 0.1 1
sbatch /projects/joma5012/GravNN/Scripts/HPC/run_harmonic_eros_regression.sh 16 True 0.0 0
sbatch /projects/joma5012/GravNN/Scripts/HPC/run_harmonic_eros_regression.sh 16 True 0.1 1

# args = (hoppers, acceleration noise, position noise)
sbatch /projects/joma5012/GravNN/Scripts/HPC/run_network_eros_regression.sh False 0.0 0
sbatch /projects/joma5012/GravNN/Scripts/HPC/run_network_eros_regression.sh False 0.1 1
sbatch /projects/joma5012/GravNN/Scripts/HPC/run_network_eros_regression.sh True 0.0 0
sbatch /projects/joma5012/GravNN/Scripts/HPC/run_network_eros_regression.sh True 0.1 1

# evaluate
sbatch /projects/joma5012/GravNN/Scripts/HPC/run_evaluation.sh
