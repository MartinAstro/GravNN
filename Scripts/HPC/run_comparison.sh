#!/bin/bash
sbatch /projects/joma5012/GravNN/Scripts/HPC/Comparison/run_comparison_PM.sh
sbatch /projects/joma5012/GravNN/Scripts/HPC/Comparison/run_comparison_SH.sh
sbatch /projects/joma5012/GravNN/Scripts/HPC/Comparison/run_comparison_Mascon.sh
sbatch /projects/joma5012/GravNN/Scripts/HPC/Comparison/run_comparison_Polyhedral.sh
sbatch /projects/joma5012/GravNN/Scripts/HPC/Comparison/run_comparison_ELM.sh
sbatch /projects/joma5012/GravNN/Scripts/HPC/Comparison/run_comparison_NN_GPU.sh
