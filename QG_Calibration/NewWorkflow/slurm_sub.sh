#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 02:00:00
#SBATCH -J qgtagging
#SBATCH --qos=regular
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=atlas
#SBATCH --output=./test_slurm/slurm-%j.out
#SBATCH --error=./test_slurm/slurm-%j.err

#OpenMP settings:
export OMP_NUM_THREADS=1

#run the application:
#applications may performance better with --gpu-bind=none instead of --gpu-bind=single:1 
conda activate ml 
srun --cpu_bind=none python -u make_histogram_new.py --output-path ./test_slurm