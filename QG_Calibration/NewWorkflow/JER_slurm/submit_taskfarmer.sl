#!/bin/sh
#SBATCH -N 2 -c 10
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -C haswell
#SBATCH -A atlas
#SBATCH --mem 110G
#SBATCH --cpus-per-task 8

export THREADS=1

runcommands.sh tasks.txt