#!/bin/bash

#SBATCH -C cpu
#SBATCH -t 00:05:00
#SBATCH -J qgtagging
#SBATCH --qos=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --account=atlas
#SBATCH --mem=50G
#SBATCH --output=./slurm-%j.out
#SBATCH --error=./slurm-%j.err
 
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_EffectiveNP_Detector1__1up