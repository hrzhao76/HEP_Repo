#!/bin/bash

#SBATCH -C cpu
#SBATCH -t 10:00:00
#SBATCH -J qgtagging_JESJER3
#SBATCH --qos=shared
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=10
#SBATCH --account=atlas
#SBATCH --mem=80G
#SBATCH --output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/slurm_subs//slurm-%j.out
#SBATCH --error=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/slurm_logs//slurm-%j.err


/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_EffectiveNP_R10_6restTerm__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_EffectiveNP_R10_6restTerm__1down
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_EffectiveNP_Statistical1__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_EffectiveNP_Statistical1__1down
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_EffectiveNP_Statistical2__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_EffectiveNP_Statistical2__1down
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_EffectiveNP_Statistical3__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_EffectiveNP_Statistical3__1down
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_EffectiveNP_Statistical4__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_EffectiveNP_Statistical4__1down
