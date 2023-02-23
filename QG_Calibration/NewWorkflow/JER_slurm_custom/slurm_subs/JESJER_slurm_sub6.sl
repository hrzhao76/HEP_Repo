#!/bin/bash

#SBATCH -C cpu
#SBATCH -t 10:00:00
#SBATCH -J qgtagging_JESJER6
#SBATCH --qos=shared
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=10
#SBATCH --account=atlas
#SBATCH --mem=80G
#SBATCH --output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/slurm_subs//slurm-%j.out
#SBATCH --error=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/slurm_logs//slurm-%j.err


/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_Flavor_Response__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_Flavor_Response__1down
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_JER_DataVsMC_MC16__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_JER_DataVsMC_MC16__1down
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_JER_EffectiveNP_1__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_JER_EffectiveNP_1__1down
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_JER_EffectiveNP_10__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_JER_EffectiveNP_10__1down
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_JER_EffectiveNP_11__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_JER_EffectiveNP_11__1down
