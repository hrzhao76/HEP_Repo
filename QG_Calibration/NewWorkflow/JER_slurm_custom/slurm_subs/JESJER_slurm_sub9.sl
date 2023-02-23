#!/bin/bash

#SBATCH -C cpu
#SBATCH -t 10:00:00
#SBATCH -J qgtagging_JESJER9
#SBATCH --qos=shared
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=10
#SBATCH --account=atlas
#SBATCH --mem=80G
#SBATCH --output=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/slurm_subs//slurm-%j.out
#SBATCH --error=/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/slurm_logs//slurm-%j.err


/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_LargeR_TopologyUncertainty_V__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_LargeR_TopologyUncertainty_V__1down
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_LargeR_TopologyUncertainty_top__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_LargeR_TopologyUncertainty_top__1down
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_Pileup_OffsetMu__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_Pileup_OffsetMu__1down
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_Pileup_OffsetNPV__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_Pileup_OffsetNPV__1down
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_Pileup_PtTerm__1up
/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/JER_slurm_custom/wrapper.sh syst_JET_Pileup_PtTerm__1down
