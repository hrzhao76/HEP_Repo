#!/bin/bash
conda_activate_path=/global/cfs/cdirs/atlas/hrzhao/miniconda3/bin/activate
source ${conda_activate_path} ml

workdir=/global/cfs/projectdirs/atlas/hrzhao/qgcal/BDT_EB3/
outputdir=/global/cfs/projectdirs/atlas/hrzhao/qgcal/BDT_EB3/perpared_dijets
sampledir=/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_New

period=${1}
tree=nominal

python -u ${workdir}/prepare_training.py --path ${sampledir}/pythia/pythia${period} --period ${period} --output-path ${outputdir}/Processed_Samples_Pythia_Nov8/period${period} --tree ${tree} | tee ${workdir}/log/log.perpare.Pythia.period${period}.txt

# The following used to check if samples are complete. 
# python -u ${workdir}/make_Histogram_Data_syst.py --path ${sampledir}/data/ --period ${period} --output-path ${outputdir}/Processed_Samples_Data_Oct18/ --reweight-file-path ${outputdir}/Processed_Samples_Pythia_Nov8/period${period} | tee ${workdir}/log/log.perpare.Data.period${period}.txt
