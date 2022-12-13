#!/bin/bash
conda_activate_path=/global/cfs/cdirs/atlas/hrzhao/miniconda3/bin/activate
source ${conda_activate_path} ml

period=${1}
tree=nominal

workdir=/global/cfs/projectdirs/atlas/hrzhao/qgcal/BDT_EB3/
outputdir=/global/cfs/projectdirs/atlas/hrzhao/qgcal/BDT_EB3/
outputdir_pythia=${outputdir}/Processed_Samples_Pythia_Nov8/

# inputdir_data=${outputdir_pythia}/period${period} 
inputdir_data=${outputdir_pythia}/period${period}/newbdt

outputdir_data=${outputdir}/Processed_Samples_Data_Oct18/
sampledir=/global/cfs/projectdirs/atlas/hrzhao/qgcal/Samples_New



python -u ${workdir}/make_Histogram_Pythia_syst_JES.py \
       --path ${sampledir}/pythia/pythia${period} --period ${period} \
       --output-path  ${outputdir_pythia}\
       --tree ${tree} --do-BDT \
       --BDT-path /global/cfs/projectdirs/atlas/hrzhao/qgcal/BDT_EB3/models/bdt_model_flat_pt_gridsearchCV.model \
       | tee ${workdir}/log/log.perpare.Pythia.period${period}.newbdt.txt

python -u ${workdir}/make_Histogram_Data_syst.py \
       --path ${sampledir}/data/ --period ${period} \
       --output-path  ${outputdir_data}\
       --reweight-file-path ${inputdir_data} \
       --do-BDT --BDT-path /global/cfs/projectdirs/atlas/hrzhao/qgcal/BDT_EB3/models/bdt_model_flat_pt_gridsearchCV.model \
       | tee ${workdir}/log/log.perpare.Data.period${period}.newbdt.txt
